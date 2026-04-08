import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int = 2, depth: int = 4, base_channels: int = 24):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2 ** i)
            self.downs.append(ConvBlock(ch_in, ch_out))
            if i < depth - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch_in = ch_out

        self.out_channels = ch_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.downs):
            x = block(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        return x


class CrossModalUNet(nn.Module):
    """
    Refactored dual-branch model for:
    - Stage A: contrastive pretraining (mode='pretrain')
    - Stage B: 3-class classification (mode='classify')

    Input format:
    - CT branch:  [B, 2, D, H, W] = [CT, Mask_CT]
    - MRI branch: [B, 2, D, H, W] = [MRI, Mask_MRI]
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        depth: int = 4,
        num_modalities: int = 2,
        mode: str = "pretrain",
        growth_rate: int = 24,
        proj_dim: int = 256,
    ):
        super().__init__()
        if num_modalities != 2:
            raise ValueError("This refactored model supports exactly 2 modalities (CT/MRI)")
        if mode not in {"pretrain", "classify"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.num_classes = num_classes

        self.ct_encoder = Encoder3D(in_channels=in_channels, depth=depth, base_channels=growth_rate)
        self.mri_encoder = Encoder3D(in_channels=in_channels, depth=depth, base_channels=growth_rate)

        feat_dim = self.ct_encoder.out_channels

        self.ct_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        self.mri_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_mode(self, mode: str):
        if mode not in {"pretrain", "classify"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    @staticmethod
    def _global_pool(x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool3d(x, output_size=1)
        return torch.flatten(x, 1)

    def encode(self, ct_input: torch.Tensor, mri_input: torch.Tensor):
        ct_feat_map = self.ct_encoder(ct_input)
        mri_feat_map = self.mri_encoder(mri_input)
        ct_feat = self._global_pool(ct_feat_map)
        mri_feat = self._global_pool(mri_feat_map)
        return ct_feat, mri_feat

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("Model expects inputs=[ct_branch, mri_branch]")

        ct_input, mri_input = inputs
        ct_feat, mri_feat = self.encode(ct_input, mri_input)

        if self.mode == "pretrain":
            return self.ct_proj(ct_feat), self.mri_proj(mri_feat)

        fused = torch.cat([ct_feat, mri_feat], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits
