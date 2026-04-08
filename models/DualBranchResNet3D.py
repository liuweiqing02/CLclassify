import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 32, blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]

        in_ch = base_channels
        stages = []
        for out_ch, n_blocks, stride in zip(channels, blocks_per_stage, strides):
            layers = [BasicBlock3D(in_ch, out_ch, stride=stride)]
            for _ in range(n_blocks - 1):
                layers.append(BasicBlock3D(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*layers))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        return x


class DualBranchResNet3D(nn.Module):
    """
    Selectable alternative model:
    - mode='pretrain': returns (z_ct, z_mri)
    - mode='classify': returns logits
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        mode: str = "pretrain",
        base_channels: int = 32,
        proj_dim: int = 256,
    ):
        super().__init__()
        if mode not in {"pretrain", "classify"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.ct_encoder = ResNet3DEncoder(in_channels=in_channels, base_channels=base_channels)
        self.mri_encoder = ResNet3DEncoder(in_channels=in_channels, base_channels=base_channels)

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
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def set_mode(self, mode: str):
        if mode not in {"pretrain", "classify"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("Model expects inputs=[ct_branch, mri_branch]")
        ct, mri = inputs

        f_ct = self.ct_encoder(ct)
        f_mri = self.mri_encoder(mri)

        if self.mode == "pretrain":
            return self.ct_proj(f_ct), self.mri_proj(f_mri)

        fused = self.fusion(torch.cat([f_ct, f_mri], dim=1))
        logits = self.classifier(fused)
        return logits

