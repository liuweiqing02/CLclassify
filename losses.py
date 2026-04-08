import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModalContrastiveLoss(nn.Module):
    """
    InfoNCE-style cross-modal contrastive loss.

    Positive pairs:
    - CT_i <-> MRI_i (same patient in a batch)

    Negative pairs:
    - CT_i with MRI_j (i != j) and vice versa
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_ct: torch.Tensor, z_mri: torch.Tensor) -> torch.Tensor:
        if z_ct.ndim != 2 or z_mri.ndim != 2:
            raise ValueError("Expected 2D embeddings [B, D]")
        if z_ct.shape[0] != z_mri.shape[0]:
            raise ValueError("Batch size mismatch between CT and MRI embeddings")

        z_ct = F.normalize(z_ct, dim=1)
        z_mri = F.normalize(z_mri, dim=1)

        logits = (z_ct @ z_mri.T) / self.temperature  # [B, B]
        labels = torch.arange(z_ct.size(0), device=z_ct.device)

        # Symmetric contrastive objective
        loss_ct_to_mri = F.cross_entropy(logits, labels)
        loss_mri_to_ct = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ct_to_mri + loss_mri_to_ct)


class ClassificationLoss(nn.Module):
    def __init__(self, class_weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)
