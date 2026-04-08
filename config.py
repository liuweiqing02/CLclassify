from dataclasses import dataclass, asdict
from typing import Optional, Tuple

PRETRAINING = 0
FINE_TUNING = 1


@dataclass
class Config:
    mode: int

    # Dataset paths
    data: str = "dataset"
    image_dir_CT_tr: str = "../data_125/Dataset004_dongmaiCT/imagesTr"
    label_dir_CT_tr: str = "../data_125/Dataset004_dongmaiCT/labelsTr"
    image_dir_MRI_tr: str = "../data_125/Dataset005_dongmaiMR/imagesTr"
    label_dir_MRI_tr: str = "../data_125/Dataset005_dongmaiMR/labelsTr"
    class_label_path: str = "../data_125/class_L_3n.xlsx"

    # Task
    num_classes: int = 3
    in_channels: int = 2

    # Preprocess
    desired_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    target_size: Tuple[int, int, int] = (128, 128, 128)
    target_size_crop: Tuple[int, int, int] = (96, 96, 96)

    # Split
    seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    split_file: Optional[str] = None

    # Runtime
    cuda: bool = True
    pin_mem: bool = True
    num_cpu_workers: int = 4

    # Model
    model: str = "DualBranchResNet3D"  # CrossModalUNet | DualBranchResNet3D
    growth_rate: int = 24
    depth: int = 4
    proj_dim: int = 256
    resnet_base_channels: int = 32

    # Shared training defaults
    batch_size: int = 8
    batch_size_val: int = 4
    weight_decay: float = 1e-4

    # Stage-specific recommended hyperparameters
    pretrain_epochs: int = 80
    pretrain_lr: float = 1e-4
    pretrain_temperature: float = 0.07

    finetune_epochs: int = 120
    finetune_lr: float = 1e-4

    # Effective fields used by runner (auto-filled by mode)
    nb_epochs: int = 80
    lr: float = 1e-4

    # Paths
    checkpoint_dir: str = "./runs"
    checkpoint_name: str = "run"
    log_dir: str = "./runs"

    # Resume / transfer
    pretrained_path: Optional[str] = None
    finetuning_checkpoint_path: Optional[str] = None
    pretraining_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        assert self.mode in {PRETRAINING, FINE_TUNING}, f"Unknown mode: {self.mode}"

        if self.mode == PRETRAINING:
            self.mod = "pretraining"
            self.nb_epochs = self.pretrain_epochs
            self.lr = self.pretrain_lr
            self.checkpoint_name = "pretrain"
        else:
            self.mod = "finetuning"
            self.nb_epochs = self.finetune_epochs
            self.lr = self.finetune_lr
            self.checkpoint_name = "classify"

        if self.split_file is None:
            self.split_file = f"{self.checkpoint_dir}/split_seed{self.seed}.json"

    def to_dict(self):
        return asdict(self)
