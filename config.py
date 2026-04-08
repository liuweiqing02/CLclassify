from dataclasses import dataclass, asdict
from typing import Optional, Tuple

PRETRAINING = 0
FINE_TUNING = 1


@dataclass
class Config:
    mode: int

    # Dataset: dongmai only
    data: str = "dongmai"
    image_dir_CT_tr: str = "../data_125/Dataset004_dongmaiCT/imagesTr"
    label_dir_CT_tr: str = "../data_125/Dataset004_dongmaiCT/labelsTr"
    image_dir_MRI_tr: str = "../data_125/Dataset005_dongmaiMR/imagesTr"
    label_dir_MRI_tr: str = "../data_125/Dataset005_dongmaiMR/labelsTr"
    class_label_path: str = "../data_125/class_L_3n.xlsx"

    # Task
    num_classes: int = 3
    in_channels: int = 2  # image + mask for each branch

    # Preprocess
    desired_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    target_size: Tuple[int, int, int] = (128, 128, 128)
    target_size_crop: Tuple[int, int, int] = (96, 96, 96)

    # Split
    seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.0

    # Runtime
    cuda: bool = True
    pin_mem: bool = True
    num_cpu_workers: int = 4

    # Model
    model: str = "CrossModalUNet"  # options: CrossModalUNet, DualBranchResNet3D
    growth_rate: int = 24
    depth: int = 4
    proj_dim: int = 256
    resnet_base_channels: int = 32

    # Training
    batch_size: int = 8
    batch_size_val: int = 2
    nb_epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Paths
    checkpoint_dir: str = "./runs"
    checkpoint_name: str = "dongmai_cls"
    log_dir: str = "./runs"

    # Optional resume / transfer
    pretrained_path: Optional[str] = None
    finetuning_checkpoint_path: Optional[str] = None
    pretraining_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        assert self.mode in {PRETRAINING, FINE_TUNING}, f"Unknown mode: {self.mode}"
        if self.mode == PRETRAINING:
            self.mod = "pretraining"
            self.nb_epochs = 100
            self.lr = 3e-4
            self.checkpoint_name = "dongmai_pretrain"
        else:
            self.mod = "finetuning"
            self.nb_epochs = 150
            self.lr = 1e-4
            self.checkpoint_name = "dongmai_classify"

    def to_dict(self):
        return asdict(self)
