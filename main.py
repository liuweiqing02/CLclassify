import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from CompatibleModel import CompatibleModel
from config import Config, FINE_TUNING, PRETRAINING
from dataset import (
    DongmaiPairDataset,
    build_dongmai_records,
    load_split_file,
    save_split_file,
    split_records_by_ids,
    split_records_patient_level,
)
from losses import ClassificationLoss, DualModalContrastiveLoss
from models.CrossModalUNet import CrossModalUNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logger(log_file: str, append: bool = False):
    logger = logging.getLogger("CLclassify")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _get_or_create_split(records, cfg: Config, split_file: str):
    if os.path.exists(split_file):
        split_data = load_split_file(split_file)
        train_records, val_records, test_records = split_records_by_ids(records, split_data)
    else:
        train_records, val_records, test_records = split_records_patient_level(
            records,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
        )
        save_split_file(split_file, train_records, val_records, test_records)
    return train_records, val_records, test_records


def _compute_class_weight_and_sampler(train_set: DongmaiPairDataset, num_classes: int):
    labels = [r.label for r in train_set.records]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0

    class_weight = counts.sum() / (num_classes * counts)
    sample_weights = np.array([class_weight[y] for y in labels], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return torch.tensor(class_weight, dtype=torch.float32), sampler, counts.tolist()


def make_loaders(cfg: Config, stage: str, split_file: str):
    records, label_decode = build_dongmai_records(cfg)
    train_records, val_records, test_records = _get_or_create_split(records, cfg, split_file)

    train_set = DongmaiPairDataset(train_records, cfg, training=(stage == "pretraining"))
    val_set = DongmaiPairDataset(val_records, cfg, training=False)
    test_set = DongmaiPairDataset(test_records, cfg, training=False)

    class_weight = None
    class_counts = None

    if stage == "classification":
        class_weight, train_sampler, class_counts = _compute_class_weight_and_sampler(train_set, cfg.num_classes)
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.num_cpu_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=DongmaiPairDataset.collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_cpu_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=DongmaiPairDataset.collate_fn,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_cpu_workers,
        pin_memory=cfg.pin_mem,
        collate_fn=DongmaiPairDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_cpu_workers,
        pin_memory=cfg.pin_mem,
        collate_fn=DongmaiPairDataset.collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_decode,
        (len(train_set), len(val_set), len(test_set)),
        class_weight,
        class_counts,
    )


def build_model(cfg: Config, mode: str):
    return CrossModalUNet(
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        depth=cfg.depth,
        num_modalities=2,
        mode=mode,
        growth_rate=cfg.growth_rate,
        proj_dim=cfg.proj_dim,
    )


def save_config(run_dir: str, cfg: Config, extra: dict = None):
    payload = cfg.to_dict()
    if extra:
        payload.update(extra)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_pretraining(args, split_file: str):
    cfg = Config(PRETRAINING)
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.checkpoint_dir = args.output_dir
    cfg.log_dir = args.output_dir
    if args.pretrain_resume:
        cfg.pretraining_checkpoint_path = args.pretrain_resume

    if args.pretrain_resume:
        run_dir = os.path.dirname(os.path.abspath(args.pretrain_resume))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.checkpoint_dir, f"pretrain_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = init_logger(os.path.join(run_dir, "train.log"), append=bool(args.pretrain_resume))
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, label_decode, sizes, _, _ = make_loaders(cfg, stage="pretraining", split_file=split_file)
    logger.info(f"Data split (patient-level): train={sizes[0]} val={sizes[1]} test={sizes[2]}")
    logger.info(f"Using fixed split file: {split_file}")
    logger.info(f"Label decode map (model_label -> original_label): {label_decode}")

    save_config(
        run_dir,
        cfg,
        extra={"stage": "pretraining", "split_sizes": sizes, "label_decode": label_decode, "split_file": split_file},
    )

    model = build_model(cfg, mode="pretrain")
    loss = DualModalContrastiveLoss(temperature=args.temperature)

    runner = CompatibleModel(model, loss, train_loader, val_loader, test_loader, cfg, run_dir)
    runner.pretraining()
    return os.path.join(run_dir, "best.pth")


def run_finetuning(args, split_file: str, pretrained_path=None):
    cfg = Config(FINE_TUNING)
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.checkpoint_dir = args.output_dir
    cfg.log_dir = args.output_dir

    cfg.pretrained_path = pretrained_path or args.pretrained
    if args.finetune_resume:
        cfg.finetuning_checkpoint_path = args.finetune_resume

    if args.finetune_resume:
        run_dir = os.path.dirname(os.path.abspath(args.finetune_resume))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.checkpoint_dir, f"classify_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = init_logger(os.path.join(run_dir, "train.log"), append=bool(args.finetune_resume))
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, label_decode, sizes, class_weight, class_counts = make_loaders(
        cfg,
        stage="classification",
        split_file=split_file,
    )
    logger.info(f"Data split (patient-level): train={sizes[0]} val={sizes[1]} test={sizes[2]}")
    logger.info(f"Using fixed split file: {split_file}")
    logger.info(f"Label decode map (model_label -> original_label): {label_decode}")
    logger.info(f"Train class counts (encoded labels): {class_counts}")
    logger.info(f"Class weights for CE: {class_weight.tolist() if class_weight is not None else None}")

    save_config(
        run_dir,
        cfg,
        extra={
            "stage": "classification",
            "split_sizes": sizes,
            "label_decode": label_decode,
            "split_file": split_file,
            "train_class_counts": class_counts,
            "class_weight": class_weight.tolist() if class_weight is not None else None,
        },
    )

    model = build_model(cfg, mode="classify")
    loss = ClassificationLoss(class_weight=class_weight)

    runner = CompatibleModel(model, loss, train_loader, val_loader, test_loader, cfg, run_dir)
    runner.fine_tuning()


def main():
    parser = argparse.ArgumentParser(description="Dongmai dual-branch contrastive pretraining + 3-class classification")
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning", "all"], required=True)
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_file", type=str, default=None, help="Fixed split json path for pretrain/finetune consistency")

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretraining best.pth for finetuning")
    parser.add_argument("--pretrain_resume", type=str, default=None, help="Resume pretraining checkpoint path")
    parser.add_argument("--finetune_resume", type=str, default=None, help="Resume finetuning checkpoint path")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_file = args.split_file or os.path.join(args.output_dir, f"split_seed{args.seed}.json")

    if args.mode == "pretraining":
        run_pretraining(args, split_file=split_file)
    elif args.mode == "finetuning":
        run_finetuning(args, split_file=split_file)
    else:
        best_pretrain = run_pretraining(args, split_file=split_file)
        run_finetuning(args, split_file=split_file, pretrained_path=best_pretrain)


if __name__ == "__main__":
    main()
