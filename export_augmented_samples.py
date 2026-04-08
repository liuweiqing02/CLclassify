import argparse
import json
import os
import random
from typing import List

import nibabel as nib
import numpy as np
import torch

from config import Config, PRETRAINING
from dataset import (
    DongmaiPairDataset,
    build_dongmai_records,
    load_split_file,
    save_split_file,
    split_records_by_ids,
    split_records_patient_level,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_train_records(cfg: Config, split_file: str):
    records, _ = build_dongmai_records(cfg)

    if os.path.exists(split_file):
        split_data = load_split_file(split_file)
        train_records, _ = split_records_by_ids(records, split_data)
    else:
        train_records, val_records = split_records_patient_level(
            records,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
        save_split_file(split_file, train_records, val_records, [])

    return train_records


def save_nifti(arr: np.ndarray, path: str):
    arr = arr.astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, path)


def tensor_stats(x: np.ndarray):
    return {
        "shape": list(x.shape),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def main():
    parser = argparse.ArgumentParser(description="Export augmented training samples for visualization")
    parser.add_argument("--output_dir", type=str, default="./runs/aug_debug")
    parser.add_argument("--split_file", type=str, default="./runs/split_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_patients", type=int, default=5, help="How many training patients to export")
    parser.add_argument("--num_augs_per_patient", type=int, default=3, help="How many augmented draws per patient")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(PRETRAINING)
    cfg.seed = args.seed

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = _prepare_train_records(cfg, args.split_file)
    ds_aug = DongmaiPairDataset(train_records, cfg, training=True)
    ds_raw = DongmaiPairDataset(train_records, cfg, training=False)

    n_patients = max(0, min(args.num_patients, len(train_records) - args.start_index))
    if n_patients <= 0:
        raise RuntimeError("No patients selected. Check --start_index / --num_patients.")

    all_meta: List[dict] = []

    for i in range(n_patients):
        idx = args.start_index + i
        pid = train_records[idx].patient_id

        patient_dir = os.path.join(args.output_dir, f"patient_{pid}")
        os.makedirs(patient_dir, exist_ok=True)

        for k in range(args.num_augs_per_patient):
            ct, mri, label, _ = ds_aug[idx]
            ct_np = ct.numpy()  # [2, D, H, W]
            mri_np = mri.numpy()  # [2, D, H, W]

            aug_dir = os.path.join(patient_dir, f"aug_{k:02d}")
            os.makedirs(aug_dir, exist_ok=True)

            ct_img = ct_np[0]
            ct_mask = ct_np[1]
            mri_img = mri_np[0]
            mri_mask = mri_np[1]

            save_nifti(ct_img, os.path.join(aug_dir, "ct_image.nii.gz"))
            save_nifti(ct_mask, os.path.join(aug_dir, "ct_mask.nii.gz"))
            save_nifti(mri_img, os.path.join(aug_dir, "mri_image.nii.gz"))
            save_nifti(mri_mask, os.path.join(aug_dir, "mri_mask.nii.gz"))

            meta = {
                "patient_id": pid,
                "dataset_index": idx,
                "aug_index": k,
                "class_label": int(label.item()),
                "ct_image": tensor_stats(ct_img),
                "ct_mask": {
                    **tensor_stats(ct_mask),
                    "unique": np.unique(ct_mask).astype(float).tolist()[:10],
                },
                "mri_image": tensor_stats(mri_img),
                "mri_mask": {
                    **tensor_stats(mri_mask),
                    "unique": np.unique(mri_mask).astype(float).tolist()[:10],
                },
            }
            all_meta.append(meta)

        # Also export one non-augmented sample for direct comparison
        ct_raw, mri_raw, label_raw, _ = ds_raw[idx]
        ct_raw_np = ct_raw.numpy()
        mri_raw_np = mri_raw.numpy()

        raw_dir = os.path.join(patient_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        save_nifti(ct_raw_np[0], os.path.join(raw_dir, "ct_image.nii.gz"))
        save_nifti(ct_raw_np[1], os.path.join(raw_dir, "ct_mask.nii.gz"))
        save_nifti(mri_raw_np[0], os.path.join(raw_dir, "mri_image.nii.gz"))
        save_nifti(mri_raw_np[1], os.path.join(raw_dir, "mri_mask.nii.gz"))

        raw_meta = {
            "patient_id": pid,
            "dataset_index": idx,
            "aug_index": "raw",
            "class_label": int(label_raw.item()),
            "ct_image": tensor_stats(ct_raw_np[0]),
            "ct_mask": {
                **tensor_stats(ct_raw_np[1]),
                "unique": np.unique(ct_raw_np[1]).astype(float).tolist()[:10],
            },
            "mri_image": tensor_stats(mri_raw_np[0]),
            "mri_mask": {
                **tensor_stats(mri_raw_np[1]),
                "unique": np.unique(mri_raw_np[1]).astype(float).tolist()[:10],
            },
        }
        all_meta.append(raw_meta)

    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"Export complete: {args.output_dir}")
    print(f"Patients: {n_patients}, augmentations each: {args.num_augs_per_patient}")


if __name__ == "__main__":
    main()
