import csv
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class CompatibleModel:
    def __init__(self, net, loss, loader_train, loader_val, loader_test, config, run_dir):
        self.model = net
        self.loss = loss
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.config = config
        self.run_dir = run_dir

        self.device = torch.device("cuda" if (config.cuda and torch.cuda.is_available()) else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found while config.cuda=True")

        self.model = DataParallel(self.model).to(self.device)
        if isinstance(self.loss, torch.nn.Module):
            self.loss = self.loss.to(self.device)
        self.logger = logging.getLogger("CLclassify")

        self.best_val_loss = float("inf")
        self.best_val_f1 = -1.0

    @staticmethod
    def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        return cm

    @staticmethod
    def _classification_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, object]:
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)

        cm = CompatibleModel._confusion_matrix(y_true_np, y_pred_np, num_classes)
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0

        precisions = []
        recalls = []
        f1s = []

        for c in range(num_classes):
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)

            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            "accuracy": acc,
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "precision_per_class": precisions,
            "recall_per_class": recalls,
            "f1_per_class": f1s,
            "confusion_matrix": cm,
        }

    @staticmethod
    def _save_csv_row(path: str, header: List[str], row: List[object]):
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow(row)

    def _save_checkpoint(self, path: str, epoch: int, optimizer, scheduler=None, extra: Dict = None):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": self.config.to_dict(),
        }
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def _load_checkpoint(self, path: str, optimizer=None, scheduler=None) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        return int(ckpt.get("epoch", 0)) + 1

    def _epoch_pretrain(self, loader, train: bool, optimizer=None) -> float:
        self.model.module.set_mode("pretrain")
        self.model.train(train)

        epoch_loss = 0.0
        n_batches = len(loader)

        pbar = tqdm(loader, total=n_batches, desc="Pretrain-Train" if train else "Pretrain-Val")
        for ct, mri, _, _ in pbar:
            ct = ct.to(self.device, non_blocking=True)
            mri = mri.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            z_ct, z_mri = self.model([ct, mri])
            loss = self.loss(z_ct, z_mri)

            if train:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())

        return epoch_loss / max(1, n_batches)

    def pretraining(self):
        os.makedirs(self.run_dir, exist_ok=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, self.config.nb_epochs))

        start_epoch = 1
        if self.config.pretraining_checkpoint_path:
            start_epoch = self._load_checkpoint(self.config.pretraining_checkpoint_path, optimizer, scheduler)

        log_csv = os.path.join(self.run_dir, "pretrain_log.csv")

        for epoch in range(start_epoch, self.config.nb_epochs + 1):
            train_loss = self._epoch_pretrain(self.loader_train, train=True, optimizer=optimizer)
            with torch.no_grad():
                val_loss = self._epoch_pretrain(self.loader_val, train=False)

            scheduler.step()

            self.logger.info(
                f"[Pretrain][Epoch {epoch}/{self.config.nb_epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

            self._save_csv_row(
                log_csv,
                ["epoch", "train_loss", "val_loss"],
                [epoch, train_loss, val_loss],
            )

            last_path = os.path.join(self.run_dir, "last.pth")
            self._save_checkpoint(last_path, epoch, optimizer, scheduler)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.run_dir, "best.pth")
                self._save_checkpoint(best_path, epoch, optimizer, scheduler)
                self.logger.info(f"[Pretrain] New best val_loss={val_loss:.6f} at epoch={epoch}")

    def _epoch_classification(self, loader, train: bool, optimizer=None) -> Tuple[float, Dict[str, object]]:
        self.model.module.set_mode("classify")
        self.model.train(train)

        epoch_loss = 0.0
        n_batches = len(loader)
        y_true, y_pred = [], []

        pbar = tqdm(loader, total=n_batches, desc="Cls-Train" if train else "Cls-Val/Test")
        for ct, mri, y, _ in pbar:
            ct = ct.to(self.device, non_blocking=True)
            mri = mri.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = self.model([ct, mri])
            loss = self.loss(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        metrics = self._classification_metrics(y_true, y_pred, self.config.num_classes)
        return epoch_loss / max(1, n_batches), metrics

    def fine_tuning(self):
        os.makedirs(self.run_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, self.config.nb_epochs))

        # Load pretrained backbone/encoder params if provided
        if self.config.pretrained_path:
            ckpt = torch.load(self.config.pretrained_path, map_location="cpu")
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state, strict=False)
            self.logger.info(f"Loaded pretrained weights from {self.config.pretrained_path}")

        start_epoch = 1
        if self.config.finetuning_checkpoint_path:
            start_epoch = self._load_checkpoint(self.config.finetuning_checkpoint_path, optimizer, scheduler)

        log_csv = os.path.join(self.run_dir, "train_val_log.csv")
        best_epoch = -1

        for epoch in range(start_epoch, self.config.nb_epochs + 1):
            train_loss, train_m = self._epoch_classification(self.loader_train, train=True, optimizer=optimizer)
            with torch.no_grad():
                val_loss, val_m = self._epoch_classification(self.loader_val, train=False)

            scheduler.step()

            self.logger.info(
                f"[Classify][Epoch {epoch}/{self.config.nb_epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_acc={train_m['accuracy']:.4f} val_acc={val_m['accuracy']:.4f} "
                f"train_precision={train_m['precision']:.4f} val_precision={val_m['precision']:.4f} "
                f"train_recall={train_m['recall']:.4f} val_recall={val_m['recall']:.4f} "
                f"train_f1={train_m['f1']:.4f} val_f1={val_m['f1']:.4f}"
            )

            self._save_csv_row(
                log_csv,
                [
                    "epoch",
                    "train_loss", "val_loss",
                    "train_acc", "val_acc",
                    "train_precision", "val_precision",
                    "train_recall", "val_recall",
                    "train_f1", "val_f1",
                ],
                [
                    epoch,
                    train_loss, val_loss,
                    train_m["accuracy"], val_m["accuracy"],
                    train_m["precision"], val_m["precision"],
                    train_m["recall"], val_m["recall"],
                    train_m["f1"], val_m["f1"],
                ],
            )

            last_path = os.path.join(self.run_dir, "last.pth")
            self._save_checkpoint(last_path, epoch, optimizer, scheduler)

            if val_m["f1"] > self.best_val_f1:
                self.best_val_f1 = val_m["f1"]
                best_epoch = epoch
                best_path = os.path.join(self.run_dir, "best.pth")
                self._save_checkpoint(
                    best_path,
                    epoch,
                    optimizer,
                    scheduler,
                    extra={"best_val_metrics": val_m},
                )
                self.logger.info(f"[Classify] New best val_f1={val_m['f1']:.4f} at epoch={epoch}")

        # Final test evaluation with best model
        best_path = os.path.join(self.run_dir, "best.pth")
        if os.path.exists(best_path):
            self._load_checkpoint(best_path)

        with torch.no_grad():
            test_loss, test_m = self._epoch_classification(self.loader_test, train=False)

        cm = test_m["confusion_matrix"]
        cm_path = os.path.join(self.run_dir, "confusion_matrix_test.csv")
        np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

        result = {
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_accuracy": test_m["accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"],
            "test_precision_per_class": test_m["precision_per_class"],
            "test_recall_per_class": test_m["recall_per_class"],
            "test_f1_per_class": test_m["f1_per_class"],
            "confusion_matrix_path": cm_path,
        }

        with open(os.path.join(self.run_dir, "final_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(
            "[Classify][Test] "
            f"loss={test_loss:.6f} acc={test_m['accuracy']:.4f} "
            f"precision={test_m['precision']:.4f} recall={test_m['recall']:.4f} f1={test_m['f1']:.4f}"
        )
