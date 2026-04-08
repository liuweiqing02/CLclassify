# CLclassify

一个独立的双模态 3D 医学图像分类项目，采用两阶段训练：

1. 跨模态对比学习预训练
2. 三分类监督训练

项目输入为双分支结构：

- CT 分支：`[CT, Mask_CT]`
- MRI 分支：`[MRI, Mask_MRI]`

分割标签仅作为辅助输入通道，不作为分割监督目标。

---

## 1. 功能概览

- 双分支 3D 编码器 + 融合分类头
- 两阶段训练（pretraining / finetuning）
- 病人级划分，支持固定 split 复用
- 类别不均衡处理（`class_weight + WeightedRandomSampler`）
- 预处理缓存（增强前缓存，增强现做）
- 支持断点续训

---

## 2. 数据格式要求

你需要准备以下 5 组数据：

1. CT 图像目录
2. CT 标签目录（mask）
3. MRI 图像目录
4. MRI 标签目录（mask）
5. 病人级分类标签表（Excel）

项目会按病人 ID 自动匹配：

`CT图像 ∩ CT标签 ∩ MRI图像 ∩ MRI标签 ∩ 分类标签`

只有交集病人会被用于训练。

> 路径在 `config.py` 中配置。

---

## 3. 训练流程

### 3.1 预训练阶段（Contrastive Pretraining）

- 正样本：同一病人的 CT/MRI 特征对
- 负样本：不同病人的跨模态特征对
- 损失：`DualModalContrastiveLoss`

### 3.2 分类阶段（3-Class Finetuning）

- 加载预训练权重
- 融合后输出三分类 logits
- 损失：`CrossEntropyLoss`（支持 class weight）

---

## 4. 增强与缓存

### 4.1 预训练阶段增强（开启）

- 几何增强：翻转、90度旋转、随机裁切后回缩放
- 几何增强在 CT/MRI 两分支严格同步
- 图像与 mask 同步几何变换
- 像素强度扰动仅作用图像通道（mask 不参与）

### 4.2 分类阶段增强（关闭）

分类训练时关闭随机增强，仅保留固定预处理。

### 4.3 预处理缓存

- 缓存内容：增强前的预处理结果
- 每次取样先 `copy` 再增强，保证增强仍为在线随机
- 典型现象：首个 epoch 较慢，后续 epoch 更快

---

## 5. 配置优先级

当前版本中以下参数只从 `config.py` 读取：

- `batch_size`
- `batch_size_val`
- `nb_epochs`

命令行不会覆盖这三项。

---

## 6. 训练命令

### 6.1 仅预训练

```bash
python main.py \
  --mode pretraining \
  --output_dir ./runs \
  --split_file ./runs/split_seed42.json \
  --lr 3e-4 \
  --temperature 0.1 \
  --seed 42
```

### 6.2 仅分类训练（加载预训练 best）

```bash
python main.py \
  --mode finetuning \
  --pretrained ./runs/pretrain_YYYYMMDD_HHMMSS/best.pth \
  --output_dir ./runs \
  --split_file ./runs/split_seed42.json \
  --lr 1e-4 \
  --seed 42
```

### 6.3 一键两阶段

```bash
python main.py \
  --mode all \
  --output_dir ./runs \
  --split_file ./runs/split_seed42.json \
  --lr 3e-4 \
  --temperature 0.1 \
  --seed 42
```

---

## 7. 断点续训

### 7.1 继续预训练

```bash
python main.py \
  --mode pretraining \
  --output_dir ./runs \
  --pretrain_resume ./runs/pretrain_YYYYMMDD_HHMMSS/last.pth \
  --split_file ./runs/split_seed42.json
```

### 7.2 继续分类训练

```bash
python main.py \
  --mode finetuning \
  --output_dir ./runs \
  --finetune_resume ./runs/classify_YYYYMMDD_HHMMSS/last.pth \
  --split_file ./runs/split_seed42.json
```

---

## 8. 输出文件

每个运行目录典型输出：

- `train.log`
- `last.pth`
- `best.pth`
- `pretrain_log.csv` 或 `train_val_log.csv`
- `experiment_config.json`
- `final_results.json`（分类结束）
- `confusion_matrix_test.csv`（分类结束）

---

## 9. 常见问题

1. 预训练验证 loss 波动大
- 尝试增大 `batch_size`
- 尽量避免每卡 batch 过小

2. 显存不足
- 降低 `batch_size`
- 降低 `depth` 或 `growth_rate`

3. 速度慢
- 首轮慢通常是缓存构建导致，后续会更快
