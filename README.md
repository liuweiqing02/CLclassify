# CLclassify (Dongmai Only, Contrastive Pretrain + 3-Class Classification)

本项目已从原始多模态分割框架重构为：

- 仅使用 `dongmai` 数据集
- 纯三分类任务（不是分割+分类多任务）
- 两阶段训练流程：
  - 阶段A：跨模态对比学习预训练
  - 阶段B：三分类监督训练

分割标签（mask）仅作为输入辅助通道，不作为监督输出目标。

---

## 1. 项目目标与当前实现

### 1.1 已删除内容

- BraTS19 训练主流程
- 分割解码头 / 分割输出 / Dice-IoU-HD95 验证链路
- 原论文创新模块相关实现（E_DAA / TSTCL）在训练主流程中的依赖

### 1.2 保留内容

- 双分支跨模态建模思路（CT分支 + MRI分支）
- 正负样本构造的对比学习预训练能力

### 1.3 当前输入组织（关键）

- CT分支：`[CT, Mask_CT]`（2通道）
- MRI分支：`[MRI, Mask_MRI]`（2通道）

---

## 2. 数据与ID匹配规则

### 2.1 目录约定（`config.py`）

- `../data_125/Dataset004_dongmaiCT/imagesTr`
- `../data_125/Dataset004_dongmaiCT/labelsTr`
- `../data_125/Dataset005_dongmaiMR/imagesTr`
- `../data_125/Dataset005_dongmaiMR/labelsTr`
- `../data_125/class_L_3n.xlsx`

### 2.2 病人ID匹配

在 `dataset.py` 中使用 `canonical_patient_id` 统一清洗文件名/Excel ID，然后取交集：

`CT图像 ∩ CT mask ∩ MRI图像 ∩ MRI mask ∩ 分类标签`

只有交集病人会进入训练。

### 2.3 划分策略

- 病人级划分（防止泄漏）
- 按类别分层划分 train/val/test
- 默认比例：`train:val:test = 0.7:0.2:0.1`（由 `val_ratio` 和 `test_ratio` 决定）

首次运行会生成固定划分文件（默认）：

- `./runs/split_seed42.json`

后续预训练/分类会复用同一 split 文件，保证一致。

---

## 3. 增强与缓存机制

### 3.1 预训练阶段增强（开启）

`DongmaiPairDataset(training=True)` 时：

- 几何增强：翻转、90度旋转、随机裁切后回缩放
- 几何增强在 CT/MRI 两分支严格同步，且 image/mask 同步
- 强度增强（scale+bias）仅作用图像通道（channel 0）
- mask 通道不参与像素强度扰动

### 3.2 分类阶段增强（关闭）

分类训练时数据集以 `training=False` 构建：

- 不做随机几何增强
- 不做随机像素增强
- 仅做固定预处理

### 3.3 预处理缓存

已启用内存缓存（`dataset.py`）：

- 缓存内容：增强前的预处理结果（每个病人一份）
- 每次取样会先 `copy` 再做增强，避免污染缓存
- 预期现象：首轮相对慢，后续 epoch 明显加速

---

## 4. 模型与损失

### 4.1 模型

文件：`models/CrossModalUNet.py`

- 双分支 3D 编码器（CT、MRI）
- `mode='pretrain'`：输出两分支投影向量（默认 256 维）
- `mode='classify'`：融合后输出三分类 logits

### 4.2 预训练损失

文件：`losses.py`

- `DualModalContrastiveLoss`（InfoNCE 对称形式）
- 正样本：同病人 CT-MRI
- 负样本：批内不同病人跨模态对

### 4.3 分类损失与不均衡处理

- 分类损失：`CrossEntropyLoss`
- 已启用类别不均衡策略：
  - `class_weight`（按训练集类别频次反比）
  - `WeightedRandomSampler`

---

## 5. 配置优先级（重要）

当前版本中这些参数**只从 `config.py` 读取**：

- `batch_size`
- `batch_size_val`
- `nb_epochs`

命令行不再提供以上参数，避免“命令行覆盖 config”造成混淆。

---

## 6. 训练命令

下面命令默认使用：

- 固定 split：`./runs/split_seed42.json`
- `config.py` 中的 batch size / epochs

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

### 6.2 仅分类训练（加载预训练）

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

说明：

- 会在该 checkpoint 所在目录继续写日志和权重
- 不会新建时间戳目录

### 7.2 继续分类训练

```bash
python main.py \
  --mode finetuning \
  --output_dir ./runs \
  --finetune_resume ./runs/classify_YYYYMMDD_HHMMSS/last.pth \
  --split_file ./runs/split_seed42.json
```

---

## 8. 训练输出

每个 run 目录下典型文件：

- `train.log`：控制台+文件日志
- `last.pth`：最新checkpoint
- `best.pth`：最佳checkpoint
- `pretrain_log.csv`（预训练）或 `train_val_log.csv`（分类）
- `experiment_config.json`：完整实验配置快照
- `final_results.json`（分类结束后）
- `confusion_matrix_test.csv`（分类结束后）

---

## 9. 快速排错建议

1. 预训练 val loss 不稳定：
- 优先增大 `config.py` 的 `batch_size`
- 保证单卡或每卡 batch 不太小（BatchNorm 稳定性）

2. 显存不足：
- 降 `batch_size`
- 降 `depth` 或 `growth_rate`

3. 训练慢：
- 首轮慢是缓存构建现象，后续 epoch 会更快

