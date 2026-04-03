# GigaGAN 复现

基于 [lucidrains/gigagan-pytorch](https://github.com/lucidrains/gigagan-pytorch) v0.3.0 的 GigaGAN 复现项目。

论文：[Scaling up GANs for Text-to-Image Synthesis (Kang et al., 2023)](https://arxiv.org/abs/2303.05511)

---

## 目录结构

```
gigagan-repro/
├── gigagan-pytorch/        # fork 的核心库（可编辑安装）
│   └── gigagan_pytorch/    # 模型定义 + 训练逻辑
├── scripts/                # 辅助脚本
│   └── download_dataset.py # 数据集下载/解压
├── data/                   # 数据目录（已 gitignore）
├── dox/                    # 文档
│   ├── architecture.md     # 架构深度分析
│   ├── codebase.md         # 代码结构与模块说明
│   └── modifications.md    # 相对上游的修改记录
├── requirements-lock.txt   # 依赖锁定
└── .gitignore
```

---

## 环境搭建

```bash
# Miniconda（若未安装）
# https://docs.conda.io/en/latest/miniconda.html

conda create -n gigagan python=3.10 -y
conda activate gigagan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ./gigagan-pytorch
pip install accelerate
```

已有环境直接激活：

```bash
source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate gigagan
```

---

## 数据准备

`ImageDataset` 要求目录内 **>100 张** 图像（jpg/png）。

### 从本地 zip 解压（推荐）

```bash
python scripts/download_dataset.py \
  --from-zip /path/to/caltech-101.zip \
  --output data/caltech101_flat
```

### 在线下载

```bash
python scripts/download_dataset.py --dataset caltech101 --output data/caltech101_flat
```

---

## 训练

### 多卡训练（推荐）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes=8 --mixed_precision=no \
  -m gigagan_pytorch.train \
  --data_folder data/caltech101_flat \
  --batch_size 2 --steps 1000 --grad_accum_every 4 \
  --learning_rate 2e-4 --max_grad_norm 1.0 --no-amp
```

> `--num_processes` 须与可见 GPU 数一致。

### 单卡训练

```bash
python -m gigagan_pytorch.train \
  --data_folder data/caltech101_flat \
  --steps 100 --grad_accum_every 4 --no-amp
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_folder` | 必填 | 图像目录 |
| `--image_size` | 256 | 生成图像分辨率 |
| `--batch_size` | 1 | 每 GPU batch size |
| `--steps` | 100 | 训练步数 |
| `--grad_accum_every` | 8 | 梯度累积次数（等效 batch = batch_size × num_gpu × grad_accum） |
| `--learning_rate` | 1e-4 | G 与 D 学习率 |
| `--max_grad_norm` | 1.0 | 梯度裁剪范数（0 关闭） |
| `--no-amp` / `--amp` | 默认开 | 建议 `--no-amp`，fp16 易 NaN |
| `--no-ema` / `--ema` | 默认关 | EMA 生成器 |

---

## Loss 指标说明

| 标签 | 含义 | 健康范围 |
|------|------|----------|
| `G` | Generator hinge loss | 0~10 |
| `MSG` | 多尺度 Generator loss | 0~10 |
| `D` | Discriminator hinge loss | 0~10 |
| `MSD` | 多尺度 Discriminator loss | 0~10 |
| `GP` | 梯度惩罚 | 趋向 0 |
| `SSL` | 辅助重建损失 | 趋向 0 |
| `VD` / `VG` | Vision-aided 判别/生成 | 无条件为 0 |
| `CL` | CLIP 对比损失 | 无条件为 0 |
| `MAL` | Matching Awareness | 无条件为 0 |

---

## 已知问题与解决方案

### fp16 梯度爆炸

GAN 在 fp16 混合精度下极易 NaN（本项目实测 step 20 即全 NaN），原因是 hinge loss + gradient penalty 的二阶梯度超出 fp16 动态范围。

**解决**：使用 fp32（`--mixed_precision=no --no-amp`）+ 梯度裁剪（`--max_grad_norm 1.0`）。

### 多卡死锁

上游 `save_sample()` 仅主进程执行，导致 DDP 步进不一致。本仓库已在 `save_sample` 后添加 `wait_for_everyone()` 同步。

---

## 文档

详细分析见 [`dox/`](dox/) 目录：

- [架构分析](dox/architecture.md) — 网络结构、数据流、损失函数
- [代码结构](dox/codebase.md) — 文件组织、模块职责、行号索引
- [修改记录](dox/modifications.md) — 相对上游的所有改动

---

## 参考

- 上游仓库：https://github.com/lucidrains/gigagan-pytorch
- 论文项目页：https://mingukkang.github.io/GigaGAN/
- 论文 arXiv：https://arxiv.org/abs/2303.05511
