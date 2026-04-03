# 代码库结构与模块说明

---

## 目录结构

```
gigagan-repro/
├── README.md                   # 项目总览、环境搭建、训练命令
├── requirements-lock.txt       # pip freeze 依赖锁定
├── .gitignore
│
├── dox/                        # 文档
│   ├── architecture.md         # 架构深度分析
│   ├── codebase.md             # 本文件：代码结构说明
│   └── modifications.md        # 相对上游的修改记录
│
├── scripts/                    # 辅助脚本
│   └── download_dataset.py     # 数据集下载/解压工具
│
├── data/                       # 数据目录（.gitignore 排除大文件）
│   ├── caltech101_flat/        # 扁平化的 Caltech-101 图像
│   └── caltech101_extract/     # zip 解压中间文件
│
└── gigagan-pytorch/            # fork 的核心库（可编辑安装）
    ├── pyproject.toml          # 包元数据
    ├── setup.py                # setuptools 入口
    └── gigagan_pytorch/        # Python 包
        ├── __init__.py         # 公开 API 导出
        ├── gigagan_pytorch.py  # 核心：G/D/GigaGAN 训练器（~2766 行）
        ├── unet_upsampler.py   # Unet 超分网络（~900 行）
        ├── attend.py           # Flash / 标准 Attention（~111 行）
        ├── data.py             # ImageDataset / TextImageDataset（~114 行）
        ├── distributed.py      # AllGather 分布式工具（~71 行）
        ├── open_clip.py        # OpenCLIP 适配器（~159 行）
        ├── optimizer.py        # AdamW 优化器工厂（~35 行）
        └── version.py          # 版本号 0.3.0
```

---

## 模块详解

### `gigagan_pytorch.py`（核心，~2766 行）

本文件包含几乎所有模型定义和训练逻辑，可分为以下区域：

| 行范围 | 内容 |
|--------|------|
| 1-119 | imports、工具函数（`exists`, `cycle`, `divisible_by` 等） |
| 120-155 | `gradient_penalty()`：R1 梯度惩罚，支持 GradScaler |
| 157-163 | GAN loss 函数（hinge loss） |
| 165-221 | 辅助损失：`aux_matching_loss`、`aux_clip_loss`、`DiffAugment` |
| 222-308 | 基础模块：RMSNorm、Blur、Upsample、Downsample、SqueezeExcite |
| 309-507 | **AdaptiveConv2DMod / AdaptiveConv1DMod**：论文核心创新 |
| 509-656 | 注意力模块：SelfAttention（L2 dist）、CrossAttention |
| 657-805 | TextAttention、FeedForward、Transformer |
| 806-867 | TextEncoder：CLIP + 可学习 Transformer |
| 869-941 | StyleNetwork（EqualLinear MLP）、Noise injection |
| 942-1250 | **Generator**：StyleGAN 范式 + 自适应卷积核 |
| 1252-1430 | **Discriminator** 相关：SimpleDecoder、RandomFixedProjection、VisionAidedDiscriminator |
| 1430-1856 | **Discriminator** 主体：多尺度输入、多尺度输出 |
| 1858-2030 | **GigaGAN** 类 `__init__`：Accelerator 初始化、优化器、参数 |
| 2030-2230 | GigaGAN 辅助方法：`save_sample`、`generate`、`generate_kwargs` 等 |
| 2230-2490 | `train_discriminator_step()`：判别器单步训练 |
| 2491-2610 | `train_generator_step()`：生成器单步训练 |
| 2611-2766 | `forward()`：主训练循环编排 |

### `unet_upsampler.py`（~900 行）

Unet 结构的超分辨率网络，支持视频（VideoGigaGAN 扩展）。关键特性：
- 编码器-解码器 + skip connection
- 每层可含 Self-Attention / Cross-Attention
- Temporal layers 支持视频帧间一致性
- PixelShuffle 上采样

### `attend.py`（~111 行）

轻量 Attention 封装，自动选择 Flash Attention 或标准 SDPA：
- A100 检测 → Flash Attention
- 其他 GPU → Math / Memory-efficient attention
- 回退 → einsum 实现

### `data.py`（~114 行）

数据加载：
- `ImageDataset`：扫描目录下 jpg/png/tiff，要求 >100 张，自动 resize + center crop
- `TextImageDataset`：未实现（占位）
- `MockTextImageDataset`：随机张量 + mock text，用于测试

### `open_clip.py`（~159 行）

OpenCLIP 适配：
- 封装 `open_clip.create_model_and_transforms`
- Hook 机制提取文本编码和视觉编码的多层特征
- `contrastive_loss()`：图文对比损失

### `distributed.py`（~71 行）

分布式工具：
- `all_gather_variable_dim()`：支持不等长 tensor 的 all_gather
- `AllGather` autograd Function：可反向传播的 all_gather

### `optimizer.py`（~35 行）

优化器工厂：
- 自动分离 weight decay 参数（≥2D 参数 vs bias/norm）
- `wd > 0` → AdamW，`wd == 0` → Adam

---

## 关键数据流

### 无条件训练

```
z ~ N(0,1) ──► StyleNetwork(z) ──► style w
                                       │
            init_block (4×4) ──► AdaptiveConv(w) ──► ... layers ... ──► RGB (256×256)
                                                                           │
real_images ──────────────────────────────────────────────────► Discriminator ──► logits
```

### 有条件训练（文本）

```
texts ──► CLIP ──► TextEncoder ──► global_tokens, fine_tokens
                                        │              │
z ──► StyleNetwork(z, global_tokens) ──► w             │
                                         │             │
      init_block ──► AdaptiveConv(w, kernel_select) ──► CrossAttn(fine_tokens) ──► RGB
```
