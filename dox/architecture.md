# GigaGAN 架构分析

> 基于 [lucidrains/gigagan-pytorch](https://github.com/lucidrains/gigagan-pytorch) v0.3.0，论文：[Scaling up GANs for Text-to-Image Synthesis (2023)](https://arxiv.org/abs/2303.05511)

---

## 1. 整体结构

GigaGAN 包含三个核心网络和一个训练编排器：

| 组件 | 类名 | 文件 | 作用 |
|------|------|------|------|
| 生成器 | `Generator` | `gigagan_pytorch.py` | 从噪声 z 生成图像（StyleGAN 范式） |
| Unet 超分生成器 | `UnetUpsampler` | `unet_upsampler.py` | 低分辨率 → 高分辨率上采样 |
| 判别器 | `Discriminator` | `gigagan_pytorch.py` | 多尺度判别 + 辅助重建损失 |
| 视觉辅助判别器 | `VisionAidedDiscriminator` | `gigagan_pytorch.py` | 借助 CLIP 视觉编码做辅助判别 |
| 训练编排器 | `GigaGAN` | `gigagan_pytorch.py` | 管理 G/D 交替训练、Accelerate DDP、保存/采样 |

```
noise (z)
   │
   ▼
StyleNetwork ──► style vector (w)
   │
   ▼
Generator / UnetUpsampler ──► fake image
   │                              │
   │                              ▼
   │                     Discriminator (多尺度)
   │                              │
   ▼                              ▼
(text conditioning)      VisionAidedDiscriminator (CLIP)
```

---

## 2. Generator（`gigagan_pytorch.py` L947-L1250）

### 2.1 核心创新：自适应卷积核选择

论文主要贡献是 **Adaptive Convolution Kernel Selection**——在每一层使用 N 个卷积核，根据 style vector 通过 softmax 加权选择：

```python
class AdaptiveConv2DMod:
    weights: (num_conv_kernels, dim_out, dim_in, k, k)

    def forward(fmap, mod, kernel_mod):
        # 1. kernel_mod → softmax → 选择核权重
        kernel_attn = kernel_mod.softmax(dim=-1)
        weights = (weights * kernel_attn).sum(dim=0)  # 加权求和
        # 2. StyleGAN2 风格的 modulation / demodulation
        weights = weights * (mod + 1)
        # 3. grouped conv2d
```

### 2.2 网络结构

- **初始块**：可学习常量 4×4 → AdaptiveConv
- **主干**：`num_layers = log2(image_size) - 1` 层，每层包含：
  - Skip Layer Excitation（来自 Lightweight GAN）
  - 2× AdaptiveConv + Noise injection + LeakyReLU（ResNet block 风格）
  - 可选 Self-Attention（L2 distance，非 dot product）
  - 可选 Cross-Attention（文本条件）
  - Upsample（bilinear + blur 或 PixelShuffle）
  - to_rgb 逐层累加
- **Style Network**：MLP（EqualLinear + LeakyReLU），接收 z + 可选 text latent

### 2.3 多尺度 RGB 输出

每层产生一个 RGB 输出并逐层累加上采样，供判别器多尺度输入使用。`return_all_rgbs=True` 时返回所有中间 RGB。

---

## 3. Discriminator（`gigagan_pytorch.py` L1252-L1856）

### 3.1 多尺度输入

接收不同分辨率的 RGB 输入（来自 Generator 各层输出），在内部下采样过程中融合：

```
image (full res) ──► downsample stages
                        ├─ 注入 rgb@128
                        ├─ 注入 rgb@64
                        └─ 注入 rgb@32
```

### 3.2 辅助重建损失（SSL）

来自 Lightweight GAN 的技巧：从判别器的中间特征图重建原始图像，提升训练稳定性。使用 `SimpleDecoder` 实现，支持 patch 级随机采样以提升效率。

### 3.3 判别器组件

- 各层：Conv → LeakyReLU → (Self-Attention) → (Cross-Attention) → Downsample
- 最终：MiniBatch StdDev → Conv → Flatten → Linear → logit

---

## 4. VisionAidedDiscriminator（`gigagan_pytorch.py` L1339-L1430）

利用冻结的 CLIP 视觉编码器的多层特征做辅助判别：

1. 将图像送入 CLIP visual encoder，取多层 hook 输出
2. 对每层特征：RandomFixedProjection → AdaptiveConv（条件化文本）→ Conv → logit
3. 多层 logits 分别贡献 hinge loss

---

## 5. 损失函数

| 损失 | 日志标签 | 来源 |
|------|----------|------|
| Generator hinge loss | `G` | `fake_logits.mean()` |
| Multiscale Generator loss | `MSG` | 多尺度 fake logits 之和 |
| Discriminator hinge loss | `D` | `relu(1+real) + relu(1-fake)` |
| Multiscale Discriminator loss | `MSD` | 多尺度 real/fake logits |
| Gradient Penalty | `GP` | R1 penalty，对 real/fake images 的二阶梯度 |
| Auxiliary Reconstruction | `SSL` | 判别器中间特征重建原图的 MSE |
| Vision-aided D loss | `VD` | CLIP 特征层 hinge loss |
| Vision-aided G loss | `VG` | 对应的 G 端 loss |
| Contrastive Loss | `CL` | CLIP text-image 对比损失 |
| Matching Awareness Loss | `MAL` | 不匹配文本-图像对的辅助损失 |

### 健康指标

- `G`, `MSG`, `D`, `MSD` 应在 0~10 范围，1000 步后仍为三位数说明异常
- `GP`, `SSL` 应趋向 0
- `GP` 偶尔尖峰是正常的

---

## 6. 训练流程（`GigaGAN.forward`，L2650-L2766）

```
for step in range(steps):
    # ─── 判别器训练 ───
    D.zero_grad()
    for _ in range(grad_accum_every):
        G.eval() → 生成 fake（no_grad）
        D.train() → 计算 real/fake logits
        hinge_loss + multiscale_loss + gradient_penalty + VD_loss + matching_loss
        backward(total_loss / grad_accum_every)
    clip_grad_norm_(D)  # 本仓库新增
    D_opt.step()

    # ─── 生成器训练 ───
    G.zero_grad()
    for _ in range(grad_accum_every):
        G.train() → 生成 fake
        D.eval() → 计算 fake logits（生成器希望降低之）
        generator_hinge_loss + multiscale_loss + VG_loss
    clip_grad_norm_(G)  # 本仓库新增
    G_opt.step()

    # ─── EMA / 保存 / 日志 ───
    wait_for_everyone()
    if main_process: update_ema, save_sample
    wait_for_everyone()
```
