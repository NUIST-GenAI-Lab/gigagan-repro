# 相对上游的修改记录

> 上游：[lucidrains/gigagan-pytorch](https://github.com/lucidrains/gigagan-pytorch) v0.3.0
> 本仓库通过 `pip install -e ./gigagan-pytorch` 可编辑安装，所有修改直接生效。

---

## 修改一览

### 1. 梯度裁剪（Gradient Clipping）

**文件**：`gigagan_pytorch/gigagan_pytorch.py`

**问题**：原版无梯度裁剪，fp16 训练在 step 20 即出现全 NaN（梯度 overflow），fp32 训练 loss 波动剧烈（G loss 峰值达百万级）。

**修改**：
- `GigaGAN.__init__` 新增 `max_grad_norm` 参数（默认 `None`，传入正数即启用）
- 在 D / VD / G 三个优化器的 `step()` 前各加入：
  ```python
  if self.max_grad_norm is not None:
      self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)
  ```

**位置**：
- D optimizer step 前（`train_discriminator_step` 末尾）
- VD optimizer step 前
- G optimizer step 前（`train_generator_step` 末尾）

---

### 2. DDP 同步修复

**文件**：`gigagan_pytorch/gigagan_pytorch.py`

**问题**：多 GPU 训练时，`save_sample()` 仅在主进程执行（保存 checkpoint + 生成样本图），其他 rank 直接进入下一训练步。DDP 要求所有 rank 在每次 `backward()` 时同步梯度，步进不一致导致死锁（表现为某张 GPU 利用率长期为 0）。

**修改**：
- 在 `save_sample()` 调用后添加 `self.accelerator.wait_for_everyone()`
- 移除 `is_first_step` 触发的首步存盘（避免训练开始即长时间 I/O 阻塞）

---

### 3. 文档与配置清理

从 fork 中删除了与复现无关的文件：
- `README.md`（上游使用文档，被本仓库 README 取代）
- `LICENSE`（MIT，保留在 git 历史中）
- `.pre-commit-config.yaml`
- `.github/workflows/python-publish.yml`
- `gigagan-architecture.png`、`gigagan-sample.png`
- `gigagan_pytorch.egg-info/`

---

## 未修改但需注意的已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| fp16 训练 NaN | 回避 | 使用 fp32 + `--no-amp`；原版 GradScaler 无法完全防止 GAN 的 fp16 overflow |
| `TextImageDataset` 未实现 | 上游原样 | `raise NotImplementedError`，仅支持无条件训练 |
| `find_unused_parameters=True` 警告 | 上游原样 | DDP 构造参数，性能影响微小 |
