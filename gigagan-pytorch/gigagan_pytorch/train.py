"""
无条件 GigaGAN 训练入口。

用法：
  python -m gigagan_pytorch.train --data_folder data/caltech101_flat --steps 100
  accelerate launch -m gigagan_pytorch.train --data_folder data/caltech101_flat ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gigagan_pytorch import GigaGAN, ImageDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train unconditional GigaGAN")
    p.add_argument("--data_folder", type=str, required=True, help="图像目录（>100 张）")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--grad_accum_every", type=int, default=8)
    p.add_argument("--model_folder", type=str, default="./gigagan-models")
    p.add_argument("--results_folder", type=str, default="./gigagan-results")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixed_precision_type", type=str, default="fp16", choices=("fp16", "bf16"))
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--apply_gradient_penalty_every", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪范数，0 关闭")
    p.add_argument("--train_only", action="store_true", help="训练后不做采样")
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    if not data_folder.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_folder}")

    dataset = ImageDataset(
        folder=str(data_folder),
        image_size=args.image_size,
        convert_image_to="RGB",
    )
    dataloader = dataset.get_dataloader(batch_size=args.batch_size)

    gan = GigaGAN(
        generator=dict(
            dim_capacity=8,
            style_network=dict(dim=64, depth=4),
            image_size=args.image_size,
            dim_max=512,
            num_skip_layers_excite=4,
            unconditional=True,
        ),
        discriminator=dict(
            dim_capacity=16,
            dim_max=512,
            image_size=args.image_size,
            num_skip_layers_excite=4,
            unconditional=True,
        ),
        learning_rate=args.learning_rate,
        apply_gradient_penalty_every=args.apply_gradient_penalty_every,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        model_folder=args.model_folder,
        results_folder=args.results_folder,
        create_ema_generator_at_init=args.ema,
    )

    try:
        gan.set_dataloader(dataloader)
        gan(steps=args.steps, grad_accum_every=args.grad_accum_every)

        if not args.train_only and gan.is_main:
            gan.print("Generating samples...")
            with torch.inference_mode():
                out = gan.generate(batch_size=4)
            gan.print(f"Output shape: {tuple(out.shape)}")
    finally:
        gan.accelerator.end_training()


if __name__ == "__main__":
    main()
