#!/usr/bin/env python3
"""
将公开图像分类数据集导出为扁平目录（jpg/png），供 gigagan_pytorch.ImageDataset 使用。

方式一（推荐，离线）：本地已有官方打包 `caltech-101.zip`（内含 `101_ObjectCategories.tar.gz`）
  python scripts/download_dataset.py --from-zip /path/to/caltech-101.zip --output data/caltech101_flat

方式二：由 Torchvision 在线下载（需可访问外网）
  python scripts/download_dataset.py --dataset caltech101 --output data/caltech101_flat
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare image folder for GigaGAN training")
    p.add_argument(
        "--dataset",
        type=str,
        default="caltech101",
        choices=("caltech101",),
        help="与 --from-zip 配合时表示数据集类型；在线模式时为 caltech101",
    )
    p.add_argument(
        "--from-zip",
        type=Path,
        default=None,
        help="本地 caltech-101.zip 路径（例如 ~/datasets/caltech-101.zip），设置后不再走在线下载",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/caltech101_flat"),
        help="导出图像目录（扁平，无子目录）",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/caltech101_extract"),
        help="解压 zip / tar.gz 的工作目录（可删，仅中间文件）",
    )
    p.add_argument(
        "--torchvision-root",
        type=Path,
        default=Path("data/torchvision"),
        help="torchvision 在线下载时的缓存根目录",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多导出张数；0 表示全部（需 >100 以满足 ImageDataset）",
    )
    p.add_argument(
        "--skip-if-count",
        type=int,
        default=101,
        help="若输出目录中已有不少于该数量的图像文件则跳过",
    )
    return p.parse_args()


def count_flat_images(folder: Path) -> int:
    n = 0
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        n += len(list(folder.glob(pat)))
    return n


def iter_image_files(obj_categories: Path):
    """遍历 101_ObjectCategories 下所有图像，跳过 macOS 垃圾与明显非图文件。"""
    for p in sorted(obj_categories.rglob("*")):
        if not p.is_file():
            continue
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if p.name == "tmp" and p.suffix == "":
            continue
        yield p


def export_caltech101_from_zip(
    zip_path: Path,
    output: Path,
    work_dir: Path,
    max_images: int,
) -> int:
    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到 zip：{zip_path}")

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)

    tgz_list = list(work_dir.rglob("101_ObjectCategories.tar.gz"))
    if not tgz_list:
        raise FileNotFoundError(
            f"在 {work_dir} 内未找到 101_ObjectCategories.tar.gz，请确认 zip 为官方 Caltech-101 结构"
        )
    tgz = tgz_list[0]
    extract_root = tgz.parent
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(extract_root)

    obj_dirs = list(work_dir.rglob("101_ObjectCategories"))
    obj_cat = None
    for d in obj_dirs:
        if d.is_dir() and (d / "BACKGROUND_Google").exists():
            obj_cat = d
            break
    if obj_cat is None:
        obj_cat = next((d for d in obj_dirs if d.is_dir()), None)
    if obj_cat is None:
        raise FileNotFoundError(f"解压后未找到 101_ObjectCategories 目录（已搜索 {work_dir}）")

    paths = list(iter_image_files(obj_cat))
    if max_images > 0:
        paths = paths[:max_images]

    output.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(tqdm(paths, desc="flatten caltech101")):
        ext = src.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        dst = output / f"{i:06d}{ext}"
        shutil.copy2(src, dst)
    return len(paths)


def export_caltech101_torchvision(output: Path, tv_root: Path, max_images: int) -> int:
    import torchvision.datasets as dset

    ds = dset.Caltech101(root=str(tv_root), download=True)
    n = len(ds)
    if max_images > 0:
        n = min(n, max_images)

    output.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n), desc="export caltech101 (torchvision)"):
        img, _ = ds[i]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output / f"{i:06d}.jpg", quality=95, subsampling=2)
    return n


def main() -> None:
    args = parse_args()
    out = args.output.resolve()
    existing = count_flat_images(out) if out.is_dir() else 0
    if existing >= args.skip_if_count:
        print(f"跳过：{out} 中已有 {existing} 张图像（>= {args.skip_if_count}）")
        return

    if args.from_zip is not None:
        n = export_caltech101_from_zip(
            args.from_zip,
            out,
            args.work_dir.resolve(),
            args.max_images,
        )
    elif args.dataset == "caltech101":
        n = export_caltech101_torchvision(out, args.torchvision_root.resolve(), args.max_images)
    else:
        raise SystemExit(f"未实现的数据集：{args.dataset}")

    if n <= 100:
        raise RuntimeError(
            f"导出仅 {n} 张，gigagan ImageDataset 要求 len(paths) > 100。请去掉 --max-images 或增大上限。"
        )
    print(f"完成：已写入 {n} 张图像到 {out}")


if __name__ == "__main__":
    main()
