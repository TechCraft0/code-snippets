#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================
Export BiseNetV2 model to ONNX

-------------------------
Command line examples
-------------------------

1. Basic export (CPU)
--------------------------------------------------
python scripts/export_onnx.py \
    --checkpoint runs/train_001/save_model/model_iter_119000.pth \
    --output bisenetv2_5_cls.onnx

2. Specify input size
--------------------------------------------------
python scripts/export_onnx.py \
    --checkpoint runs/train_001/save_model/model_iter_119000.pth \
    --height 512 \
    --width 960

3. Export using CUDA (if available)
--------------------------------------------------
python scripts/export_onnx.py \
    --checkpoint runs/train_001/save_model/model_iter_119000.pth \
    --device cuda

-------------------------
Python API example
-------------------------

from scripts.export_onnx import build_model, load_checkpoint, export_onnx

device = "cpu"

model = build_model(
    in_channels=3,
    mid_channels=16,
    num_classes=5,
    mode="export",
    device=device
)

load_checkpoint(
    model,
    "runs/train_001/save_model/model_iter_119000.pth",
    device
)

export_onnx(
    model,
    "bisenetv2_5_cls.onnx",
    device,
    input_shape=(1, 3, 512, 960)
)
====================================================
"""

import os
import sys
import argparse
import torch

# =========================================================
# 项目路径设置（允许在 scripts/ 下直接运行）
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from model.bisenetv2 import BisenetV2
from model.bisenetv2_stdc import BisenetV2STDC


# =========================================================
# 构建模型
# =========================================================
def build_model(
    in_channels: int,
    mid_channels: int,
    num_classes: int,
    mode: str,
    device: str
) -> torch.nn.Module:
    """
    Build BiseNetV2 model
    """
    model = BisenetV2STDC(
        in_channels,
        mid_channels,
        num_classes,
        mode=mode
    )
    model.to(device)
    model.eval()
    return model


# =========================================================
# 加载 checkpoint
# =========================================================
def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str
) -> None:
    """
    Load model_state_dict from training checkpoint
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "Invalid checkpoint: 'model_state_dict' not found"
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("[INFO] Checkpoint loaded successfully.")


# =========================================================
# 导出 ONNX
# =========================================================
def export_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    device: str,
    input_shape: tuple,
    opset: int = 12
) -> None:
    """
    Export PyTorch model to ONNX
    """
    dummy_input = torch.randn(*input_shape, device=device)

    print("[INFO] Exporting ONNX...")
    print(f"       Output file : {onnx_path}")
    print(f"       Input shape : {input_shape}")
    print(f"       Opset       : {opset}")

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        verbose=False
    )

    print("[INFO] ONNX export success ✅")


# =========================================================
# 主函数
# =========================================================
def main(args) -> None:
    device = args.device

    model = build_model(
        in_channels=args.in_channels,
        mid_channels=args.mid_channels,
        num_classes=args.num_classes,
        mode="export",
        device=device
    )

    load_checkpoint(
        model=model,
        ckpt_path=args.checkpoint,
        device=device
    )

    export_onnx(
        model=model,
        onnx_path=args.output,
        device=device,
        input_shape=(
            1,
            args.in_channels,
            args.height,
            args.width
        ),
        opset=args.opset
    )


# =========================================================
# CLI 入口
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BiseNetV2 model to ONNX",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # -------- 模型参数 --------
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Number of input channels (default: 3)")
    parser.add_argument("--mid-channels", type=int, default=16,
                        help="Middle channels of BiseNetV2 (default: 16)")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Number of output classes (default: 5)")

    # -------- 输入尺寸 --------
    parser.add_argument("--height", type=int, default=512,
                        help="Input image height (default: 512)")
    parser.add_argument("--width", type=int, default=960,
                        help="Input image width (default: 960)")

    # -------- 路径 --------
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth) containing model_state_dict"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="bisenetv2.onnx",
        help="Output ONNX file path"
    )

    # -------- ONNX --------
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (default: 12)")

    # -------- 设备 --------
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for export")

    args = parser.parse_args()
    main(args)
