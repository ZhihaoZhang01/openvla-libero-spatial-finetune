#!/usr/bin/env python3
"""
环境验证：GPU + Flash-Attn + LIBERO + OpenVLA-7B 加载。

用法:
  source scripts/env/activate_openvla.sh
  python run/test_final.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import setup_env, setup_sys_path

setup_sys_path()
setup_env()

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

print("=" * 50)
print("[Step 1] Hardware & Flash Attention")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import flash_attn  # noqa: F401

    print(f"flash-attn: {flash_attn.__version__} OK")
    use_flash = True
except ImportError:
    print("flash-attn: NOT FOUND (will use sdpa)")
    use_flash = False

print("\n[Step 2] MuJoCo & LIBERO")
try:
    import mujoco  # noqa: F401
    from libero.libero import benchmark  # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    print("MuJoCo & LIBERO: OK")
except Exception as e:
    print(f"LIBERO Error: {e}")
    print("  Fix: bash scripts/setup/install_libero.sh")

print("\n[Step 3] Loading openvla/openvla-7b")
print("Tip: 下载中断 → bash scripts/download/resume_model_download.sh")

model_id = "openvla/openvla-7b"
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2" if use_flash else "sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    print(f"\nSUCCESS: OpenVLA loaded on {torch.cuda.get_device_name(0)}")
    print("Next: python run/quick_start.py")
except Exception as e:
    print(f"\nModel load failed: {e}")
    print("  bash scripts/download/resume_model_download.sh")

print("=" * 50)
