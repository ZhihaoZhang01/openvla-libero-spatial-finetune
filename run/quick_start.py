#!/usr/bin/env python3
"""
未微调基座冒烟：LIBERO 仿真 + OpenVLA 推理 + 保存 MP4。

用法:
  source scripts/env/activate_openvla.sh
  pip install imageio[ffmpeg] accelerate -q
  python run/quick_start.py

输出: outputs/videos/quick_start_spatial_pick_black_bowl.mp4
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import VIDEOS_DIR, setup_env, setup_sys_path

setup_sys_path()
setup_env()

import os
import shutil

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# --- 配置 ---
MODEL_PATH = "openvla/openvla-7b"
BDDL_FOLDER = "libero_spatial"
BDDL_FILE = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
DEVICE = "cuda:0"
STEPS = 300  # 50 步仅 ~5s；300 步 @ 10fps ≈ 30s
FPS = 10
UNNORM_KEY = "bridge_orig"  # 基座无 libero 统计量

PROMPT = "In: What action should the robot take to {}?\nOut:"
INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"

OUTPUT_VIDEO = VIDEOS_DIR / "quick_start_spatial_pick_black_bowl.mp4"


def main() -> None:
    print("=" * 50)
    print("[1/4] LIBERO simulation")

    bddl_path = os.path.join(get_libero_path("bddl_files"), BDDL_FOLDER, BDDL_FILE)
    if not os.path.isfile(bddl_path):
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    )
    print(f"Simulation ready: {BDDL_FILE}")

    def close_sim() -> None:
        try:
            env.close()
        except Exception:
            pass

    print("\n[2/4] Load OpenVLA (flash_attention_2)")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    vla.eval()
    print("Model loaded.")

    print(f"\n[3/4] Inference ({STEPS} steps, {FPS} fps → ~{STEPS / FPS:.0f}s video)")
    print("Expect small motion before finetune — that is normal.")

    frames: list[np.ndarray] = []
    obs = env.reset()

    for step in range(STEPS):
        img = obs["agentview_image"][::-1, ::-1]
        frames.append(img.copy())

        image = Image.fromarray(img)
        inputs = processor(
            text=PROMPT.format(INSTRUCTION),
            images=image,
            return_tensors="pt",
        ).to(DEVICE, dtype=torch.bfloat16)

        with torch.inference_mode():
            action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

        if step % 30 == 0:
            print(f"  step {step}: action[:3] = {np.array(action)[:3]}")

        obs, reward, done, info = env.step(
            action.tolist() if hasattr(action, "tolist") else action
        )
        if done:
            obs = env.reset()
        print(f"\r  step {step + 1}/{STEPS}", end="", flush=True)

    # 先释放 MuJoCo/EGL，再 ffmpeg 编码，避免退出时 EGL_NOT_INITIALIZED 告警
    close_sim()
    del vla, processor, env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n\n[4/4] Save video")
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames)
    imageio.mimsave(str(OUTPUT_VIDEO), stack, fps=FPS)

    stamped = VIDEOS_DIR / (
        f"quick_start_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    shutil.copy2(OUTPUT_VIDEO, stamped)

    print(f"Video: {OUTPUT_VIDEO}")
    print(f"Copy:  {stamped}")
    print("=" * 50)
    print("If arm barely moves: OK for base model. Next → download_data → LoRA finetune.")


if __name__ == "__main__":
    main()
