"""项目路径与环境变量（run/ 下脚本共用）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AUTODL_ROOT", Path(__file__).resolve().parents[1])).resolve()

OPENVLA_DIR = ROOT / "openvla"
DLIMP_DIR = ROOT / "dlimp"
LIBERO_DIR = ROOT / "LIBERO"
HF_CACHE = ROOT / "hf_cache"
DATASETS_DIR = ROOT / "datasets"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = ROOT / "logs"


def setup_sys_path() -> None:
    for p in (DLIMP_DIR, LIBERO_DIR, OPENVLA_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def setup_env() -> None:
    os.environ.setdefault("AUTODL_ROOT", str(ROOT))
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # imageio/ffmpeg 会 fork；避免 tokenizers 并行与 fork 冲突的告警
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    setup_sys_path()
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
