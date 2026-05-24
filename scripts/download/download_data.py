#!/usr/bin/env python3
"""下载 LIBERO-Spatial RLDS。用法: python scripts/download/download_data.py"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("AUTODL_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = ROOT / "datasets" / "openvla-libero-spatial"
os.makedirs(DATA_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print(f"Downloading to {DATA_ROOT} ...")
snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir=str(DATA_ROOT),
    allow_patterns="*libero_spatial*",
    resume_download=True,
)
print("Done.")
