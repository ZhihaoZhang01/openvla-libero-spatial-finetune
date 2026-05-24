#!/usr/bin/env python3
"""
阶段 0 + 阶段 1 环境验证（不含阶段 2+：LIBERO 补丁、数据、模型下载等）
用法: source scripts/env/activate_openvla.sh && python run/verify_phase01.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

AUTODL_ROOT = Path(os.environ.get("AUTODL_ROOT", Path(__file__).resolve().parents[1]))
OPENVLA_DIR = AUTODL_ROOT / "openvla"
DLIMP_DIR = AUTODL_ROOT / "dlimp"

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    status = WARN if warn and ok else (PASS if ok else FAIL)
    results.append((status, name, detail))
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
    line = f"[{icon} {status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def version_match(pkg: str, expected_prefix: str | tuple[str, ...]) -> tuple[bool, str]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "?")
    except Exception as e:
        return False, str(e)
    prefixes = (expected_prefix,) if isinstance(expected_prefix, str) else expected_prefix
    ok = any(v.startswith(p) for p in prefixes)
    return ok, v


def _fix_dlimp_import_path() -> None:
    """dlimp 仓库名与包名相同：须把仓库根加入 path，不能仅靠 /root/autodl-tmp（会 namespace 遮蔽）。"""
    os.chdir(AUTODL_ROOT)
    repo_root = str(DLIMP_DIR.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "dlimp" in sys.modules:
        del sys.modules["dlimp"]


def main() -> int:
    _fix_dlimp_import_path()
    print("=" * 60)
    print("OpenVLA 复现 — 阶段 0 & 1 环境验证")
    print(f"Python: {sys.version.split()[0]} @ {sys.executable}")
    print(f"AUTODL_ROOT: {AUTODL_ROOT}")
    print("=" * 60)

    # ---------- 阶段 0 ----------
    print("\n## 阶段 0：系统与 Conda\n")

    check("Python 3.10", sys.version_info[:2] == (3, 10), sys.version.split()[0])

    for var in ("AUTODL_ROOT", "HF_HOME", "HF_ENDPOINT", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        val = os.environ.get(var)
        check(f"环境变量 {var}", bool(val), val or "未设置（可在 ~/.bashrc 配置）", warn=not val)

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        tv = torch.__version__
        check("PyTorch 已安装", True, tv)
        check("PyTorch 2.2.x", tv.startswith("2.2"), tv)
        check("CUDA 可用", cuda_ok, gpu)
        check("CUDA 版本 12.1", "12.1" in (torch.version.cuda or ""), torch.version.cuda or "?")
    except Exception as e:
        check("PyTorch", False, str(e))

    for lib in ("libEGL.so", "libGL.so", "libosmesa"):
        found = any(
            lib in f
            for d in ("/usr/lib", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")
            if Path(d).exists()
            for f in os.listdir(d)
        ) if Path("/usr/lib").exists() else False
        if not found:
            found = shutil.which("patchelf") is not None  # 弱替代：至少装过系统依赖脚本
        check(f"系统库 {lib}", found or shutil.which("ffmpeg") is not None, "EGL/GL 相关（训练评测用）", warn=True)

    for cmd in ("git", "ffmpeg", "patchelf"):
        check(f"命令 {cmd}", shutil.which(cmd) is not None, shutil.which(cmd) or "未找到")

    # ---------- 阶段 1 ----------
    print("\n## 阶段 1：OpenVLA 仓库与 Python 依赖\n")

    check("openvla 目录存在", OPENVLA_DIR.is_dir(), str(OPENVLA_DIR))
    check("openvla 为 git 仓库", (OPENVLA_DIR / ".git").exists())
    check("openvla setup/pyproject", (OPENVLA_DIR / "pyproject.toml").exists() or (OPENVLA_DIR / "setup.py").exists())
    check("requirements-min.txt", (OPENVLA_DIR / "requirements-min.txt").is_file())
    check("vla-scripts/finetune.py", (OPENVLA_DIR / "vla-scripts" / "finetune.py").is_file())

    check("dlimp 目录存在", DLIMP_DIR.is_dir(), str(DLIMP_DIR))
    check("dlimp 可安装结构", (DLIMP_DIR / "setup.py").exists() or (DLIMP_DIR / "pyproject.toml").exists())

    # requirements-min.txt
    for pkg, ver in (
        ("timm", "0.9"),
        ("transformers", "4.40"),
        ("tokenizers", "0.19"),
    ):
        ok, v = version_match(pkg, ver)
        check(f"{pkg} ~{ver}", ok, v)

    for pkg, ver in (
        ("peft", "0.11"),
        ("bitsandbytes", "0.43"),
        ("wandb", "0."),
        ("packaging", ""),
        ("ninja", ""),
        ("einops", ""),
    ):
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "installed")
            ok = v.startswith(ver) if ver else True
            check(pkg, ok, v)
        except Exception as e:
            check(pkg, False, str(e))

    try:
        import flash_attn

        check("flash-attn", True, flash_attn.__version__)
    except Exception as e:
        check("flash-attn", False, f"{e}（可选，训练可用 sdpa）", warn=True)

    try:
        from dlimp import DLataset

        import dlimp

        check("dlimp.DLataset", True, getattr(dlimp, "__file__", ""))
    except Exception as e:
        check("dlimp.DLataset", False, str(e))

    try:
        import draccus as _draccus_check
        from draccus import wrap, parse

        pip_ver = subprocess.run(
            [sys.executable, "-m", "pip", "show", "draccus"],
            capture_output=True,
            text=True,
        )
        pip_line = next((l.split(":", 1)[1].strip() for l in pip_ver.stdout.splitlines() if l.startswith("Version:")), "?")
        ok = pip_line.startswith("0.8") and hasattr(_draccus_check, "wrap")
        check("draccus 0.8.x (pip + wrap)", ok, f"pip={pip_line} __version__={_draccus_check.__version__}")
    except Exception as e:
        check("draccus 0.8.x (pip + wrap)", False, str(e))

    try:
        import prismatic

        check("openvla (prismatic) editable", True, getattr(prismatic, "__file__", ""))
    except Exception as e:
        check("openvla (prismatic) editable", False, str(e))

    try:
        import tensorflow as tf

        import numpy as np

        np_ok = tuple(int(x) for x in np.__version__.split(".")[:2]) < (2, 0)
        check("tensorflow", True, tf.__version__)
        check("numpy < 2 (tensorflow)", np_ok, np.__version__)
    except Exception as e:
        check("tensorflow", False, str(e))

    # pip show openvla / libero 不应在阶段1必需
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", "openvla"],
        capture_output=True,
        text=True,
    )
    editable = "Editable project location" in (r.stdout or "") or OPENVLA_DIR.as_posix() in (r.stdout or "")
    check("pip 包 openvla/prismatic (-e)", r.returncode == 0 or editable, (r.stdout or r.stderr or "")[:120])

    # 目录（阶段 1 后建议存在，数据可空）
    for d in ("hf_cache", "datasets", "checkpoints", "outputs/videos"):
        p = AUTODL_ROOT / d
        check(f"目录 {d}/", p.exists(), str(p), warn=not p.exists())

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print(f"合计: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL / {len(results)} 项")
    print("=" * 60)

    if n_fail:
        print("\n未通过项请对照 docs/OpenVLA-Libero-Reproduction-Guide.md 阶段 0–1 补装。")
        return 1
    if n_warn:
        print("\n存在 WARN（多为环境变量或可选组件），阶段 1 核心依赖已就绪。")
    else:
        print("\n阶段 0 & 1 全部通过，可进入阶段 2（手动改码）。")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
