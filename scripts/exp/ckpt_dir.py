#!/usr/bin/env python3
"""Print OpenVLA finetune checkpoint directory name (must match vla-scripts/finetune.py)."""
from __future__ import annotations

import argparse
from pathlib import Path


def exp_id(
    *,
    vla_name: str = "openvla-7b",
    dataset_name: str = "libero_spatial_no_noops",
    batch_size: int = 16,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
    use_quantization: bool = False,
    run_id_note: str | None = None,
    image_aug: bool = True,
) -> str:
    eid = f"{vla_name}+{dataset_name}+b{batch_size * grad_accumulation_steps}+lr-{learning_rate}"
    if use_lora:
        eid += f"+lora-r{lora_rank}+dropout-{lora_dropout}"
    if use_quantization:
        eid += "+q-4bit"
    if run_id_note:
        eid += f"--{run_id_note}"
    if image_aug:
        eid += "--image_aug"
    return eid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/checkpoints"))
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--image-aug", choices=("true", "false"), default="true")
    p.add_argument("--run-id-note", default="")
    p.add_argument("--id-only", action="store_true", help="Print exp id only, not full path")
    args = p.parse_args()

    note = args.run_id_note or None
    eid = exp_id(lora_dropout=args.lora_dropout, run_id_note=note, image_aug=args.image_aug == "true")
    if args.id_only:
        print(eid)
    else:
        print(args.root / eid)


if __name__ == "__main__":
    main()
