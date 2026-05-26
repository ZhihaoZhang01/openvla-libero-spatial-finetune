# LIBERO LoRA 实验自动化

## 保存策略（避免覆盖）

| 内容 | 位置 | 是否覆盖 |
|------|------|----------|
| 各实验 canonical 权重 | `checkpoints/<exp_id>/` | 不同实验 **不同目录**，互不影响 |
| 重训前旧权重 | `checkpoints/_archive/<exp_id>/<时间戳>/` | **mv 走**，不删 |
| 每次 sweep 完整记录 | `experiments/runs/<SWEEP_RUN_ID>/<exp_id>/` | **新目录**，不覆盖旧 sweep |
| 总表 | `experiments/results.csv` | **只追加**行 |
| 仿真 eval 原文 | `openvla/experiments/logs/EVAL-...` + 复制到 snapshot |

每个 `experiments/runs/<SWEEP_RUN_ID>/<exp_id>/` 含：

- `train.log` — 本次训练控制台输出
- `eval_console.log` — 评测控制台输出
- `eval_libero_spatial.log` — 结构化成功率日志（从 eval 脚本复制）
- `checkpoint/` — 该实验权重的**副本**（canonical 仍在 `checkpoints/`）
- `meta.json` — 超参、成功率、路径
- `canonical_ckpt_path.txt` — 指向 `checkpoints/` 中的目录

## 用法

```bash
screen -S exp-sweep
source scripts/env/activate_openvla.sh
bash scripts/exp/run_libero_sweep.sh 2>&1 | tee logs/exp_sweep_$(date +%Y%m%d).log
```

指定批次 ID（便于区分多次 sweep）：

```bash
SWEEP_RUN_ID=20260522-batch1 bash scripts/exp/run_libero_sweep.sh
```

强制重训（会先归档旧 checkpoint）：

```bash
FORCE_RETRAIN=1 bash scripts/exp/run_libero_sweep.sh D1
```

## 实验列表

见 `experiments.conf`。
