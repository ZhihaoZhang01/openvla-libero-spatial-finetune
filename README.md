# OpenVLA × LIBERO-Spatial LoRA 微调与消融实验

在 [OpenVLA-7B](https://github.com/openvla/openvla) 上对 **LIBERO-Spatial** 进行 LoRA 微调，完成 5 组超参消融训练，并在仿真中评测。本仓库包含 **补丁**、**自动化脚本**、**实验日志** 与 **rollout 视频**，**不包含**完整上游 OpenVLA 代码树。

**仓库地址：** [github.com/ZhihaoZhang01/openvla-libero-spatial-finetune](https://github.com/ZhihaoZhang01/openvla-libero-spatial-finetune)

---

## 项目概览

| 项目 | 说明 |
|------|------|
| 基座模型 | OpenVLA-7B |
| 基准任务 | LIBERO-Spatial（`libero_spatial_no_noops`） |
| 微调方式 | LoRA（`rank=32`，`lr=5e-4`，`batch=16`） |
| 训练硬件 | NVIDIA A800 80GB（AutoDL） |
| 最优配置 | **S10k** — 10000 step，`dropout=0.1`，关闭图像增强 |
| 评测协议 | 前 **3 个任务** × 每任务 **10 次 trial** = 每模型 30 个 episode |

### 仿真成功率（评测批次 `20260522-eval`）

| 排名 | 配置 | 训练步数 | Dropout | 图像增强 | 总成功率 |
|------|------|----------|---------|----------|----------|
| 1 | **S10k** | 10000 | 0.1 | 关 | **70.0%**（21/30） |
| 2 | **A-F** | 6500 | 0.0 | 关 | **50.0%**（15/30） |
| 3 | D1 | 6500 | 0.1 | 开 | 26.7%（8/30） |
| 4 | D0 | 6500 | 0.0 | 开 | 16.7%（5/30） |
| 5 | S3k | 3000 | 0.0 | 开 | 0.0%（0/30） |

分任务明细见 [`experiments/REPORT_libero_spatial_sweep_20260522.md`](experiments/REPORT_libero_spatial_sweep_20260522.md) 与 [`experiments/results.csv`](experiments/results.csv)。

---

## 阶段一：Quick Start 冒烟（基座模型）

微调前，用 `run/quick_start.py` 在**未微调**的 OpenVLA-7B 上跑通整条链路（GPU、LIBERO 仿真、模型加载），确认环境可用。

### 基座模型 Rollout 视频

<video controls width="640" src="experiments/demos/quick_start_spatial_pick_black_bowl.mp4"></video>

*任务：拿起黑碗并放到盘子上（spatial 套件）。基座模型有动作意图，但成功率很低。*

### 训练曲线（冒烟阶段 — 请插入你的截图）

| 图表 | 占位文件路径 | 建议内容 |
|------|--------------|----------|
| 训练损失 | `experiments/assets/training_curves/quick_start_loss.png` | W&B `train/loss`；若仅推理可标注 N/A |
| 动作准确率 | `experiments/assets/training_curves/quick_start_action_accuracy.png` | 可选基座指标 |

![Quick Start — 训练损失](experiments/assets/training_curves/quick_start_loss.png)

![Quick Start — 动作准确率](experiments/assets/training_curves/quick_start_action_accuracy.png)

> 从 W&B 导出 PNG 后覆盖上述路径即可显示。命名说明见 [`experiments/assets/training_curves/README.md`](experiments/assets/training_curves/README.md)。

---

## 阶段二：消融训练 — 训练曲线

五组 LoRA 实验（训练批次 `20260522-train`）。固定超参：`lora_rank=32`，`lr=5e-4`，`batch=16`，数据集 `libero_spatial_no_noops`。

| 编号 | `lora_dropout` | `image_aug` | `max_steps` | 说明 |
|------|----------------|-------------|-------------|------|
| D0 | 0.0 | 开 | 6500 | 基线 |
| D1 | 0.1 | 开 | 6500 | Dropout 消融 |
| A-F | 0.0 | **关** | 6500 | 关闭图像增强 |
| S3k | 0.0 | 开 | **3000** | 短训（步数不足） |
| S10k | 0.1 | **关** | **10000** | 长训（最优） |

### 总览对比（请插入 5 条曲线合一图）

![消融实验 — 训练损失对比](experiments/assets/training_curves/sweep_train_loss_compare.png)

![消融实验 — 动作准确率对比](experiments/assets/training_curves/sweep_action_accuracy_compare.png)

### 各实验单独曲线（请插入 W&B 截图）

| D0 @ 6500 | D1 @ 6500 |
|-----------|-----------|
| ![D0 训练曲线](experiments/assets/training_curves/D0_curves.png) | ![D1 训练曲线](experiments/assets/training_curves/D1_curves.png) |

| A-F @ 6500 | S3k @ 3000 |
|------------|------------|
| ![A-F 训练曲线](experiments/assets/training_curves/A-F_curves.png) | ![S3k 训练曲线](experiments/assets/training_curves/S3k_curves.png) |

| S10k @ 10000（最优） |
|----------------------|
| ![S10k 训练曲线](experiments/assets/training_curves/S10k_curves.png) |

**训练曲线观察（W&B，定性）：**

- **S10k** 在 10000 step 时训练指标最好（loss 最低、action token accuracy 最高）。
- 相同 6500 step 下，**A-F**（无增强）比 **D1**（dropout 0.1 + 增强）更稳定。
- **S3k** 明显欠拟合，与仿真 **0%** 成功率一致。

---

## 阶段三：微调后仿真评测 — 视频演示

评测批次：`20260522-eval` · LIBERO-Spatial 前 3 任务 × 10 trials · 全部视频见 [`experiments/rollouts/2026_05_24/`](experiments/rollouts/2026_05_24/)（共 150 个 mp4）。

### 任务 0 — 拿起 plate 与 ramekin 之间的黑碗

| S10k ✅ 成功 | D0 ❌ 失败 |
|-------------|-----------|
| <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-19_15_22--episode=1--success=True--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4"></video> | <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-16_40_01--episode=1--success=False--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4"></video> |

*S10k 该任务 80% · D0 该任务 20%*

### 任务 1 — 拿起 ramekin 旁边的黑碗

| A-F ✅ 成功 | D1 ✅ 成功 |
|------------|-----------|
| <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-18_01_39--episode=11--success=True--task=pick_up_the_black_bowl_next_to_the_ramekin_and_pla.mp4"></video> | <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-17_19_34--episode=11--success=True--task=pick_up_the_black_bowl_next_to_the_ramekin_and_pla.mp4"></video> |

*A-F 90% · D1 60% · 本子集中相对最容易的任务。*

### 任务 2 — 拿起桌面中央的黑碗（最难）

| S10k ✅ 成功 | D0 ❌ 失败 |
|-------------|-----------|
| <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-19_15_22--episode=28--success=True--task=pick_up_the_black_bowl_from_table_center_and_place.mp4"></video> | <video controls width="300" src="experiments/rollouts/2026_05_24/2026_05_24-16_40_01--episode=21--success=False--task=pick_up_the_black_bowl_from_table_center_and_place.mp4"></video> |

*S10k 50% · D0/D1/A-F 约 10% · 弱配置的主要瓶颈。*

### S3k @ 3000 step — 完全失败

<video controls width="480" src="experiments/rollouts/2026_05_24/2026_05_24-18_33_30--episode=1--success=False--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4"></video>

*30 个 episode 全部失败；3000 step 不足以学到可用的空间操作策略。*

---

## 仓库结构

```
├── patches/openvla/          # finetune.py、run_libero_eval.py 补丁（含 upstream.diff）
├── scripts/
│   ├── env/                  # activate_openvla.sh
│   ├── train/                # LoRA 训练
│   ├── eval/                 # 单 checkpoint 评测
│   └── exp/                  # 5 组 sweep（experiments.conf、run_libero_sweep.sh）
├── experiments/
│   ├── rollouts/             # 150 个评测 mp4（2026_05_24）
│   ├── demos/                # quick_start 基座模型视频
│   ├── eval_logs/            # 各模型 EVAL 文本日志
│   ├── runs/                 # sweep 元数据（不含权重）
│   ├── assets/training_curves/  # ← 在此放入 W&B 截图
│   └── results.csv
└── run/                      # verify_phase01.py、quick_start.py 等
```

**不在本仓库中：** `openvla/`、`LIBERO/`、`dlimp/`、数据集、基座权重、LoRA checkpoint（每个约 465MB）。

---

## 复现步骤

### 1. 克隆仓库并打补丁

```bash
git clone https://github.com/ZhihaoZhang01/openvla-libero-spatial-finetune.git
git clone https://github.com/openvla/openvla.git

export OPENVLA_ROOT=/path/to/openvla
export PATCH_ROOT=/path/to/openvla-libero-spatial-finetune/patches/openvla
cp "${PATCH_ROOT}/finetune.py" "${OPENVLA_ROOT}/vla-scripts/finetune.py"
cp "${PATCH_ROOT}/run_libero_eval.py" "${OPENVLA_ROOT}/experiments/robot/libero/run_libero_eval.py"
```

### 2. 环境依赖

安装 OpenVLA、LIBERO、`dlimp`，并下载 **OpenVLA-7B** 与 **libero_spatial_no_noops** RLDS 数据（见 `scripts/setup/`、`scripts/download/`）。

### 3. 训练与评测

```bash
source scripts/env/activate_openvla.sh

# 单组基线训练
bash scripts/train/train_libero.sh

# 完整 5 组消融
TRAIN_ONLY=1 SWEEP_RUN_ID=my-train bash scripts/exp/run_libero_sweep.sh
EVAL_ONLY=1  SWEEP_RUN_ID=my-eval  bash scripts/exp/run_libero_sweep.sh
```

---

## 相对上游 OpenVLA 的修改

| 文件 | 修改内容 |
|------|----------|
| `vla-scripts/finetune.py` | 权重保存到 run 目录；训练结束不 merge LoRA（避免 80GB OOM） |
| `experiments/robot/libero/run_libero_eval.py` | 修复 LIBERO 路径；新增 `--num_tasks` 支持子集评测 |

详见 [`patches/openvla/README.md`](patches/openvla/README.md)。

---

## 致谢

- [openvla/openvla](https://github.com/openvla/openvla) — 基座 VLA 模型
- [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 基准与仿真环境
- [Escapist-coder/OpenVLA-Libero-Reproduction-Finetune](https://github.com/Escapist-coder/OpenVLA-Libero-Reproduction-Finetune) — 复现参考与 README 结构借鉴

---

## 引用

若使用 OpenVLA 或 LIBERO，请引用其原始论文。本仓库为个人学习与消融实验记录，与官方实现无隶属关系。
