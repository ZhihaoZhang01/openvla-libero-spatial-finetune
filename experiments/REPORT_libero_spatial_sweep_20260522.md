# OpenVLA × LIBERO-Spatial LoRA 消融实验报告

| 项目 | 内容 |
|------|------|
| **报告日期** | 2026-05-24 |
| **训练批次** | `20260522-train` |
| **评测批次** | `20260522-eval` |
| **基础模型** | OpenVLA-7B + LoRA (`rank=32`) |
| **数据集** | `libero_spatial_no_noops` |
| **仿真套件** | `libero_spatial`（子集评测：前 3 任务） |

---

## 1. 摘要

在 AutoDL A800 上对 OpenVLA 进行 LIBERO-Spatial LoRA 微调，并完成 5 组超参消融的训练与仿真评测。评测协议为 **3 个任务 × 每任务 10 次 trial = 30 episodes/模型**。

**主要结论：**

1. **S10k**（10000 step、`dropout=0.1`、关闭图像增强）表现最佳，总成功率 **70.0%**（21/30）。
2. **A-F**（6500 step、无增强）次之，**50.0%**（15/30）；说明在本子集上 **关闭 image aug** 明显优于基线 D0。
3. **S3k**（3000 step）在仿真上 **0%**，训练步数不足时模型无法完成任何任务。
4. **Task 2**（从桌面中央取碗）对所有模型最难；S10k 在该任务上仍达 50%，其余多数仅 10%。

---

## 2. 实验环境与复现

### 2.1 硬件与路径

| 项 | 值 |
|----|-----|
| 平台 | AutoDL，NVIDIA A800 80GB |
| 工作目录 | `/root/autodl-tmp` |
| Conda 环境 | `openvla`（`scripts/env/activate_openvla.sh`） |
| 训练数据 | `/root/autodl-tmp/datasets/openvla-libero-spatial` |

### 2.2 固定超参数（未消融）

| 参数 | 值 |
|------|-----|
| `lora_rank` | 32 |
| `learning_rate` | 5e-4 |
| `batch_size` | 16 |
| `dataset` | `libero_spatial_no_noops` |
| W&B 项目 | `openvla-LoRA-ablation` |

### 2.3 评测协议

| 参数 | 值 |
|------|-----|
| `NUM_TASKS` | 3（suite 内 task 0–2） |
| `NUM_TRIALS` | 10 |
| `TASK_SUITE` | `libero_spatial` |
| 总 episodes | 30 / 模型，合计 150 |

**center_crop 规则**（与训练增强一致）：

| 模型 | `image_aug` 训练 | 评测 `center_crop` |
|------|------------------|-------------------|
| D0, D1, S3k | True | `True` |
| A-F, S10k | False | `False` |

### 2.4 复现命令

```bash
# 训练全套（已完成）
TRAIN_ONLY=1 SWEEP_RUN_ID=20260522-train bash /root/autodl-tmp/scripts/exp/run_libero_sweep.sh

# 评测全套（已完成）
EVAL_ONLY=1 SWEEP_RUN_ID=20260522-eval bash /root/autodl-tmp/scripts/exp/run_libero_sweep.sh

# 单模型评测示例（S10k）
source /root/autodl-tmp/scripts/env/activate_openvla.sh
CKPT=/root/autodl-tmp/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.1--exp-S10k
CENTER_CROP=False NUM_TASKS=3 NUM_TRIALS=10 \
  bash /root/autodl-tmp/scripts/eval/eval_libero_checkpoint.sh
```

---

## 3. 实验设计

配置定义见 `scripts/exp/experiments.conf`。

| ID | LoRA dropout | image_aug | max_steps | run_id 后缀 | 说明 |
|----|--------------|-----------|-----------|-------------|------|
| **D0** | 0.0 | 开 | 6500 | （默认） | 基线 |
| **D1** | 0.1 | 开 | 6500 | `exp-D1-drop01` | dropout 消融 |
| **A-F** | 0.0 | **关** | 6500 | `exp-A-augF` | 图像增强消融 |
| **S3k** | 0.0 | 开 | **3000** | `exp-S3k` | 短训消融 |
| **S10k** | 0.1 | **关** | **10000** | `exp-S10k` | 长训 + 无 aug |

### 3.1 评测任务（前 3 个）

| Task ID | 自然语言指令（摘要） |
|---------|---------------------|
| 0 | 拿起 plate 与 ramekin **之间**的黑碗，放到 plate 上 |
| 1 | 拿起 ramekin **旁边**的黑碗，放到 plate 上 |
| 2 | 拿起**桌面中央**的黑碗，放到 plate 上 |

---

## 4. 仿真评测结果

### 4.1 总表（按总成功率排序）

| 排名 | ID | 总成功率 | 成功数 | Task 0 | Task 1 | Task 2 | 最终日志成功率 |
|------|-----|----------|--------|--------|--------|--------|----------------|
| 1 | **S10k** | **70.0%** | 21/30 | 80% | 80% | 50% | 0.70 |
| 2 | **A-F** | **50.0%** | 15/30 | 50% | 90% | 10% | 0.50 |
| 3 | **D1** | **26.7%** | 8/30 | 10% | 60% | 10% | 0.267 |
| 4 | **D0** | **16.7%** | 5/30 | 20% | 20% | 10% | 0.167 |
| 5 | **S3k** | **0.0%** | 0/30 | 0% | 0% | 0% | 0.00 |

> Task 列 = 该任务 10 次 trial 的成功率；与日志中 `Current task success rate` 一致。

### 4.2 分任务成功率矩阵

```
              Task0   Task1   Task2   Total
S10k           80%     80%     50%     70%
A-F            50%     90%     10%     50%
D1             10%     60%     10%     27%
D0             20%     20%     10%     17%
S3k             0%      0%      0%      0%
```

### 4.3 Checkpoint 路径

| ID | Checkpoint 目录 |
|----|-----------------|
| D0 | `/root/autodl-tmp/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug` |
| D1 | `.../dropout-0.1--exp-D1-drop01--image_aug` |
| A-F | `.../dropout-0.0--exp-A-augF` |
| S3k | `.../dropout-0.0--exp-S3k--image_aug` |
| S10k | `.../dropout-0.1--exp-S10k` |

---

## 5. 结果文件索引

| 类型 | 路径 |
|------|------|
| **本报告** | `/root/autodl-tmp/experiments/REPORT_libero_spatial_sweep_20260522.md` |
| 汇总 CSV | `/root/autodl-tmp/experiments/results.csv` |
| 评测快照（每实验） | `/root/autodl-tmp/experiments/runs/20260522-eval/{D0,D1,A-F,S3k,S10k}/` |
| 快照内容 | `meta.json`, `eval_console.log`, `checkpoint/`（LoRA 副本） |
| 原始评测日志 | `/root/autodl-tmp/openvla/experiments/logs/EVAL-libero_spatial-openvla-*--sweep-20260522-eval-*.txt` |
| Rollout 视频 | `/root/autodl-tmp/openvla/rollouts/2026_05_24/` |
| 训练快照 | `/root/autodl-tmp/experiments/runs/20260522-train/` |
| 实验配置 | `/root/autodl-tmp/scripts/exp/experiments.conf` |

### 5.1 各模型评测日志文件名

| ID | 日志文件 |
|----|----------|
| D0 | `EVAL-...-16_40_01--sweep-20260522-eval-D0.txt` |
| D1 | `EVAL-...-17_19_34--sweep-20260522-eval-D1--exp-D1-drop01.txt` |
| A-F | `EVAL-...-18_01_39--sweep-20260522-eval-A-F--exp-A-augF.txt` |
| S3k | `EVAL-...-18_33_30--sweep-20260522-eval-S3k--exp-S3k.txt` |
| S10k | `EVAL-...-19_15_22--sweep-20260522-eval-S10k--exp-S10k.txt` |

快速提取成功率：

```bash
grep -E "Current (task|total) success rate" \
  /root/autodl-tmp/openvla/experiments/logs/EVAL*sweep-20260522-eval*.txt
```

### 5.2 CSV 说明

`experiments/results.csv` 中 `20260522-eval` 行的 `success_rate` 列可能为空或混入控制台日志（解析 bug）。**请以评测日志与本文表格为准。**

---

## 6. 分析与讨论

### 6.1 训练步数

- **S3k vs D0**：相同超参仅将步数从 6500 降至 3000，仿真成功率从 16.7% 跌至 **0%**，说明 3000 step 远未收敛。
- **S10k vs D0**：步数增至 10000，并配合 `dropout=0.1`、关闭 aug，成功率从 16.7% 升至 **70%**，为本次 sweep 最大增益来源。

### 6.2 图像增强（D0 vs A-F）

- **A-F（50%）** 显著高于 **D0（16.7%）**，在相同 6500 step、dropout=0 下，仅改变 aug 与评测 crop。
- 可能机制：训练时 aug 与评测 crop 的分布不匹配；或该 3 任务子集上 aug 引入的域偏移大于正则收益。
- **注意**：二者评测 `center_crop` 不同，严格 ablation 需在相同 crop 下再评。

### 6.3 Dropout（D0 vs D1）

- **D1（26.7%）** 略高于 **D0（16.7%）**，但远低于 **A-F（50%）**。
- **S10k** 同样使用 `dropout=0.1` 却表现最好，说明 dropout 效果强烈依赖 **训练步数** 与 **是否使用 aug**。

### 6.4 任务难度

- **Task 1**（碗在 ramekin 旁）对 D1、A-F、S10k 相对容易（60%–90%）。
- **Task 2**（桌面中央）是主要瓶颈：除 S10k（50%）外，D0/D1/A-F 均为 **10%**。
- 建议结合视频分析抓取失败模式：`rollouts/2026_05_24/*table_center*`。

### 6.5 与历史冒烟评测

早期曾对 D0 做 **10 任务 × 2 trials** 冒烟，约 **25%**。本次 D0 为 **3 任务 × 10 trials = 16.7%**。协议不同，**不宜直接数值对比**，仅作定性参考。

---

## 7. 局限性

1. **任务子集**：仅评 3/10 个 LIBERO-Spatial 任务，不能代表全套 benchmark。
2. **样本量**：每任务 10 trials，置信区间较宽；正式对比建议 50 trials。
3. **crop/aug 耦合**：A-F/S10k 与 D0 系列评测设置不完全相同。
4. **无官方论文数字对齐**：未与 OpenVLA 论文中 LIBERO 全量结果做同协议对比。

---

## 8. 建议后续工作

| 优先级 | 内容 |
|--------|------|
| 高 | 对 **S10k**、**A-F** 运行 `NUM_TASKS=10 NUM_TRIALS=50` 全套评测 |
| 中 | 修复 `libero_exp_common.sh` 中 CSV `success_rate` 解析，避免日志写入 CSV |
| 中 | 在相同 `center_crop` 下重做 D0 vs A-F 公平对比 |
| 低 | 对 Task 2 失败 case 做视频级错误分类（抓取/放置/碰撞） |

---

## 9. 附录：训练时间线（`20260522-train`）

| ID | 完成时间（CSV timestamp） |
|----|---------------------------|
| D0 | 2026-05-22 23:06（训练跳过，沿用已有 ckpt） |
| D1 | 2026-05-23 01:30 |
| A-F | 2026-05-23 03:49 |
| S3k | 2026-05-23 04:53 |
| S10k | 2026-05-23 08:33 |

## 10. 附录：评测时间线（`20260522-eval`）

| ID | 评测开始（CSV） | 日志时间戳 |
|----|-----------------|------------|
| D0 | 2026-05-24 17:19 | 16:40 |
| D1 | — | 17:19 |
| A-F | — | 18:01 |
| S3k | — | 18:33 |
| S10k | — | 19:15 |

---

*报告由 sweep `20260522-train` / `20260522-eval` 日志自动整理生成。*
