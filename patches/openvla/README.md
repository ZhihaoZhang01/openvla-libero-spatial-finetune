# OpenVLA 本地修改（不含完整 upstream 仓库）

基于官方 [openvla/openvla](https://github.com/openvla/openvla) `main`，仅保留本复现项目改动的文件。

## 修改文件

| 文件 | 说明 |
|------|------|
| `vla-scripts/finetune.py` | `save_dir` 指向 run 目录；训练结束不 merge LoRA（避免 OOM） |
| `experiments/robot/libero/run_libero_eval.py` | LIBERO 路径修复；新增 `--num_tasks` 子集评测 |

## 安装到已有 OpenVLA 克隆

```bash
export OPENVLA_ROOT=/path/to/openvla   # 官方 git clone 目录
export PATCH_ROOT=/path/to/this-repo/patches/openvla

cp "${PATCH_ROOT}/finetune.py" "${OPENVLA_ROOT}/vla-scripts/finetune.py"
cp "${PATCH_ROOT}/run_libero_eval.py" "${OPENVLA_ROOT}/experiments/robot/libero/run_libero_eval.py"
```

或使用 diff（在 `openvla` 仓库根目录执行）：

```bash
cd "${OPENVLA_ROOT}"
git apply /path/to/this-repo/patches/openvla/upstream.diff
```

## 首次克隆（推荐）

```bash
git clone https://github.com/openvla/openvla.git
# 然后按上面 cp 或 git apply
```

本 GitHub 仓库**不包含** `openvla/` 目录，避免与上游重复且体积过大。
