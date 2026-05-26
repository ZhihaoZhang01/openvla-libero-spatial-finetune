# openpi 本地修改

基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。

| 文件 | 说明 |
|------|------|
| `examples_libero_main.py` | LIBERO 评测客户端：`num_tasks`、日志/视频命名对齐 OpenVLA sweep |

安装到克隆的 openpi 仓库：

```bash
cp patches/openpi/examples_libero_main.py openpi/examples/libero/main.py
```

或由 `scripts/eval/setup_openpi_libero.sh` 自动覆盖。

评测官方 **π₀.₅-LIBERO**：

```bash
bash scripts/eval/setup_openpi_libero.sh   # 首次
bash scripts/eval/eval_pi05_libero.sh      # 3 tasks × 10 trials
```

**国内加速下载权重**（OpenPI JAX 格式，可替代 `gs://`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/eval/download_pi05_libero_hf.sh
```

默认从 `bf-jeon/pi05_libero` 拉取，目录与 `OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero` 一致。
