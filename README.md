实现了第一版本的naive InternVL3.5 token pruning baseline.
可以在以下路径的配置文件中选择是否开启：

```
vlmeval/vlm/internvl/local_models/internvl3_5_30b_a3b/config.json
```

## 运行命令

```bash
torchrun --nproc-per-node=8 run.py \
  --data MMBench_DEV_EN_V11 \
  --model InternVL3_5-30B-A3B \
  --verbose
```

## 测试脚本

- `test_tile_pruning.py` - 用于测试单个case的pruning效果
- `visualize_tile_pruning.py` - 用于可视化当前case