实现了第一版本的naive InternVL3.5 token pruning baseline.
可以在/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/vlmeval/vlm/internvl/local_models/internvl3_5_30b_a3b/config.json
下选择是否开启

torchrun --nproc-per-node=8 run.py \
  --data MMBench_DEV_EN_V11 \
  --model InternVL3_5-30B-A3B \
  --verbose

test_tile_pruning.py用于测试单个case的pruning效果
visualiza_tile_pruning.py用于可视化当前case