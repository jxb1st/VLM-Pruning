import transformers
print("transformers version:")
print(transformers.__version__)

import sys
sys.path.insert(0, '/gpfs/projects/embodied3d/jianxu/vlm_pruning/DeepSeek-VL2')


from vlmeval.config import supported_VLM

model = supported_VLM['deepseek_vl2_tiny']()
# 前向单张图片
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # 这张图片上有一个带叶子的红苹果
# 前向多张图片
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # 提供的图片中有两个苹果