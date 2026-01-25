#!/usr/bin/env python
"""
独立测试 IDEFICS 模型
不依赖 vlmeval 的全局导入，避免版本冲突
"""
import torch
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor
import warnings
import sys

print("="*70)
print("IDEFICS 独立测试脚本")
print("="*70)

class IDEFICSStandalone:
    """独立的 IDEFICS 模型封装"""
    
    def __init__(self, model_path='HuggingFaceM4/idefics-9b-instruct'):
        print(f"\n📦 正在加载模型: {model_path}")
        print("   这可能需要几分钟...")
        
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = {'max_new_tokens': 512}
        
        print("✅ 模型加载成功！\n")

    def generate(self, message):
        """
        生成回复
        Args:
            message: list，格式为 [图片路径1, 图片路径2, ..., 问题文本]
        """
        # 解析输入：分离图片和文本
        images = []
        text = ""
        
        for item in message:
            if isinstance(item, str) and item.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                try:
                    img = Image.open(item)
                    images.append(img)
                except Exception as e:
                    print(f"⚠️  警告：无法加载图片 {item}: {e}")
            else:
                text = item
        
        # 构建 IDEFICS 的 prompt 格式
        prompts = ['Users:']
        for img in images:
            prompts.append(img)
        prompts.append(text)
        prompts.extend(['<end_of_utterance>', '\nAssistant: '])
        
        # 推理
        try:
            inputs = self.processor(
                prompts, 
                add_end_of_utterance_token=False, 
                return_tensors='pt'
            ).to('cuda')
            
            exit_condition = self.processor.tokenizer(
                '<end_of_utterance>', 
                add_special_tokens=False
            ).input_ids
            
            bad_words_ids = self.processor.tokenizer(
                ['<image>', '<fake_token_around_image>'], 
                add_special_tokens=False
            ).input_ids

            generated_ids = self.model.generate(
                **inputs,
                eos_token_id=exit_condition,
                bad_words_ids=bad_words_ids,
                **self.kwargs,
            )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            result = generated_text[0].split('\nAssistant: ')[-1]
            return result
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主测试函数"""
    
    # 初始化模型
    try:
        model = IDEFICSStandalone()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 测试 1: 单张图片
    print("="*70)
    print("📸 测试 1: 单张图片理解")
    print("="*70)
    print("输入: ['assets/apple.jpg', 'What is in this image?']")
    
    try:
        ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
        if ret:
            print(f"✅ 输出: {ret}")
        else:
            print("❌ 生成失败（返回 None）")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 2: 多张图片
    print("\n" + "="*70)
    print("📸📸 测试 2: 多张图片理解")
    print("="*70)
    print("输入: ['assets/apple.jpg', 'assets/apple.jpg', 'How many apples...']")
    
    try:
        ret = model.generate([
            'assets/apple.jpg', 
            'assets/apple.jpg', 
            'How many apples are there in the provided images?'
        ])
        if ret:
            print(f"✅ 输出: {ret}")
        else:
            print("❌ 生成失败（返回 None）")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

