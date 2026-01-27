# src/data/llm_expansion.py
import json
import openai
from tqdm import tqdm
import time
import os

# 配置API
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

PROMPT_TEMPLATE = """Role: You are an assistant for social media sentiment analysis.

Task: Expand the given aspect word into a short phrase describing its visual and textual features in a review context.

Constraint: Use simple, casual English. Maximum 10 words. No introductory filler.

Input Aspect: "food"
Output: "taste presentation portion size and freshness of dishes"

Input Aspect: "service"
Output: "waiter attitude serving speed and customer care quality"

Input Aspect: "{aspect_word}"
Output:"""

def expand_aspect_openai(aspect_word, model="gpt-4o"):
    """使用OpenAI API扩写方面词"""
    prompt = PROMPT_TEMPLATE.format(aspect_word=aspect_word)
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 较低温度保持一致性
            max_tokens=30
        )
        expansion = response.choices[0].message.content.strip()
        return expansion
    except Exception as e:
        print(f"API调用失败: {e}")
        return aspect_word  # 失败时返回原词

def expand_dataset_openai(input_jsonl, output_jsonl):
    """批量扩写数据集"""
    # 读取数据
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 收集所有唯一的方面词
    unique_aspects = list(set([s['aspect'] for s in samples]))
    print(f"发现 {len(unique_aspects)} 个唯一方面: {unique_aspects}")
    
    # 批量扩写
    aspect_expansions = {}
    for aspect in tqdm(unique_aspects, desc="扩写方面词"):
        expansion = expand_aspect_openai(aspect)
        aspect_expansions[aspect] = expansion
        print(f"  {aspect} -> {expansion}")
        time.sleep(0.5)  # 避免API限流
    
    # 添加扩写到每个样本
    for sample in samples:
        sample['aspect_desc'] = aspect_expansions[sample['aspect']]
    
    # 保存
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！保存到 {output_jsonl}")
    return aspect_expansions

if __name__ == '__main__':
    # 扩写训练集
    expand_dataset_openai(
        'data/raw/train.jsonl',
        'data/processed/train_expanded.jsonl'
    )
    
    # 扩写验证集
    expand_dataset_openai(
        'data/raw/dev.jsonl',
        'data/processed/dev_expanded.jsonl'
    )
    
    # 扩写测试集
    expand_dataset_openai(
        'data/raw/test.jsonl',
        'data/processed/test_expanded.jsonl'
    )
