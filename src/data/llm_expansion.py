# src/data/llm_expansion.py
"""
LLM方面词扩写脚本 - 支持多线程和自定义API配置
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

# ==================== 配置参数 ====================
# API配置
API_BASE_URL = "https://api.vveai.com/v1"  # API基础URL
API_KEY = "sk-tIxPGi9xnL3cf74w518fE609AfE34c308c717b24811a0722"                # API密钥
# MODEL_NAME = "qwen3-vl-235b-a22b-instruct"                        # 模型名称
MODEL_NAME = "gpt-5.2"                        # 模型名称
# 请求参数
TEMPERATURE = 0.3  # 生成温度，越低越确定性
MAX_TOKENS = 30    # 最大生成token数
TIMEOUT = 30       # 请求超时时间（秒）

# 重试配置
MAX_RETRIES = 6           # 最大重试次数
RETRY_DELAY = 1.0         # 重试延迟（秒）
RETRY_BACKOFF = 2.0       # 重试延迟倍数（指数退避）

# 并发配置
NUM_THREADS = 20           # 并发线程数
RATE_LIMIT_DELAY = 0.3    # 请求间隔（秒），避免触发限流

# Prompt配置
SYSTEM_PROMPT = """You are an assistant for social media sentiment analysis.
Your task is to expand aspect words into short descriptive phrases."""

USER_PROMPT_TEMPLATE = """Expand the given aspect word into a short phrase describing its visual and textual features in a review context.

Constraint: Use simple, casual English. Maximum 10 words. No introductory filler.

Examples:
- Input: "food" → Output: "taste presentation portion size and freshness of dishes"
- Input: "service" → Output: "waiter attitude serving speed and customer care quality"

Input Aspect: "{aspect_word}"
Output:"""

# 输入输出路径
INPUT_FILES = {
    'train': 'data/raw/train.jsonl',
    'dev': 'data/raw/dev.jsonl',
    'test': 'data/raw/test.jsonl'
}
OUTPUT_FILES = {
    'train': 'data/processed/train_expanded.jsonl',
    'dev': 'data/processed/dev_expanded.jsonl',
    'test': 'data/processed/test_expanded.jsonl'
}

# ==================== 核心函数 ====================

def call_llm_api(
    messages: List[Dict[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL_NAME
) -> Optional[str]:
    """
    调用LLM API
    
    Args:
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大token数
        model: 模型名称
        
    Returns:
        生成的文本，失败返回None
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        
        # 提取模型返回的文本
        content = result['choices'][0]['message']['content'].strip()
        
        # 只打印模型返回的内容
        print(f"  [模型返回] {content}")
        
        return content
        
    except requests.exceptions.Timeout:
        print(f"⚠ 请求超时 (timeout={TIMEOUT}s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠ 请求失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"⚠ 响应解析失败: {e}")
        return None
    except Exception as e:
        print(f"⚠ 未知错误: {e}")
        return None


def expand_aspect_with_retry(
    aspect_word: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
    retry_backoff: float = RETRY_BACKOFF
) -> str:
    """
    带重试机制的方面词扩写
    
    Args:
        aspect_word: 方面词
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟
        retry_backoff: 重试延迟倍数
        
    Returns:
        扩写后的描述，失败返回原词
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(aspect_word=aspect_word)}
    ]
    
    current_delay = retry_delay
    
    for attempt in range(max_retries):
        result = call_llm_api(messages)
        
        if result is not None:
            return result
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries - 1:
            print(f"  第 {attempt + 1}/{max_retries} 次尝试失败，{current_delay:.1f}秒后重试...")
            time.sleep(current_delay)
            current_delay *= retry_backoff
    
    # 所有重试都失败，返回原词
    print(f"  ✗ 方面词 '{aspect_word}' 扩写失败，使用原词")
    return aspect_word


def expand_aspect_worker(aspect_word: str) -> tuple[str, str]:
    """
    线程工作函数：扩写单个方面词
    
    Args:
        aspect_word: 方面词
        
    Returns:
        (aspect_word, expansion) 元组
    """
    expansion = expand_aspect_with_retry(aspect_word)
    time.sleep(RATE_LIMIT_DELAY)  # 限流
    return aspect_word, expansion


def load_cache(cache_file: str) -> Dict[str, str]:
    """
    加载缓存的方面词扩写结果
    
    Args:
        cache_file: 缓存文件路径
        
    Returns:
        已完成的方面词扩写字典
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ 读取缓存失败: {e}")
            return {}
    return {}


def save_cache(cache_file: str, expansions: Dict[str, str]):
    """
    保存方面词扩写结果到缓存
    
    Args:
        cache_file: 缓存文件路径
        expansions: 方面词扩写字典
    """
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(expansions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ 保存缓存失败: {e}")


def expand_aspects_parallel(
    aspects: List[str],
    num_threads: int = NUM_THREADS,
    cache_file: str = None
) -> Dict[str, str]:
    """
    并行扩写多个方面词（支持断点续传）
    
    Args:
        aspects: 方面词列表
        num_threads: 线程数
        cache_file: 缓存文件路径（可选）
        
    Returns:
        方面词到扩写描述的映射字典
    """
    # 加载缓存
    aspect_expansions = {}
    if cache_file:
        aspect_expansions = load_cache(cache_file)
        if aspect_expansions:
            print(f"✓ 从缓存加载 {len(aspect_expansions)} 个已完成的方面词")
    
    # 过滤出未处理的方面词
    remaining_aspects = [a for a in aspects if a not in aspect_expansions]
    
    if not remaining_aspects:
        print(f"✓ 所有方面词已处理完成")
        return aspect_expansions
    
    print(f"使用 {num_threads} 个线程并行处理 {len(remaining_aspects)} 个方面词（共 {len(aspects)} 个）...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_aspect = {
            executor.submit(expand_aspect_worker, aspect): aspect 
            for aspect in remaining_aspects
        }
        
        # 使用tqdm显示进度
        completed_count = len(aspect_expansions)
        with tqdm(total=len(aspects), initial=completed_count, desc="扩写进度") as pbar:
            for future in as_completed(future_to_aspect):
                aspect_word, expansion = future.result()
                aspect_expansions[aspect_word] = expansion
                print(f"  ✓ {aspect_word} → {expansion}")
                pbar.update(1)
                
                # 每完成一个就保存缓存
                if cache_file:
                    save_cache(cache_file, aspect_expansions)
    
    return aspect_expansions


def expand_dataset(
    input_jsonl: str,
    output_jsonl: str,
    num_threads: int = NUM_THREADS
) -> Dict[str, str]:
    """
    扩写整个数据集（支持断点续传）
    
    Args:
        input_jsonl: 输入JSONL文件路径
        output_jsonl: 输出JSONL文件路径
        num_threads: 并发线程数
        
    Returns:
        方面词扩写映射字典
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {input_jsonl}")
    print(f"{'='*60}")
    
    # 生成缓存文件路径
    cache_file = input_jsonl.replace('.jsonl', '_cache.json')
    
    # 读取数据
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"✓ 读取 {len(samples)} 个样本")
    
    # 收集唯一的方面词
    unique_aspects = sorted(set([s['aspect'] for s in samples]))
    print(f"✓ 发现 {len(unique_aspects)} 个唯一方面词")
    
    # 并行扩写（带缓存）
    aspect_expansions = expand_aspects_parallel(unique_aspects, num_threads, cache_file)
    
    # 添加扩写到每个样本
    for sample in samples:
        sample['aspect_desc'] = aspect_expansions[sample['aspect']]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # 保存
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 完成！保存到 {output_jsonl}")
    print(f"✓ 成功扩写: {len([v for v in aspect_expansions.values() if v])} / {len(unique_aspects)}")
    
    # 删除缓存文件
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"✓ 清理缓存文件")
        except Exception as e:
            print(f"⚠ 清理缓存文件失败: {e}")
    
    return aspect_expansions


def main():
    """主函数"""
    print("="*60)
    print("LLM 方面词扩写工具")
    print("="*60)
    print(f"API地址: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print(f"温度: {TEMPERATURE}")
    print(f"并发线程数: {NUM_THREADS}")
    print(f"最大重试次数: {MAX_RETRIES}")
    print("="*60)
    
    # 扩写所有数据集
    all_expansions = {}
    
    for split, input_file in INPUT_FILES.items():
        output_file = OUTPUT_FILES[split]
        
        if not os.path.exists(input_file):
            print(f"\n⚠ 跳过 {split}：文件不存在 {input_file}")
            continue
        
        try:
            expansions = expand_dataset(input_file, output_file, NUM_THREADS)
            all_expansions[split] = expansions
        except Exception as e:
            print(f"\n✗ 处理 {split} 时出错: {e}")
            continue
    
    # 总结
    print("\n" + "="*60)
    print("所有数据集处理完成！")
    print("="*60)
    
    for split, expansions in all_expansions.items():
        print(f"{split.upper()}: {len(expansions)} 个方面词已扩写")
    
    print("="*60)
    
    # 打印所有方面词及其描述
    print("\n" + "="*60)
    print("方面词及其描述汇总")
    print("="*60)
    
    # 合并所有数据集的方面词（去重）
    all_unique_aspects = {}
    for split, expansions in all_expansions.items():
        for aspect, desc in expansions.items():
            if aspect not in all_unique_aspects:
                all_unique_aspects[aspect] = desc
    
    # 按字母顺序排序
    sorted_aspects = sorted(all_unique_aspects.items())
    
    print(f"\n共 {len(sorted_aspects)} 个唯一方面词：\n")
    for idx, (aspect, desc) in enumerate(sorted_aspects, 1):
        print(f"{idx:4d}. {aspect:30s} → {desc}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
