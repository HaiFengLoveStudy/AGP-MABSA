#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载测试脚本
演示如何加载和使用转换后的JSONL数据
"""

import json
import os
from pathlib import Path
from collections import Counter

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_jsonl(file_path: str):
    """加载JSONL文件"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def test_load_data():
    """测试数据加载"""
    print("=" * 60)
    print("Twitter15 数据加载测试")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent
    
    # 加载训练集
    train_path = data_dir / 'raw' / 'train.jsonl'
    print(f"\n加载训练集: {train_path}")
    train_samples = load_jsonl(str(train_path))
    print(f"✓ 成功加载 {len(train_samples)} 个样本")
    
    # 显示前3个样本
    print("\n前3个样本:")
    print("-" * 60)
    for i, sample in enumerate(train_samples[:3], 1):
        print(f"\n样本 {i}:")
        print(f"  ID: {sample['sample_id']}")
        print(f"  文本: {sample['text'][:60]}...")
        print(f"  目标实体: {sample['aspect']}")
        print(f"  情感标签: {sample['label']} ({'负面' if sample['label']==0 else '中性' if sample['label']==1 else '正面'})")
        print(f"  图像: {sample['image_paths'][0]}")
        print(f"  Pair ID: {sample['pair_id']}")
    
    # 测试图像加载
    print("\n" + "=" * 60)
    print("测试图像加载")
    print("=" * 60)
    
    images_dir = data_dir / 'images'
    sample = train_samples[0]
    image_path = images_dir / sample['image_paths'][0]
    
    print(f"\n尝试加载图像: {image_path}")
    
    if not PIL_AVAILABLE:
        print(f"⚠ PIL/Pillow 未安装，跳过图像加载测试")
        print(f"  提示: pip install Pillow")
    elif image_path.exists():
        try:
            img = Image.open(image_path)
            print(f"✓ 图像加载成功")
            print(f"  尺寸: {img.size}")
            print(f"  格式: {img.format}")
            print(f"  模式: {img.mode}")
        except Exception as e:
            print(f"✗ 图像加载失败: {e}")
    else:
        print(f"✗ 图像文件不存在: {image_path}")
    
    # 统计同一图像的多个样本
    print("\n" + "=" * 60)
    print("同一图像的多个目标实体示例")
    print("=" * 60)
    
    # 找到第一个有多个样本的图像
    pair_counter = Counter([s['pair_id'] for s in train_samples])
    multi_sample_pairs = [pid for pid, count in pair_counter.items() if count > 1]
    
    if multi_sample_pairs:
        example_pair = multi_sample_pairs[0]
        related_samples = [s for s in train_samples if s['pair_id'] == example_pair]
        
        print(f"\nPair ID: {example_pair}")
        print(f"共有 {len(related_samples)} 个样本:")
        
        for i, sample in enumerate(related_samples, 1):
            print(f"\n  样本 {i}:")
            print(f"    目标实体: {sample['aspect']}")
            print(f"    情感: {['负面', '中性', '正面'][sample['label']]}")
            print(f"    文本: {sample['text'][:50]}...")
    
    # 数据集统计
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    
    for split in ['train', 'dev', 'test']:
        file_path = data_dir / 'raw' / f'{split}.jsonl'
        samples = load_jsonl(str(file_path))
        
        label_counts = Counter([s['label'] for s in samples])
        pair_ids = set([s['pair_id'] for s in samples])
        
        print(f"\n{split.upper()}:")
        print(f"  样本数: {len(samples)}")
        print(f"  唯一图像数: {len(pair_ids)}")
        print(f"  标签分布:")
        print(f"    负面: {label_counts[0]} ({label_counts[0]/len(samples)*100:.1f}%)")
        print(f"    中性: {label_counts[1]} ({label_counts[1]/len(samples)*100:.1f}%)")
        print(f"    正面: {label_counts[2]} ({label_counts[2]/len(samples)*100:.1f}%)")
        print(f"  平均每张图像的样本数: {len(samples)/len(pair_ids):.2f}")
    
    print("\n" + "=" * 60)
    print("✓ 数据加载测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    test_load_data()
