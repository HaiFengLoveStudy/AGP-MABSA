#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证脚本
验证生成的JSONL文件的完整性和正确性
"""

import json
import os
from pathlib import Path
from collections import Counter


def verify_jsonl_file(jsonl_path: str, images_dir: str):
    """
    验证JSONL文件
    
    Args:
        jsonl_path: JSONL文件路径
        images_dir: 图像目录路径
    """
    print(f"\n验证文件: {Path(jsonl_path).name}")
    print("-" * 60)
    
    samples = []
    missing_images = []
    label_counts = Counter()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
                
                # 验证必需字段
                required_fields = ['sample_id', 'text', 'aspect', 'image_paths', 'label', 'pair_id']
                for field in required_fields:
                    if field not in sample:
                        print(f"⚠ 第{line_num}行缺少字段: {field}")
                
                # 统计标签分布
                label_counts[sample['label']] += 1
                
                # 验证图像文件是否存在
                for image_path in sample['image_paths']:
                    full_image_path = os.path.join(images_dir, image_path)
                    if not os.path.exists(full_image_path):
                        missing_images.append(image_path)
                        
            except json.JSONDecodeError as e:
                print(f"⚠ 第{line_num}行JSON解析错误: {e}")
    
    # 输出统计信息
    print(f"✓ 总样本数: {len(samples)}")
    print(f"✓ 标签分布:")
    print(f"  - 负面 (0): {label_counts[0]}")
    print(f"  - 中性 (1): {label_counts[1]}")
    print(f"  - 正面 (2): {label_counts[2]}")
    
    if missing_images:
        print(f"⚠ 缺失的图像: {len(missing_images)} 个")
        if len(missing_images) <= 10:
            for img in missing_images:
                print(f"  - {img}")
    else:
        print(f"✓ 所有图像文件都存在")
    
    # 显示样本示例
    if samples:
        print(f"\n样本示例:")
        sample = samples[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    return len(samples), label_counts, len(missing_images)


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent
    
    print("=" * 60)
    print("Twitter15 数据验证")
    print("=" * 60)
    
    raw_dir = data_dir / 'raw'
    images_dir = data_dir / 'images'
    
    # 验证各个数据集
    splits = ['train', 'dev', 'test']
    total_samples = 0
    all_label_counts = Counter()
    total_missing = 0
    
    for split in splits:
        jsonl_path = raw_dir / f'{split}.jsonl'
        
        if not jsonl_path.exists():
            print(f"⚠ 文件不存在: {jsonl_path}")
            continue
        
        samples, label_counts, missing = verify_jsonl_file(
            str(jsonl_path), 
            str(images_dir)
        )
        
        total_samples += samples
        all_label_counts += label_counts
        total_missing += missing
    
    # 总体统计
    print("\n" + "=" * 60)
    print("总体统计")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"总体标签分布:")
    print(f"  - 负面 (0): {all_label_counts[0]} ({all_label_counts[0]/total_samples*100:.1f}%)")
    print(f"  - 中性 (1): {all_label_counts[1]} ({all_label_counts[1]/total_samples*100:.1f}%)")
    print(f"  - 正面 (2): {all_label_counts[2]} ({all_label_counts[2]/total_samples*100:.1f}%)")
    
    if total_missing == 0:
        print(f"\n✓ 数据验证通过！所有文件完整。")
    else:
        print(f"\n⚠ 发现 {total_missing} 个缺失的图像文件")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
