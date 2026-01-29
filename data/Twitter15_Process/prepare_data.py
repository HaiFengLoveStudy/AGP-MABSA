#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter15数据集准备脚本
将Twitter15的TSV格式转换为AGP-MABSA所需的JSONL格式
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def parse_tsv_file(tsv_path: str) -> List[Dict]:
    """
    解析Twitter15的TSV文件
    
    Args:
        tsv_path: TSV文件路径
        
    Returns:
        解析后的数据列表
    """
    samples = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 跳过第一行表头
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 5:
            continue
            
        index = parts[0]
        label = int(parts[1])  # 0=负面, 1=中性, 2=正面
        image_id = parts[2]
        text_with_target = parts[3]  # 带有$T$标记的文本
        target_entity = parts[4]  # 目标实体
        
        # 将$T$替换为实际的目标实体，得到完整文本
        text = text_with_target.replace('$T$', target_entity)
        
        samples.append({
            'index': index,
            'label': label,
            'image_id': image_id,
            'text': text,
            'target_entity': target_entity,
            'text_with_mask': text_with_target
        })
    
    return samples


def convert_to_jsonl_format(samples: List[Dict], split_name: str, output_dir: str):
    """
    将解析的样本转换为JSONL格式并保存
    
    Args:
        samples: 解析后的样本列表
        split_name: 数据集划分名称 (train/dev/test)
        output_dir: 输出目录
    """
    output_path = os.path.join(output_dir, f'{split_name}.jsonl')
    
    # 创建一个字典来跟踪同一图文对的不同方面
    pair_id_counter = {}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(samples):
            # 使用图像ID作为pair_id的基础
            image_base = sample['image_id'].replace('.jpg', '')
            
            # 生成唯一的sample_id
            sample_id = f"twitter15_{split_name}_{idx+1:06d}"
            
            # 生成pair_id（同一张图片的不同样本共享相同的pair_id前缀）
            if image_base not in pair_id_counter:
                pair_id_counter[image_base] = 0
            pair_id_counter[image_base] += 1
            pair_id = f"twitter15_{split_name}_{image_base}"
            
            # 构建JSONL格式的数据
            jsonl_record = {
                'sample_id': sample_id,
                'text': sample['text'],
                'aspect': sample['target_entity'],  # 使用目标实体作为aspect
                'image_paths': [f"twitter2015_images/{sample['image_id']}"],
                'label': sample['label'],
                'pair_id': pair_id
            }
            
            # 写入文件
            f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
    
    print(f"✓ 已生成 {output_path}")
    print(f"  - 样本数量: {len(samples)}")
    print(f"  - 唯一图像数: {len(pair_id_counter)}")


def create_directory_structure(base_dir: str):
    """
    创建所需的目录结构
    
    Args:
        base_dir: 基础目录路径
    """
    directories = [
        os.path.join(base_dir, 'raw'),
        os.path.join(base_dir, 'processed'),
        os.path.join(base_dir, 'images')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent  # data目录
    
    # Twitter15数据路径（修正：数据在 Twitter15 文件夹中）
    twitter15_dir = data_dir / 'Twitter15' / 'twitter2015'
    
    # 创建输出目录结构
    print("=" * 60)
    print("步骤 1: 创建目录结构")
    print("=" * 60)
    create_directory_structure(str(data_dir))
    
    # 转换数据集
    print("\n" + "=" * 60)
    print("步骤 2: 转换数据集为JSONL格式")
    print("=" * 60)
    
    output_dir = data_dir / 'raw'
    
    splits = ['train', 'dev', 'test']
    total_samples = 0
    
    for split in splits:
        print(f"\n处理 {split} 集...")
        tsv_path = twitter15_dir / f'{split}.tsv'
        
        if not tsv_path.exists():
            print(f"⚠ 警告: 文件不存在 {tsv_path}")
            continue
        
        # 解析TSV文件
        samples = parse_tsv_file(str(tsv_path))
        total_samples += len(samples)
        
        # 转换并保存为JSONL
        convert_to_jsonl_format(samples, split, str(output_dir))
    
    # 检查图像目录
    print("\n" + "=" * 60)
    print("步骤 3: 检查图像文件")
    print("=" * 60)
    
    # 图像源路径（修正：图像在 Twitter15 文件夹中）
    images_source = data_dir / 'Twitter15' / 'twitter2015_images'
    images_target = data_dir / 'images' / 'twitter2015_images'
    
    if images_source.exists():
        image_count = len(list(images_source.glob('*.jpg')))
        print(f"✓ 找到图像文件: {image_count} 张")
        print(f"  源路径: {images_source}")
        
        # 如果目标路径不存在，创建符号链接
        if not images_target.exists():
            try:
                os.symlink(images_source, images_target)
                print(f"✓ 创建符号链接: {images_target} -> {images_source}")
            except Exception as e:
                print(f"⚠ 无法创建符号链接: {e}")
                print(f"  请手动将图像文件复制或链接到: {images_target}")
        else:
            print(f"✓ 图像目录已存在: {images_target}")
    else:
        print(f"⚠ 警告: 未找到图像目录 {images_source}")
    
    # 总结
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"\n生成的文件:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'dev.jsonl'}")
    print(f"  - {output_dir / 'test.jsonl'}")
    print(f"\n下一步:")
    print(f"  1. 验证生成的JSONL文件")
    print(f"  2. 确保图像路径正确")
    print(f"  3. (可选) 运行 LLM 扩写脚本生成 aspect_desc 字段")
    print("=" * 60)


if __name__ == '__main__':
    main()
