# src/data/create_dataloaders.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
from .dataset import MABSADataset, collate_fn

def create_dataloaders(
    train_jsonl='data/processed/train_expanded.jsonl',
    dev_jsonl='data/processed/dev_expanded.jsonl',
    test_jsonl='data/processed/test_expanded.jsonl',
    image_root='data/images',
    batch_size=32,
    num_workers=4
):
    """创建训练、验证和测试数据加载器"""
    
    # 初始化tokenizer和image processor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # 创建数据集
    train_dataset = MABSADataset(
        train_jsonl, image_root, tokenizer, image_processor
    )
    dev_dataset = MABSADataset(
        dev_jsonl, image_root, tokenizer, image_processor
    )
    test_dataset = MABSADataset(
        test_jsonl, image_root, tokenizer, image_processor
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader, train_dataset.num_aspects

# 测试
if __name__ == '__main__':
    train_loader, dev_loader, test_loader, num_aspects = create_dataloaders(
        batch_size=8
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(dev_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"方面类别数: {num_aspects}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n批次数据形状:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
