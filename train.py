# train.py
import torch
import yaml
import os
import random
import numpy as np
from src.data.create_dataloaders import create_dataloaders
from src.models.agp_model import AGPModel
from src.losses.total_loss import TotalLoss
from src.training.trainer import Trainer

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path='configs/training_config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config()
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 创建数据加载器
    print("\n1. 创建数据加载器...")
    train_loader, dev_loader, test_loader, num_aspects = create_dataloaders(
        train_jsonl=config['train_jsonl'],
        dev_jsonl=config['dev_jsonl'],
        test_jsonl=config['test_jsonl'],
        image_root=config['image_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("\n2. 创建模型...")
    model = AGPModel(
        num_aspects=num_aspects,
        hidden_dim=config['hidden_dim'],
        proj_dim=config['proj_dim'],
        num_queries=config['num_queries'],
        num_classes=config['num_classes'],
        freeze_bert_layers=config['freeze_bert_layers'],
        use_lora=config['use_lora'],
        lora_rank=config['lora_rank']
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 创建损失函数
    print("\n3. 创建损失函数...")
    loss_fn = TotalLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        temperature_infonce=config['temperature_infonce'],
        temperature_supcon=config['temperature_supcon']
    )
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    trainer.train()
    
    # 保存训练历史
    import json
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train': trainer.train_history,
            'dev': trainer.dev_history
        }, f, indent=2)
    print(f"\n训练历史保存到: {history_path}")

if __name__ == '__main__':
    main()
