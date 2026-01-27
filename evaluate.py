# evaluate.py
import torch
import yaml
import os
from tqdm import tqdm
from src.data.create_dataloaders import create_dataloaders
from src.models.agp_model import AGPModel
from src.evaluation.metrics import MetricsCalculator

def load_checkpoint(checkpoint_path, model, device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载模型: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']+1}")
    print(f"   Best Dev F1: {checkpoint['best_dev_f1']:.4f}")
    return model

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="评估中"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        outputs = model(batch)
        preds = outputs['sentiment_logits'].argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    return all_preds, all_labels

def main():
    # 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    print("\n1. 创建数据加载器...")
    _, _, test_loader, num_aspects = create_dataloaders(
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
    ).to(device)
    
    # 加载最佳模型
    print("\n3. 加载模型...")
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
    model = load_checkpoint(checkpoint_path, model, device)
    
    # 评估
    print("\n4. 开始评估...")
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # 计算指标
    print("\n5. 计算指标...")
    calculator = MetricsCalculator()
    metrics = calculator.compute_metrics(all_preds, all_labels)
    
    # 打印指标
    calculator.print_metrics(metrics)
    
    # 绘制混淆矩阵
    os.makedirs('results', exist_ok=True)
    calculator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='results/confusion_matrix.png'
    )
    
    # 保存结果
    import json
    results_path = 'results/test_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'per_class_f1': metrics['per_class_f1']
        }, f, indent=2)
    print(f"\n✅ 结果保存到: {results_path}")

if __name__ == '__main__':
    main()
