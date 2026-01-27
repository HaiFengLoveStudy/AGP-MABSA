# src/evaluation/metrics.py
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, label_names=['Negative', 'Neutral', 'Positive']):
        self.label_names = label_names
    
    def compute_metrics(self, all_preds, all_labels):
        """计算所有指标"""
        # 基础指标
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 各类别F1
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        
        # 分类报告
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.label_names,
            digits=4
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_f1': {
                self.label_names[i]: per_class_f1[i]
                for i in range(len(self.label_names))
            },
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ 混淆矩阵保存到: {save_path}")
    
    def print_metrics(self, metrics):
        """打印指标"""
        print("\n" + "="*60)
        print("评估指标")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\n各类别F1分数:")
        for label, f1 in metrics['per_class_f1'].items():
            print(f"  {label}: {f1:.4f}")
        print("\n分类报告:")
        print(metrics['classification_report'])
