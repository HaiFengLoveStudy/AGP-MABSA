# src/losses/classification.py
import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """情感分类损失"""
    
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 3] 情感预测logits
            labels: [B] 真实标签
        
        Returns:
            loss: scalar
        """
        return self.criterion(logits, labels)
