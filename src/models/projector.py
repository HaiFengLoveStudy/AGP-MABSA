# src/models/projector.py
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """投影头：用于对比学习"""
    
    def __init__(self, input_dim=768, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            h: [B, D'] L2归一化的投影特征
        """
        h = self.projection(x)
        h = nn.functional.normalize(h, p=2, dim=1)  # L2归一化
        return h

class SentimentClassifier(nn.Module):
    """情感分类器"""
    
    def __init__(self, input_dim=1536, hidden_dim=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*D] 拼接的多模态特征
        Returns:
            logits: [B, 3] 情感预测logits
        """
        return self.classifier(x)

class AspectClassifier(nn.Module):
    """方面分类器（辅助任务）"""
    
    def __init__(self, input_dim=768, num_aspects=3, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_aspects)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 单模态特征
        Returns:
            logits: [B, num_aspects] 方面预测logits
        """
        return self.classifier(x)
