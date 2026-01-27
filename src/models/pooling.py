# src/models/pooling.py
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """注意力池化：智能聚合多查询特征"""
    
    def __init__(self, hidden_dim=768, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 全局可学习聚合向量
        self.aggregator = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.aggregator)
        
        # 注意力模块
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, Z):
        """
        Args:
            Z: [B, m, D] 多查询特征
        
        Returns:
            pooled: [B, D] 聚合后的单一特征
        """
        batch_size = Z.size(0)
        
        # 扩展聚合向量到batch
        query = self.aggregator.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # 注意力池化：Q=aggregator, K=Z, V=Z
        output, attn_weights = self.attn(
            query=query,
            key=Z,
            value=Z,
            need_weights=True
        )  # output: [B, 1, D], attn_weights: [B, 1, m]
        
        pooled = output.squeeze(1)  # [B, D]
        
        return pooled, attn_weights.squeeze(1)  # [B, D], [B, m]

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pooling = AttentionPooling().to(device)
    
    # 输入多查询特征
    Z = torch.randn(4, 9, 768).to(device)
    
    # 池化
    pooled, attn_weights = pooling(Z)
    print(f"池化前: {Z.shape}")  # [4, 9, 768]
    print(f"池化后: {pooled.shape}")  # [4, 768]
    print(f"注意力权重: {attn_weights.shape}")  # [4, 9]
    print(f"权重和（应约等于1）: {attn_weights[0].sum().item():.4f}")
    print(f"✅ 注意力池化正常工作")
