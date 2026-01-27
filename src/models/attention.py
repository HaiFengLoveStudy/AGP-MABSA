# src/models/attention.py
import torch
import torch.nn as nn

class AspectGuidedCrossAttention(nn.Module):
    """方面引导的交叉注意力模块"""
    
    def __init__(
        self,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        feedforward_dim=2048
    ):
        super().__init__()
        
        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, key_padding_mask=None):
        """
        Args:
            queries: [B, m, D] 方面查询
            keys: [B, L, D] 文本/图像特征
            values: [B, L, D] 文本/图像特征
            key_padding_mask: [B, L] padding mask (True表示padding位置)
        
        Returns:
            output: [B, m, D] 提取的方面相关特征
        """
        # 交叉注意力 + 残差
        attn_output, _ = self.cross_attn(
            query=queries,
            key=keys,
            value=values,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        queries = self.norm1(queries + self.dropout(attn_output))
        
        # FFN + 残差
        ffn_output = self.ffn(queries)
        output = self.norm2(queries + ffn_output)
        
        return output

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cross_attn = AspectGuidedCrossAttention().to(device)
    
    # 准备输入
    queries = torch.randn(4, 9, 768).to(device)  # [B, m, D]
    text_features = torch.randn(4, 80, 768).to(device)  # [B, L, D]
    
    # 交叉注意力
    output = cross_attn(queries, text_features, text_features)
    print(f"输出形状: {output.shape}")  # [4, 9, 768]
    print(f"✅ 交叉注意力模块正常工作")
