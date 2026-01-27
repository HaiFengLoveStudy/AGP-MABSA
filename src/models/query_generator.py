# src/models/query_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQueryGenerator(nn.Module):
    """混合查询生成器：隐式查询 + 显式查询"""
    
    def __init__(
        self,
        num_aspects,
        hidden_dim=768,
        num_learnable_queries=8
    ):
        """
        Args:
            num_aspects: 方面类别数量
            hidden_dim: 隐藏维度（768）
            num_learnable_queries: 可学习查询数量（默认8）
        """
        super().__init__()
        self.num_aspects = num_aspects
        self.hidden_dim = hidden_dim
        self.num_learnable_queries = num_learnable_queries
        
        # 方面Embedding（每个方面一个基础向量）
        self.aspect_embeddings = nn.Embedding(num_aspects, hidden_dim)
        
        # 可学习查询参数（共享给所有方面）
        self.learnable_params = nn.Parameter(
            torch.randn(num_learnable_queries, hidden_dim)
        )
        nn.init.xavier_uniform_(self.learnable_params)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, aspect_ids, aspect_desc_encoding, text_encoder):
        """
        Args:
            aspect_ids: [B] 方面ID
            aspect_desc_encoding: dict with 'input_ids' [B, L] and 'attention_mask' [B, L]
            text_encoder: TextEncoder实例（共享BERT权重）
        
        Returns:
            queries: [B, 9, D] 混合查询（8隐式 + 1显式）
        """
        batch_size = aspect_ids.size(0)
        device = aspect_ids.device
        
        # === Part A: 构造隐式查询 ===
        # 1. 获取方面基础向量 [B, D]
        base_aspect = self.aspect_embeddings(aspect_ids)
        
        # 2. 广播相加：[B, 1, D] + [1, 8, D] -> [B, 8, D]
        implicit_queries = base_aspect.unsqueeze(1) + self.learnable_params.unsqueeze(0)
        
        # 3. 层归一化
        implicit_queries = self.layer_norm(implicit_queries)  # [B, 8, D]
        
        # === Part B: 构造显式查询 ===
        # 使用text_encoder的BERT编码LLM描述
        desc_features = text_encoder(
            input_ids=aspect_desc_encoding['input_ids'],
            attention_mask=aspect_desc_encoding['attention_mask']
        )  # [B, L, D]
        
        # 取[CLS] token作为显式查询
        explicit_query = desc_features[:, 0, :].unsqueeze(1)  # [B, 1, D]
        
        # === Part C: 拼接 ===
        total_queries = torch.cat([implicit_queries, explicit_query], dim=1)  # [B, 9, D]
        
        return total_queries

# 测试
if __name__ == '__main__':
    from encoders import TextEncoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建编码器和查询生成器
    text_encoder = TextEncoder().to(device)
    query_generator = HybridQueryGenerator(
        num_aspects=3,
        hidden_dim=768,
        num_learnable_queries=8
    ).to(device)
    
    # 准备输入
    batch_size = 4
    aspect_ids = torch.randint(0, 3, (batch_size,)).to(device)
    aspect_desc_encoding = {
        'input_ids': torch.randint(0, 30000, (batch_size, 30)).to(device),
        'attention_mask': torch.ones(batch_size, 30).to(device)
    }
    
    # 生成查询
    queries = query_generator(aspect_ids, aspect_desc_encoding, text_encoder)
    print(f"混合查询形状: {queries.shape}")  # [4, 9, 768]
    print(f"✅ 查询生成成功: {8}个隐式查询 + {1}个显式查询")
