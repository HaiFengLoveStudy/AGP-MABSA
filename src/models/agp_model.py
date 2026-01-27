# src/models/agp_model.py
import torch
import torch.nn as nn
from .encoders import TextEncoder, ImageEncoder
from .query_generator import HybridQueryGenerator
from .attention import AspectGuidedCrossAttention
from .pooling import AttentionPooling
from .projector import ProjectionHead, SentimentClassifier, AspectClassifier

class AGPModel(nn.Module):
    """完整的AGP模型"""
    
    def __init__(
        self,
        num_aspects,
        hidden_dim=768,
        proj_dim=256,
        num_queries=8,
        num_classes=3,
        freeze_bert_layers=10,
        use_lora=True,
        lora_rank=8
    ):
        super().__init__()
        
        # 编码器
        self.text_encoder = TextEncoder(freeze_layers=freeze_bert_layers)
        self.image_encoder = ImageEncoder(use_lora=use_lora, lora_rank=lora_rank)
        
        # 查询生成器
        self.query_generator = HybridQueryGenerator(
            num_aspects=num_aspects,
            hidden_dim=hidden_dim,
            num_learnable_queries=num_queries
        )
        
        # 交叉注意力
        self.text_cross_attn = AspectGuidedCrossAttention(hidden_dim=hidden_dim)
        self.image_cross_attn = AspectGuidedCrossAttention(hidden_dim=hidden_dim)
        
        # 注意力池化
        self.text_pooling = AttentionPooling(hidden_dim=hidden_dim)
        self.image_pooling = AttentionPooling(hidden_dim=hidden_dim)
        
        # 投影头（用于对比学习）
        self.text_proj = ProjectionHead(hidden_dim, proj_dim)
        self.image_proj = ProjectionHead(hidden_dim, proj_dim)
        
        # 分类器
        self.sentiment_classifier = SentimentClassifier(
            input_dim=hidden_dim * 2,
            num_classes=num_classes
        )
        
        # 辅助任务：方面分类器
        self.aspect_classifier_text = AspectClassifier(hidden_dim, num_aspects)
        self.aspect_classifier_image = AspectClassifier(hidden_dim, num_aspects)
    
    def forward(self, batch):
        """
        Args:
            batch: dict包含所有输入
        
        Returns:
            dict包含所有输出
        """
        # 1. 编码文本和图像
        text_features = self.text_encoder(
            batch['text_input_ids'],
            batch['text_attention_mask']
        )  # [B, L, D]
        
        image_features = self.image_encoder(batch['images'])  # [B, P, D]
        
        # 2. 生成混合查询
        aspect_desc_encoding = {
            'input_ids': batch['aspect_input_ids'],
            'attention_mask': batch['aspect_attention_mask']
        }
        queries = self.query_generator(
            batch['aspect_ids'],
            aspect_desc_encoding,
            self.text_encoder
        )  # [B, m, D]
        
        # 3. 交叉注意力提取方面相关特征
        # 注意：需要反转attention_mask（1->False, 0->True）
        text_padding_mask = (batch['text_attention_mask'] == 0)
        
        Z_text = self.text_cross_attn(
            queries=queries,
            keys=text_features,
            values=text_features,
            key_padding_mask=text_padding_mask
        )  # [B, m, D]
        
        Z_image = self.image_cross_attn(
            queries=queries,
            keys=image_features,
            values=image_features
        )  # [B, m, D]
        
        # 4. 注意力池化
        g_text, text_attn_weights = self.text_pooling(Z_text)  # [B, D]
        g_image, image_attn_weights = self.image_pooling(Z_image)  # [B, D]
        
        # 5. 投影到对比学习空间
        h_text = self.text_proj(g_text)  # [B, D']
        h_image = self.image_proj(g_image)  # [B, D']
        
        # 6. 拼接多模态特征
        multimodal_feature = torch.cat([g_text, g_image], dim=1)  # [B, 2D]
        
        # 7. 情感分类
        sentiment_logits = self.sentiment_classifier(multimodal_feature)  # [B, 3]
        
        # 8. 辅助任务：方面分类
        aspect_logits_text = self.aspect_classifier_text(g_text)
        aspect_logits_image = self.aspect_classifier_image(g_image)
        
        return {
            'sentiment_logits': sentiment_logits,
            'aspect_logits_text': aspect_logits_text,
            'aspect_logits_image': aspect_logits_image,
            'h_text': h_text,
            'h_image': h_image,
            'g_text': g_text,
            'g_image': g_image,
            'Z_text': Z_text,
            'Z_image': Z_image,
            'text_attn_weights': text_attn_weights,
            'image_attn_weights': image_attn_weights
        }

# 测试完整模型
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AGPModel(
        num_aspects=3,
        num_queries=8
    ).to(device)
    
    # 模拟一个batch
    batch = {
        'text_input_ids': torch.randint(0, 30000, (4, 80)).to(device),
        'text_attention_mask': torch.ones(4, 80).to(device),
        'aspect_input_ids': torch.randint(0, 30000, (4, 30)).to(device),
        'aspect_attention_mask': torch.ones(4, 30).to(device),
        'images': torch.randn(4, 3, 224, 224).to(device),
        'aspect_ids': torch.randint(0, 3, (4,)).to(device)
    }
    
    # 前向传播
    with torch.no_grad():
        outputs = model(batch)
    
    print("=== 模型输出 ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 参数统计 ===")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"✅ 模型构建成功！")
