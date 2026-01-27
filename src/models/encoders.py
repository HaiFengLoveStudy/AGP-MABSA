# src/models/encoders.py
import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from peft import LoraConfig, get_peft_model

class TextEncoder(nn.Module):
    """BERT文本编码器（部分冻结）"""
    
    def __init__(self, model_name='bert-base-uncased', freeze_layers=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size  # 768
        
        # 冻结前N层
        if freeze_layers > 0:
            # BERT有12层（layer 0-11）
            for layer_idx in range(freeze_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
            
            print(f"✅ 冻结BERT前{freeze_layers}层，微调后{12-freeze_layers}层")
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            outputs: [B, L, D] token级别的特征
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state  # [B, L, 768]

class ImageEncoder(nn.Module):
    """ViT图像编码器（LoRA微调）"""
    
    def __init__(
        self,
        model_name='google/vit-base-patch16-224',
        use_lora=True,
        lora_rank=8,
        lora_alpha=16
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_dim = self.vit.config.hidden_size  # 768
        
        if use_lora:
            # 配置LoRA
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],  # 只对Q和V注入LoRA
                lora_dropout=0.1,
                bias="none"
            )
            
            # 应用LoRA
            self.vit = get_peft_model(self.vit, lora_config)
            
            # 打印可训练参数
            trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vit.parameters())
            print(f"✅ ViT应用LoRA (rank={lora_rank}, alpha={lora_alpha})")
            print(f"   可训练参数: {trainable_params:,} / {total_params:,} "
                  f"({100*trainable_params/total_params:.2f}%)")
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, 3, 224, 224]
        Returns:
            outputs: [B, P, D] patch级别的特征
        """
        outputs = self.vit(
            pixel_values=pixel_values,
            return_dict=True
        )
        # 返回所有patch特征（不包括CLS token）
        return outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]

# 测试编码器
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试文本编码器
    text_encoder = TextEncoder(freeze_layers=10).to(device)
    input_ids = torch.randint(0, 30000, (4, 80)).to(device)
    attention_mask = torch.ones(4, 80).to(device)
    text_features = text_encoder(input_ids, attention_mask)
    print(f"文本特征形状: {text_features.shape}")  # [4, 80, 768]
    
    # 测试图像编码器
    image_encoder = ImageEncoder(use_lora=True, lora_rank=8).to(device)
    images = torch.randn(4, 3, 224, 224).to(device)
    image_features = image_encoder(images)
    print(f"图像特征形状: {image_features.shape}")  # [4, 196, 768]
