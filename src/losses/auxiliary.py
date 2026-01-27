# src/losses/auxiliary.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryAspectLoss(nn.Module):
    """辅助任务：方面分类损失"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, aspect_logits_text, aspect_logits_image, aspect_ids):
        """
        Args:
            aspect_logits_text: [B, num_aspects] 文本方面预测
            aspect_logits_image: [B, num_aspects] 图像方面预测
            aspect_ids: [B] 真实方面标签
        
        Returns:
            loss: scalar
        """
        loss_text = self.criterion(aspect_logits_text, aspect_ids)
        loss_image = self.criterion(aspect_logits_image, aspect_ids)
        
        # 取平均
        loss = (loss_text + loss_image) / 2
        
        return loss, {'loss_text': loss_text.item(), 'loss_image': loss_image.item()}

# 测试
if __name__ == '__main__':
    aux_loss = AuxiliaryAspectLoss()
    
    # 模拟预测
    aspect_logits_text = torch.randn(4, 3)
    aspect_logits_image = torch.randn(4, 3)
    aspect_ids = torch.tensor([0, 1, 2, 0])
    
    loss, info = aux_loss(aspect_logits_text, aspect_logits_image, aspect_ids)
    print(f"辅助损失: {loss.item():.4f}")
    print(f"文本损失: {info['loss_text']:.4f}")
    print(f"图像损失: {info['loss_image']:.4f}")
