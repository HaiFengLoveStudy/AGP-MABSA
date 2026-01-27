# src/losses/total_loss.py
import torch
import torch.nn as nn
from .classification import ClassificationLoss
from .infonce import InfoNCELoss
from .supcon import MultiViewSupConLoss
from .auxiliary import AuxiliaryAspectLoss

class TotalLoss(nn.Module):
    """联合损失函数"""
    
    def __init__(
        self,
        alpha=1.0,  # InfoNCE权重
        beta=0.5,   # SupCon权重
        gamma=0.3,  # 辅助任务权重
        temperature_infonce=0.07,
        temperature_supcon=0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 各个损失函数
        self.cls_loss = ClassificationLoss()
        self.infonce_loss = InfoNCELoss(temperature=temperature_infonce)
        self.supcon_loss = MultiViewSupConLoss(temperature=temperature_supcon)
        self.aux_loss = AuxiliaryAspectLoss()
    
    def forward(self, outputs, labels, aspect_ids, pair_id_mask):
        """
        Args:
            outputs: 模型输出字典
            labels: [B] 情感标签
            aspect_ids: [B] 方面ID
            pair_id_mask: [B, B] pair_id掩码
        
        Returns:
            total_loss: scalar
            loss_dict: 各项损失的字典
        """
        # 1. 分类损失
        loss_cls = self.cls_loss(outputs['sentiment_logits'], labels)
        
        # 2. InfoNCE损失
        loss_infonce, infonce_info = self.infonce_loss(
            outputs['h_text'],
            outputs['h_image'],
            pair_id_mask
        )
        
        # 3. SupCon损失
        loss_supcon = self.supcon_loss(
            outputs['h_text'],
            outputs['h_image'],
            labels,
            aspect_ids
        )
        
        # 4. 辅助任务损失
        loss_aux, aux_info = self.aux_loss(
            outputs['aspect_logits_text'],
            outputs['aspect_logits_image'],
            aspect_ids
        )
        
        # 5. 总损失
        total_loss = (
            loss_cls +
            self.alpha * loss_infonce +
            self.beta * loss_supcon +
            self.gamma * loss_aux
        )
        
        # 损失字典
        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'infonce': loss_infonce.item(),
            'infonce_t2i': infonce_info['loss_t2i'],
            'infonce_i2t': infonce_info['loss_i2t'],
            'supcon': loss_supcon.item(),
            'aux': loss_aux.item(),
            'aux_text': aux_info['loss_text'],
            'aux_image': aux_info['loss_image']
        }
        
        return total_loss, loss_dict

# 测试
if __name__ == '__main__':
    import torch.nn.functional as F
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_loss_fn = TotalLoss(
        alpha=1.0,
        beta=0.5,
        gamma=0.3
    ).to(device)
    
    # 模拟模型输出
    batch_size = 8
    outputs = {
        'sentiment_logits': torch.randn(batch_size, 3).to(device),
        'aspect_logits_text': torch.randn(batch_size, 3).to(device),
        'aspect_logits_image': torch.randn(batch_size, 3).to(device),
        'h_text': F.normalize(torch.randn(batch_size, 256), p=2, dim=1).to(device),
        'h_image': F.normalize(torch.randn(batch_size, 256), p=2, dim=1).to(device)
    }
    
    labels = torch.randint(0, 3, (batch_size,)).to(device)
    aspect_ids = torch.randint(0, 3, (batch_size,)).to(device)
    pair_id_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool).to(device)
    
    # 计算损失
    total_loss, loss_dict = total_loss_fn(outputs, labels, aspect_ids, pair_id_mask)
    
    print("=== 损失函数测试 ===")
    print(f"总损失: {loss_dict['total']:.4f}")
    print(f"  分类损失: {loss_dict['cls']:.4f}")
    print(f"  InfoNCE损失: {loss_dict['infonce']:.4f}")
    print(f"    - T2I: {loss_dict['infonce_t2i']:.4f}")
    print(f"    - I2T: {loss_dict['infonce_i2t']:.4f}")
    print(f"  SupCon损失: {loss_dict['supcon']:.4f}")
    print(f"  辅助损失: {loss_dict['aux']:.4f}")
    print(f"    - Text: {loss_dict['aux_text']:.4f}")
    print(f"    - Image: {loss_dict['aux_image']:.4f}")
    print(f"✅ 联合损失函数测试成功")
