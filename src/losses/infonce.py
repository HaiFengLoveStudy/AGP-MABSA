# src/losses/infonce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE跨模态对齐损失（带Pair-ID掩码）"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, h_text, h_image, pair_id_mask):
        """
        Args:
            h_text: [B, D'] L2归一化的文本特征
            h_image: [B, D'] L2归一化的图像特征
            pair_id_mask: [B, B] bool矩阵，True表示相同pair_id（需排除）
        
        Returns:
            loss: scalar
        """
        device = h_text.device
        batch_size = h_text.shape[0]
        
        # 1. 计算相似度矩阵
        # text-to-image: [B, B]
        sim_t2i = torch.matmul(h_text, h_image.T) / self.temperature
        # image-to-text: [B, B]
        sim_i2t = torch.matmul(h_image, h_text.T) / self.temperature
        
        # 2. 构建正样本mask（对角线）
        pos_mask = torch.eye(batch_size, device=device).bool()
        
        # 3. 构建负样本mask（排除自己和相同pair_id）
        # 负样本 = 不是自己 AND 不是同一pair_id
        neg_mask_t2i = ~(pos_mask | pair_id_mask)
        neg_mask_i2t = ~(pos_mask | pair_id_mask)
        
        # 4. 计算text-to-image损失
        # 分子：正样本相似度
        pos_sim_t2i = sim_t2i.diagonal()  # [B]
        
        # 分母：正样本 + 所有有效负样本
        # 为数值稳定，减去最大值
        logits_max_t2i, _ = torch.max(sim_t2i, dim=1, keepdim=True)
        exp_sim_t2i = torch.exp(sim_t2i - logits_max_t2i.detach())
        
        # 只保留有效的负样本
        exp_sim_t2i = exp_sim_t2i * neg_mask_t2i.float()
        # 加上正样本
        exp_sim_t2i.diagonal().copy_(torch.exp(pos_sim_t2i - logits_max_t2i.squeeze()))
        
        denominator_t2i = exp_sim_t2i.sum(dim=1)
        loss_t2i = -torch.log(
            torch.exp(pos_sim_t2i - logits_max_t2i.squeeze()) / (denominator_t2i + 1e-8)
        ).mean()
        
        # 5. 计算image-to-text损失（对称）
        pos_sim_i2t = sim_i2t.diagonal()
        logits_max_i2t, _ = torch.max(sim_i2t, dim=1, keepdim=True)
        exp_sim_i2t = torch.exp(sim_i2t - logits_max_i2t.detach())
        exp_sim_i2t = exp_sim_i2t * neg_mask_i2t.float()
        exp_sim_i2t.diagonal().copy_(torch.exp(pos_sim_i2t - logits_max_i2t.squeeze()))
        
        denominator_i2t = exp_sim_i2t.sum(dim=1)
        loss_i2t = -torch.log(
            torch.exp(pos_sim_i2t - logits_max_i2t.squeeze()) / (denominator_i2t + 1e-8)
        ).mean()
        
        # 6. 双向平均
        loss = (loss_t2i + loss_i2t) / 2
        
        return loss, {'loss_t2i': loss_t2i.item(), 'loss_i2t': loss_i2t.item()}

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    infonce_loss = InfoNCELoss(temperature=0.07).to(device)
    
    # 模拟特征（已归一化）
    h_text = F.normalize(torch.randn(8, 256), p=2, dim=1).to(device)
    h_image = F.normalize(torch.randn(8, 256), p=2, dim=1).to(device)
    
    # 模拟pair_id_mask：样本0和1共享相同pair_id
    pair_id_mask = torch.zeros(8, 8, dtype=torch.bool).to(device)
    pair_id_mask[0, 1] = True
    pair_id_mask[1, 0] = True
    
    loss, info = infonce_loss(h_text, h_image, pair_id_mask)
    print(f"InfoNCE损失: {loss.item():.4f}")
    print(f"Text-to-Image损失: {info['loss_t2i']:.4f}")
    print(f"Image-to-Text损失: {info['loss_i2t']:.4f}")
    print(f"✅ InfoNCE损失计算成功")
