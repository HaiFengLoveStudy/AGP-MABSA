# src/losses/supcon.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectAwareSupConLoss(nn.Module):
    """方面感知的监督对比学习损失"""
    
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels, aspect_ids):
        """
        Args:
            features: [B, D'] 归一化的特征（h_text或h_image）
            labels: [B] 情感标签
            aspect_ids: [B] 方面ID
        
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. 计算相似度矩阵 [B, B]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 2. 定义正样本mask：情感相同 AND 方面相同 (排除自己)
        label_match = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
        aspect_match = aspect_ids.unsqueeze(1) == aspect_ids.unsqueeze(0)  # [B, B]
        pos_mask = (label_match & aspect_match).float()
        pos_mask.fill_diagonal_(0)  # 排除自己
        
        # 3. 定义硬负例权重
        weights = torch.ones_like(sim_matrix)
        
        # 情况A: 同方面、异情感（最难负例）-> 权重 2.0
        hard_senti_mask = aspect_match & (~label_match)
        weights[hard_senti_mask] = 2.0
        
        # 情况B: 同情感、异方面（方面混淆）-> 权重 1.5
        hard_aspect_mask = label_match & (~aspect_match)
        weights[hard_aspect_mask] = 1.5
        
        # 4. 计算对比损失
        # 为数值稳定性，减去最大值
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # 计算加权的exp
        exp_logits = torch.exp(logits) * weights
        
        # 分母：所有负样本的加权和（排除自己）
        mask_self = torch.eye(batch_size, device=device).bool()
        exp_logits.masked_fill_(mask_self, 0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 对每个正样本计算损失
        # 只对有正样本的样本计算
        pos_per_sample = pos_mask.sum(dim=1)
        valid_samples = pos_per_sample > 0
        
        if valid_samples.sum() == 0:
            # 如果batch中没有正样本对，返回0
            return torch.tensor(0.0, device=device)
        
        # 计算平均对数概率
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_per_sample + 1e-8)
        
        # 只对有正样本的样本计算损失
        loss = -mean_log_prob_pos[valid_samples].mean()
        
        return loss * (self.temperature / self.base_temperature)

class MultiViewSupConLoss(nn.Module):
    """多视图SupCon：同时对文本和图像特征做对比"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.supcon = AspectAwareSupConLoss(temperature=temperature)
    
    def forward(self, h_text, h_image, labels, aspect_ids):
        """
        Args:
            h_text: [B, D'] 文本投影特征
            h_image: [B, D'] 图像投影特征
            labels: [B] 情感标签
            aspect_ids: [B] 方面ID
        
        Returns:
            loss: scalar
        """
        # 堆叠为多视图 [2B, D']
        features = torch.cat([h_text, h_image], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        aspect_ids = torch.cat([aspect_ids, aspect_ids], dim=0)
        
        loss = self.supcon(features, labels, aspect_ids)
        
        return loss

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    supcon_loss = MultiViewSupConLoss(temperature=0.1).to(device)
    
    # 模拟特征
    h_text = F.normalize(torch.randn(8, 256), p=2, dim=1).to(device)
    h_image = F.normalize(torch.randn(8, 256), p=2, dim=1).to(device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1]).to(device)
    aspect_ids = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1]).to(device)
    
    loss = supcon_loss(h_text, h_image, labels, aspect_ids)
    print(f"Aspect-Aware SupCon损失: {loss.item():.4f}")
    print(f"✅ SupCon损失计算成功")
