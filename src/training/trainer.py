# src/training/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

class Trainer:
    """AGP模型训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        dev_loader,
        loss_fn,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.config = config
        
        # 优化器和调度器
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.get('use_amp', False) else None
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_dev_f1 = 0.0
        
        # 日志
        self.train_history = []
        self.dev_history = []
    
    def _create_optimizer(self):
        """创建优化器（分层学习率）"""
        # 分组参数
        backbone_params = []
        new_params = []
        
        # BERT参数（冻结的层不加入优化）
        for name, param in self.model.text_encoder.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        
        # ViT LoRA参数
        for name, param in self.model.image_encoder.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        
        # 新增模块参数
        for module in [
            self.model.query_generator,
            self.model.text_cross_attn,
            self.model.image_cross_attn,
            self.model.text_pooling,
            self.model.image_pooling,
            self.model.text_proj,
            self.model.image_proj,
            self.model.sentiment_classifier,
            self.model.aspect_classifier_text,
            self.model.aspect_classifier_image
        ]:
            new_params.extend(list(module.parameters()))
        
        # 优化器配置
        optimizer = AdamW([
            {'params': backbone_params, 'lr': self.config['lr_backbone']},
            {'params': new_params, 'lr': self.config['lr_head']}
        ], weight_decay=self.config['weight_decay'])
        
        # 学习率调度器（带warmup）
        num_training_steps = len(self.train_loader) * self.config['num_epochs']
        num_warmup_steps = int(num_training_steps * self.config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"✅ 优化器配置:")
        print(f"  Backbone LR: {self.config['lr_backbone']}")
        print(f"  Head LR: {self.config['lr_head']}")
        print(f"  Warmup steps: {num_warmup_steps}")
        print(f"  Total steps: {num_training_steps}")
        
        return optimizer, scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播（混合精度）
            if self.scaler:
                with autocast():
                    outputs = self.model(batch)
                    loss, loss_dict = self.loss_fn(
                        outputs,
                        batch['labels'],
                        batch['aspect_ids'],
                        batch['pair_id_mask']
                    )
            else:
                outputs = self.model(batch)
                loss, loss_dict = self.loss_fn(
                    outputs,
                    batch['labels'],
                    batch['aspect_ids'],
                    batch['pair_id_mask']
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            # 计算准确率
            preds = outputs['sentiment_logits'].argmax(dim=1)
            acc = (preds == batch['labels']).float().mean().item()
            
            epoch_losses.append(loss_dict)
            epoch_metrics.append({'acc': acc})
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'acc': acc,
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        # 计算epoch平均指标
        avg_loss = {k: np.mean([d[k] for d in epoch_losses]) 
                   for k in epoch_losses[0].keys()}
        avg_acc = np.mean([m['acc'] for m in epoch_metrics])
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self):
        """验证集评估"""
        self.model.eval()
        all_preds = []
        all_labels = []
        epoch_losses = []
        
        pbar = tqdm(self.dev_loader, desc=f"Epoch {self.epoch+1} [Dev]")
        
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(batch)
            loss, loss_dict = self.loss_fn(
                outputs,
                batch['labels'],
                batch['aspect_ids'],
                batch['pair_id_mask']
            )
            
            preds = outputs['sentiment_logits'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            epoch_losses.append(loss_dict)
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        avg_loss = {k: np.mean([d[k] for d in epoch_losses]) 
                   for k in epoch_losses[0].keys()}
        
        return avg_loss, {'acc': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 - {self.config['num_epochs']} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            dev_loss, dev_metrics = self.evaluate()
            
            # 记录历史
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'acc': train_acc
            })
            self.dev_history.append({
                'epoch': epoch + 1,
                'loss': dev_loss,
                **dev_metrics
            })
            
            # 打印摘要
            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"Train Loss: {train_loss['total']:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Dev Loss: {dev_loss['total']:.4f} | Dev Acc: {dev_metrics['acc']:.4f} | "
                  f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f}")
            
            # 保存最佳模型
            if dev_metrics['macro_f1'] > self.best_dev_f1:
                self.best_dev_f1 = dev_metrics['macro_f1']
                self.save_checkpoint(
                    os.path.join(self.config['save_dir'], 'best_model.pt'),
                    is_best=True
                )
                print(f"✅ 保存最佳模型 (F1: {self.best_dev_f1:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(
                    os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch+1}.pt')
                )
        
        print(f"\n{'='*60}")
        print(f"训练完成！最佳Dev F1: {self.best_dev_f1:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dev_f1': self.best_dev_f1,
            'config': self.config,
            'train_history': self.train_history,
            'dev_history': self.dev_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"{'Best model' if is_best else 'Checkpoint'} saved to {path}")
