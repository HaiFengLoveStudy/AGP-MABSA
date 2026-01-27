# 模型检查点目录

此目录用于保存训练过程中的模型检查点文件。

## 文件说明

- `best_model.pt` - 验证集上表现最好的模型
- `checkpoint_epoch_N.pt` - 第N个epoch的检查点

## 检查点内容

每个检查点文件包含：
- 模型权重 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 学习率调度器状态 (`scheduler_state_dict`)
- 训练历史 (`train_history`, `dev_history`)
- 最佳验证F1分数 (`best_dev_f1`)
- 训练配置 (`config`)

## 使用方式

### 恢复训练
```python
checkpoint = torch.load('models/checkpoints/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 加载最佳模型
```python
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 注意事项

- 此目录中的文件通常较大（几百MB到几GB）
- 建议定期清理旧的检查点以节省空间
- 只保留最佳模型和最近的几个检查点即可
