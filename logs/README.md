# 训练日志目录

此目录用于保存训练过程中的日志文件。

## 文件说明

- `training.log` - 训练过程的完整日志输出
- `tensorboard/` - TensorBoard可视化日志（如果启用）

## 日志内容

训练日志包含：
- 每个epoch的训练损失和准确率
- 验证集上的损失和指标
- 学习率变化
- 模型保存信息
- 错误和警告信息

## 使用方式

### 查看训练日志
```bash
tail -f logs/training.log
```

### 使用TensorBoard可视化
```bash
tensorboard --logdir=logs/tensorboard --port=6006
```

## 注意事项

- 日志文件会随着训练进行不断增长
- 建议定期清理旧的日志文件
- 重要的训练记录建议备份
