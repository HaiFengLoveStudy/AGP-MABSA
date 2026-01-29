# AGP-MABSA 快速启动指南

## 🚀 5分钟快速开始

### 1️⃣ 验证环境
```bash
python verify_setup.py
```

### 2️⃣ 安装依赖
```bash
conda create -n agp_mabsa python=3.9
conda activate agp_mabsa
pip install -r requirements.txt
```

### 3️⃣ 准备数据

**数据格式示例** (`data/raw/train.jsonl`):
```json
{"sample_id": "001", "text": "The food was great!", "aspect": "food", "image_paths": ["001.jpg"], "label": 2, "pair_id": "001"}
{"sample_id": "002", "text": "The food was great!", "aspect": "service", "image_paths": ["001.jpg"], "label": 1, "pair_id": "001"}
```

**目录结构**:
```
data/
├── raw/
│   ├── train.jsonl  ← 原始训练数据
│   ├── dev.jsonl    ← 原始验证数据
│   └── test.jsonl   ← 原始测试数据
└── images/
    └── twitter2015_images/
        ├── 001.jpg
        ├── 002.jpg
        └── ...
```

### 4️⃣ LLM知识扩写

**配置API密钥**:
```bash
export OPENAI_API_KEY="sk-..."
```

**运行扩写**:
```bash
python src/data/llm_expansion.py
```

**输出**:
- `data/processed/train_expanded.jsonl`
- `data/processed/dev_expanded.jsonl`
- `data/processed/test_expanded.jsonl`

### 5️⃣ 训练模型

**修改配置** (可选):
```bash
vim configs/training_config.yaml
```

**开始训练**:
```bash
# 前台运行
python train.py

# 后台运行（推荐）
nohup python train.py > logs/training_202601290215.log 2>&1 &

# 查看日志
tail -f logs/training_202601290215.log
```

**训练过程**:
```
Epoch 1: Train Loss: 3.24 | Dev F1: 0.40
Epoch 2: Train Loss: 2.78 | Dev F1: 0.48
...
Epoch 15: Train Loss: 1.23 | Dev F1: 0.69
✅ Best model saved!
```

### 6️⃣ 评估模型

```bash
python evaluate.py
```

**输出结果**:
- 控制台: 详细的评估指标
- `results/test_results.json`: JSON格式结果
- `results/confusion_matrix.png`: 混淆矩阵图

## 🎯 配置调优

### 减少显存占用
```yaml
batch_size: 16      # 从32降到16
num_queries: 6      # 从8降到6
lora_rank: 4        # 从8降到4
```

### 加速训练
```yaml
num_workers: 8      # 增加数据加载线程
use_amp: true       # 使用混合精度
batch_size: 64      # 增大batch（如果显存足够）
```

### 提升性能
```yaml
alpha: 1.5          # 增强跨模态对齐
beta: 0.8           # 增强情感对比
gamma: 0.5          # 增强方面识别
lr_backbone: 2e-5   # 提高backbone学习率
```

## 📊 监控训练

### 查看训练历史
```python
import json
with open('models/checkpoints/training_history.json') as f:
    history = json.load(f)
print(f"Best Dev F1: {max(h['macro_f1'] for h in history['dev'])}")
```

### 检查检查点
```bash
ls -lh models/checkpoints/
# best_model.pt           - 最佳模型
# checkpoint_epoch_5.pt   - 第5个epoch
# checkpoint_epoch_10.pt  - 第10个epoch
```

## 🐛 常见问题

### 问题1: CUDA out of memory
**解决**: 减小batch_size
```yaml
batch_size: 16  # 或更小
```

### 问题2: Loss = NaN
**解决**: 降低学习率
```yaml
lr_backbone: 5e-6
lr_head: 5e-5
max_grad_norm: 0.5
```

### 问题3: 图像加载失败
**检查**: 图像路径是否正确
```bash
# 应该能访问到
ls data/images/twitter2015_images/001.jpg
```

### 问题4: 过拟合
**解决**: 增加正则化
```yaml
weight_decay: 0.05  # 从0.01增加
# 在模型中增加dropout
```

## 🔬 测试单个模块

### 测试编码器
```bash
cd src/models
python encoders.py
```

### 测试查询生成器
```bash
python query_generator.py
```

### 测试完整模型
```bash
python agp_model.py
```

### 测试损失函数
```bash
cd ../losses
python total_loss.py
```

## 📈 实验技巧

### 1. 小规模测试
先用少量数据测试代码:
```bash
head -100 data/raw/train.jsonl > data/raw/train_small.jsonl
# 修改配置指向train_small.jsonl
python train.py  # 快速验证流程
```

### 2. 调试模式
在trainer.py中添加:
```python
if batch_idx == 0:  # 只跑第一个batch
    break
```

### 3. 可视化注意力权重
在模型输出中包含了attention weights:
```python
outputs['text_attn_weights']  # [B, 9]
outputs['image_attn_weights']  # [B, 9]
```

## 🎓 进阶使用

### 恢复训练
```python
# 在train.py中添加
checkpoint = torch.load('models/checkpoints/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### 消融实验
创建不同配置文件:
```bash
cp configs/training_config.yaml configs/ablation_no_supcon.yaml
# 修改: beta: 0.0
python train.py --config configs/ablation_no_supcon.yaml
```

### 推理单个样本
```python
from src.models.agp_model import AGPModel
model = AGPModel(...)
model.load_state_dict(torch.load('models/checkpoints/best_model.pt')['model_state_dict'])
model.eval()

# 准备单个样本
output = model(batch)
pred = output['sentiment_logits'].argmax(dim=1)
```

## 📚 学习路径

**Day 1**: 理解数据流
- 阅读 `dataset.py`
- 运行 `create_dataloaders.py`

**Day 2**: 理解模型结构
- 阅读 `encoders.py`, `query_generator.py`
- 运行单元测试

**Day 3**: 理解损失函数
- 阅读 `total_loss.py`
- 理解对比学习机制

**Day 4-5**: 运行完整训练
- 配置环境
- 运行训练
- 分析结果

## 🎉 成功标志

✅ `verify_setup.py` 全部通过  
✅ 数据加载器成功创建  
✅ 模型前向传播无错误  
✅ 训练loss稳定下降  
✅ Dev F1 达到 0.66+  
✅ 混淆矩阵合理  

---

**需要帮助?** 查看 `PROJECT_SUMMARY.md` 和 `AGP_EXPERIMENT_PROCEDURE.md`
