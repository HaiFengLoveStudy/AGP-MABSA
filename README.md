# AGP-MABSA：方面引导的多模态情感分析

本项目实现了AGP（Aspect-Guided Prompting）方法，用于多模态方面级情感分析（MABSA）任务。

## 🎯 项目特点

- **混合查询生成**：结合隐式查询（可学习参数）和显式查询（LLM扩写的方面描述）
- **方面引导交叉注意力**：从文本和图像中提取方面相关特征
- **注意力池化**：智能聚合多查询特征
- **多任务学习**：联合优化情感分类、跨模态对齐、方面识别
- **改进的对比学习**：Aspect-Aware SupCon损失，避免方面混淆

## 📁 项目结构

```
AGP-MABSA/
├── data/                      # 数据目录
│   ├── raw/                   # 原始JSONL数据
│   ├── processed/             # LLM扩写后的数据
│   └── images/                # 图像文件
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── dataset.py         # 数据集类
│   │   ├── llm_expansion.py   # LLM知识扩写
│   │   └── create_dataloaders.py
│   ├── models/                # 模型模块
│   │   ├── encoders.py        # BERT和ViT编码器
│   │   ├── query_generator.py # 混合查询生成器
│   │   ├── attention.py       # 交叉注意力
│   │   ├── pooling.py         # 注意力池化
│   │   ├── projector.py       # 投影头和分类器
│   │   └── agp_model.py       # 完整AGP模型
│   ├── losses/                # 损失函数
│   │   ├── classification.py  # 分类损失
│   │   ├── infonce.py         # InfoNCE损失
│   │   ├── supcon.py          # SupCon损失
│   │   ├── auxiliary.py       # 辅助任务损失
│   │   └── total_loss.py      # 联合损失
│   ├── training/              # 训练模块
│   │   └── trainer.py         # 训练器
│   └── evaluation/            # 评估模块
│       ├── metrics.py         # 评估指标
│       └── error_analysis.py  # 错误分析
├── configs/                   # 配置文件
│   └── training_config.yaml
├── models/                    # 模型保存
│   ├── pretrained/            # 预训练模型
│   └── checkpoints/           # 训练检查点
├── train.py                   # 训练脚本
├── evaluate.py                # 评估脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n agp_mabsa python=3.9
conda activate agp_mabsa

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将原始JSONL数据放入 `data/raw/` 目录：
- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`

将图像文件放入 `data/images/` 目录。

### 3. LLM知识扩写

```bash
# 配置OpenAI API密钥
export OPENAI_API_KEY="your-api-key"

# 运行扩写脚本
python src/data/llm_expansion.py
```

### 4. 训练模型

```bash
python train.py
```

训练配置在 `configs/training_config.yaml` 中修改。

### 5. 评估模型

```bash
python evaluate.py
```

## 📊 模型架构

### 核心组件

1. **编码器**
   - 文本编码器：BERT-base（部分冻结）
   - 图像编码器：ViT-base（LoRA微调）

2. **混合查询生成器**
   - 8个隐式查询（可学习参数）
   - 1个显式查询（LLM扩写描述）

3. **方面引导交叉注意力**
   - 从文本/图像中提取方面相关特征
   - Transformer结构（Multi-head Attention + FFN）

4. **注意力池化**
   - 使用可学习聚合向量智能加权

5. **投影头和分类器**
   - 对比学习投影头（768→256）
   - 情感分类器（3分类）
   - 方面分类器（辅助任务）

### 损失函数

总损失：`L = L_cls + α·L_InfoNCE + β·L_SupCon + γ·L_aux`

- **L_cls**: 情感分类交叉熵损失
- **L_InfoNCE**: 跨模态对齐损失（带Pair-ID掩码）
- **L_SupCon**: Aspect-Aware监督对比损失
- **L_aux**: 方面分类辅助损失

默认权重：α=1.0, β=0.5, γ=0.3

## ⚙️ 超参数配置

主要超参数在 `configs/training_config.yaml`:

```yaml
# 模型参数
hidden_dim: 768
proj_dim: 256
num_queries: 8
freeze_bert_layers: 10
lora_rank: 8

# 训练参数
num_epochs: 15
batch_size: 32
lr_backbone: 1.0e-5
lr_head: 1.0e-4

# 损失权重
alpha: 1.0    # InfoNCE
beta: 0.5     # SupCon
gamma: 0.3    # Auxiliary
```

## 📈 预期性能

在Twitter2015/2017数据集上：
- **准确率**: 68-72%
- **Macro F1**: 66-70%
- **训练时间**: 2-3小时（单A100 GPU，15 epochs）

## 🔧 调试建议

### 常见问题

1. **Loss不下降**：降低学习率，检查pair_id_mask
2. **显存溢出**：减小batch_size或num_queries
3. **过拟合**：增大dropout和weight_decay
4. **方面混淆**：增大辅助任务权重gamma

### 性能优化

- 增加`num_workers`加速数据加载
- 使用混合精度训练（`use_amp: true`）
- 梯度累积应对小显存

## 📚 参考文档

- `AGP_EXPERIMENT_PROCEDURE.md`: 完整实验步骤
- `AGP METHOD GUIDE.md`: 方法论和设计理念
- `CONTRASTIVE_LEARNING_ANALYSIS.md`: 对比学习改进建议

## 📄 许可证

本项目遵循研究和教育使用许可。

## 👥 联系方式

如有问题，请查阅文档或提交Issue。

---

**最后更新**: 2026-01-27
**版本**: 1.0
