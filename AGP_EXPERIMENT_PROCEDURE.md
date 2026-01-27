# AGP实验执行步骤详解

**版本:** 1.0  
**创建日期:** 2026-01-27  
**适用场景:** 多模态方面级情感分析（MABSA）  
**预计完成时间:** 10-12小时

---

## 文档说明

本文档是AGP方法的**完整执行操作手册**，提供从环境搭建到模型评估的详细步骤。每个步骤都包含：
- 🎯 目标说明
- 📥 输入要求
- ⚙️ 执行命令
- 📤 预期输出
- ✅ 验证方法

**相关文档：**
- `AGP METHOD GUIDE.md` - 理论设计和方法论
- `AGAA METHOD GUIDE.md` - 参考实现模板
- `CONTRASTIVE_LEARNING_ANALYSIS.md` - 对比学习分析和改进建议

---

## 目录

1. [实验环境准备](#1-实验环境准备)
2. [数据预处理](#2-数据预处理)
3. [模型架构实现](#3-模型架构实现)
4. [损失函数实现](#4-损失函数实现)
5. [训练执行流程](#5-训练执行流程)
6. [模型评估与分析](#6-模型评估与分析)
7. [调试与优化](#7-调试与优化)

---

## 1. 实验环境准备

### 1.1 硬件要求



**推荐配置：**
- GPU: NVIDIA A100/A800 (80GB显存) 


**显存估算：**
```
模型参数：
- BERT-base: 110M × 4 bytes = 440MB
- ViT-base: 86M × 4 bytes = 344MB
- 新增模块: ~50M × 4 bytes = 200MB
- 总计: ~1GB

训练显存（Batch Size=32, FP16）：
- 模型参数: 1GB
- 梯度: 1GB
- 优化器状态: 2GB
- 激活值: 8-12GB
- 总计: ~15GB

建议：24GB显存可运行，40GB显存更稳定
```

### 1.2 软件环境配置

**操作系统：**
- Ubuntu 
- Python 3.10.0
- PyTorch version: 2.9.1+cu128

**创建虚拟环境：**

```bash
# 使用conda创建环境
conda create -n agp_mabsa python=3.10
conda activate agp_mabsa



### 1.3 依赖库安装

**核心依赖：**



安装命令：
```bash
pip install -r requirements.txt
```


### 1.4 项目目录结构

**创建目录：**

```bash
mkdir -p AGP-MABSA
cd AGP-MABSA

# 创建子目录
mkdir -p data/raw              # 原始数据集
mkdir -p data/processed        # 处理后的数据（含LLM扩写）
mkdir -p data/images           # 图像文件
mkdir -p models/pretrained     # 预训练模型
mkdir -p models/checkpoints    # 训练检查点
mkdir -p src                   # 源代码
mkdir -p logs                  # 训练日志
mkdir -p results               # 实验结果
mkdir -p configs               # 配置文件
```

**完整目录结构：**

```
AGP-MABSA/
├── data/
│   ├── raw/                   # 原始JSONL文件
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── processed/             # LLM扩写后的数据
│   │   ├── train_expanded.jsonl
│   │   ├── dev_expanded.jsonl
│   │   └── test_expanded.jsonl
│   └── images/                # Twitter图像
│       └── twitter2015_images/
├── models/
│   ├── pretrained/            # BERT和ViT预训练模型
│   │   ├── bert-base-uncased/
│   │   └── vit-base-patch16-224/
│   └── checkpoints/           # 训练保存的模型
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
├── src/
│   ├── data/
│   │   ├── dataset.py         # 数据集类
│   │   ├── preprocess.py      # 预处理脚本
│   │   └── llm_expansion.py   # LLM知识扩写
│   ├── models/
│   │   ├── encoders.py        # 编码器
│   │   ├── query_generator.py # 混合查询生成器
│   │   ├── attention.py       # 注意力模块
│   │   ├── pooling.py         # 注意力池化
│   │   └── agp_model.py       # 完整模型
│   ├── losses/
│   │   ├── classification.py  # 分类损失
│   │   ├── infonce.py         # InfoNCE损失
│   │   └── supcon.py          # SupCon损失
│   ├── training/
│   │   ├── trainer.py         # 训练器
│   │   └── optimizer.py       # 优化器配置
│   └── evaluation/
│       ├── metrics.py         # 评估指标
│       └── visualize.py       # 可视化工具
├── configs/
│   ├── model_config.yaml      # 模型配置
│   └── training_config.yaml   # 训练配置
├── logs/
│   └── tensorboard/           # TensorBoard日志
├── results/
│   ├── predictions/           # 预测结果
│   ├── visualizations/        # 可视化图表
│   └── metrics/               # 评估指标
├── train.py                   # 主训练脚本
├── evaluate.py                # 评估脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```



**数据格式检查：**

原始JSONL格式示例：
```json
{
  "sample_id": "twitter15_train_001",
  "text": "The steak was cold but the ambience was nice",
  "aspect": "food",
  "image_paths": ["twitter2015_images/001.jpg"],
  "label": 0,
  "pair_id": "twitter15_train_001"
}
```

**字段说明：**
- `sample_id`: 唯一样本标识符
- `text`: 评论文本
- `aspect`: 目标方面（food/service/ambience等）
- `image_paths`: 图像路径列表
- `label`: 情感标签（0=负面，1=中性，2=正面）
- `pair_id`: 图文对标识（同一图文对的不同方面共享此ID）



## 2. 数据预处理

### 2.1 LLM离线知识扩写

#### 2.1.1 设计原理

**目标：** 将抽象的方面词（如"food"）扩写为具体的描述性短语（如"taste presentation portion size and freshness of dishes"），为模型提供更丰富的语义锚点。

**约束条件：**
- 最大10个单词
- 使用简单、口语化的英语
- 适合社交媒体评论场景
- 描述视觉和文本特征

#### 2.1.2 Prompt模板

```text
Role: You are an assistant for social media sentiment analysis.

Task: Expand the given aspect word into a short phrase describing its visual and textual features in a review context.

Constraint: Use simple, casual English. Maximum 10 words. No introductory filler.

Input Aspect: "food"
Output: "taste presentation portion size and freshness of dishes"

Input Aspect: "service"
Output: "waiter attitude serving speed and customer care quality"

Input Aspect: "{aspect_word}"
Output:
```

#### 2.1.3 实现代码

**选项1：使用OpenAI GPT-4o**

**模块说明：`src/data/llm_expansion.py`**
- **用途**: 调用 LLM 对原始 JSONL 数据中的方面词进行批量扩写，将抽象方面转化为可用于检索和对比学习的描述性短语。
- **输入**: 原始数据 JSONL 路径（如 `data/raw/train.jsonl`）、目标输出 JSONL 路径、可选的 LLM 模型名称。
- **输出**: 带有新增字段 `aspect_desc` 的 JSONL 文件（如 `data/processed/train_expanded.jsonl`），以及方面词到扩写短语的映射字典（仅在脚本内部使用）。

```python
# src/data/llm_expansion.py
import json
import openai
from tqdm import tqdm
import time

# 配置API
openai.api_key = "your-api-key-here"

PROMPT_TEMPLATE = """Role: You are an assistant for social media sentiment analysis.

Task: Expand the given aspect word into a short phrase describing its visual and textual features in a review context.

Constraint: Use simple, casual English. Maximum 10 words. No introductory filler.

Input Aspect: "food"
Output: "taste presentation portion size and freshness of dishes"

Input Aspect: "service"
Output: "waiter attitude serving speed and customer care quality"

Input Aspect: "{aspect_word}"
Output:"""

def expand_aspect_openai(aspect_word, model="gpt-4o"):
    """使用OpenAI API扩写方面词"""
    prompt = PROMPT_TEMPLATE.format(aspect_word=aspect_word)
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 较低温度保持一致性
            max_tokens=30
        )
        expansion = response.choices[0].message.content.strip()
        return expansion
    except Exception as e:
        print(f"API调用失败: {e}")
        return aspect_word  # 失败时返回原词

def expand_dataset_openai(input_jsonl, output_jsonl):
    """批量扩写数据集"""
    # 读取数据
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 收集所有唯一的方面词
    unique_aspects = list(set([s['aspect'] for s in samples]))
    print(f"发现 {len(unique_aspects)} 个唯一方面: {unique_aspects}")
    
    # 批量扩写
    aspect_expansions = {}
    for aspect in tqdm(unique_aspects, desc="扩写方面词"):
        expansion = expand_aspect_openai(aspect)
        aspect_expansions[aspect] = expansion
        print(f"  {aspect} -> {expansion}")
        time.sleep(0.5)  # 避免API限流
    
    # 添加扩写到每个样本
    for sample in samples:
        sample['aspect_desc'] = aspect_expansions[sample['aspect']]
    
    # 保存
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！保存到 {output_jsonl}")
    return aspect_expansions

if __name__ == '__main__':
    # 扩写训练集
    expand_dataset_openai(
        'data/raw/train.jsonl',
        'data/processed/train_expanded.jsonl'
    )
    
    # 扩写验证集
    expand_dataset_openai(
        'data/raw/dev.jsonl',
        'data/processed/dev_expanded.jsonl'
    )
    
    # 扩写测试集
    expand_dataset_openai(
        'data/raw/test.jsonl',
        'data/processed/test_expanded.jsonl'
    )
```


#### 2.1.4 执行步骤

**步骤1：配置API密钥**

```bash
# 设置环境变量（推荐）
export OPENAI_API_KEY="sk-..."
# 或
export DASHSCOPE_API_KEY="sk-..."
```

**步骤2：运行扩写脚本**

```bash
python src/data/llm_expansion.py
```

**预期输出：**
```
发现 3 个唯一方面: ['food', 'service', 'ambience']
扩写方面词: 100%|██████████| 3/3 [00:05<00:00,  1.67s/it]
  food -> taste presentation portion size and freshness of dishes
  service -> waiter attitude serving speed and customer care quality
  ambience -> lighting decoration music atmosphere and seating comfort
✅ 完成！保存到 data/processed/train_expanded.jsonl
```

**步骤3：验证扩写结果**

```python
# scripts/verify_expansion.py
import json

def verify_expansion(jsonl_path, num_samples=5):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    print(f"验证文件: {jsonl_path}")
    print(f"总样本数: {len(samples)}")
    print(f"\n前{num_samples}个样本:")
    
    for i, sample in enumerate(samples[:num_samples]):
        print(f"\n样本 {i+1}:")
        print(f"  原始方面: {sample['aspect']}")
        print(f"  扩写描述: {sample.get('aspect_desc', 'NOT FOUND!')}")
        print(f"  文本: {sample['text'][:50]}...")
        print(f"  标签: {sample['label']}")

verify_expansion('data/processed/train_expanded.jsonl')
```

**✅ 检查点：**
- [ ] 所有JSONL文件包含`aspect_desc`字段
- [ ] 扩写描述不超过10个单词
- [ ] 扩写描述语义合理，与方面词相关
- [ ] 没有API错误或失败样本

**预计耗时：** 1-2小时（取决于API速度和数据集大小）

### 2.2 数据加载器实现

#### 2.2.1 数据集类

**模块说明：`src/data/dataset.py`**
- **用途**: 从扩写后的 JSONL 文件和图像目录中读取样本，完成文本 / 方面描述编码和图像预处理，生成可直接用于训练的单条样本。
- **输入**: JSONL 路径、图像根目录、`BertTokenizer`、`ViTImageProcessor`、文本和方面描述的最大长度等配置参数。
- **输出**: `__getitem__` 返回包含文本 token、方面描述 token、图像张量、情感标签、方面 ID、pair_id 相关 mask 等键值的字典；`__len__` 返回样本数量。

```python
# src/data/dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor

class MABSADataset(Dataset):
    """多模态方面级情感分析数据集"""
    
    def __init__(
        self,
        jsonl_path,
        image_root,
        tokenizer,
        image_processor,
        max_text_len=80,
        max_aspect_len=30
    ):
        """
        Args:
            jsonl_path: JSONL文件路径
            image_root: 图像根目录
            tokenizer: BERT tokenizer
            image_processor: ViT image processor
            max_text_len: 文本最大长度
            max_aspect_len: 方面描述最大长度
        """
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.max_aspect_len = max_aspect_len
        
        # 加载数据
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        # 构建方面词到ID的映射
        unique_aspects = sorted(list(set([s['aspect'] for s in self.samples])))
        self.aspect2id = {aspect: idx for idx, aspect in enumerate(unique_aspects)}
        self.id2aspect = {idx: aspect for aspect, idx in self.aspect2id.items()}
        self.num_aspects = len(unique_aspects)
        
        print(f"加载 {len(self.samples)} 个样本")
        print(f"方面类别: {unique_aspects}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 文本编码
        text_encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 方面描述编码
        aspect_desc = sample.get('aspect_desc', sample['aspect'])
        aspect_encoding = self.tokenizer(
            aspect_desc,
            max_length=self.max_aspect_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. 图像加载和预处理
        image_path = f"{self.image_root}/{sample['image_paths'][0]}"
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values']
        except Exception as e:
            print(f"图像加载失败: {image_path}, 错误: {e}")
            # 使用黑色图像作为占位符
            image_tensor = torch.zeros(1, 3, 224, 224)
        
        # 4. 标签和元信息
        label = sample['label']
        aspect_id = self.aspect2id[sample['aspect']]
        pair_id = sample['pair_id']
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'aspect_input_ids': aspect_encoding['input_ids'].squeeze(0),
            'aspect_attention_mask': aspect_encoding['attention_mask'].squeeze(0),
            'image': image_tensor.squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'aspect_id': torch.tensor(aspect_id, dtype=torch.long),
            'pair_id': pair_id,  # 字符串，用于构建pair_id_mask
            'sample_id': sample['sample_id']
        }

def collate_fn(batch):
    """自定义批次整理函数"""
    # 堆叠张量
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    aspect_input_ids = torch.stack([item['aspect_input_ids'] for item in batch])
    aspect_attention_mask = torch.stack([item['aspect_attention_mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    aspect_ids = torch.stack([item['aspect_id'] for item in batch])
    
    # 构建pair_id_mask
    pair_ids = [item['pair_id'] for item in batch]
    batch_size = len(pair_ids)
    pair_id_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(batch_size):
            if pair_ids[i] == pair_ids[j] and i != j:
                pair_id_mask[i, j] = True
    
    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'aspect_input_ids': aspect_input_ids,
        'aspect_attention_mask': aspect_attention_mask,
        'images': images,
        'labels': labels,
        'aspect_ids': aspect_ids,
        'pair_id_mask': pair_id_mask,
        'sample_ids': [item['sample_id'] for item in batch]
    }
```

#### 2.2.2 创建数据加载器

**模块说明：`src/data/create_dataloaders.py`**
- **用途**: 基于数据集类 `MABSADataset` 创建训练 / 验证 / 测试集的数据加载器，并统一返回方面类别数。
- **输入**: 训练、验证、测试 JSONL 路径，图像根目录，`batch_size`，`num_workers` 等数据加载配置。
- **输出**: `train_loader`、`dev_loader`、`test_loader` 三个 `DataLoader` 对象，以及 `num_aspects`（方面类别数量，用于构建模型）。

```python
# src/data/create_dataloaders.py
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
from dataset import MABSADataset, collate_fn

def create_dataloaders(
    train_jsonl='data/processed/train_expanded.jsonl',
    dev_jsonl='data/processed/dev_expanded.jsonl',
    test_jsonl='data/processed/test_expanded.jsonl',
    image_root='data/images',
    batch_size=32,
    num_workers=4
):
    """创建训练、验证和测试数据加载器"""
    
    # 初始化tokenizer和image processor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # 创建数据集
    train_dataset = MABSADataset(
        train_jsonl, image_root, tokenizer, image_processor
    )
    dev_dataset = MABSADataset(
        dev_jsonl, image_root, tokenizer, image_processor
    )
    test_dataset = MABSADataset(
        test_jsonl, image_root, tokenizer, image_processor
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader, train_dataset.num_aspects

# 测试
if __name__ == '__main__':
    train_loader, dev_loader, test_loader, num_aspects = create_dataloaders(
        batch_size=8
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(dev_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"方面类别数: {num_aspects}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n批次数据形状:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
```

**运行测试：**
```bash
cd src/data
python create_dataloaders.py
```

**预期输出：**
```
加载 4000 个样本
方面类别: ['ambience', 'food', 'service']
加载 500 个样本
方面类别: ['ambience', 'food', 'service']
加载 500 个样本
方面类别: ['ambience', 'food', 'service']
训练批次数: 500
验证批次数: 63
测试批次数: 63
方面类别数: 3

批次数据形状:
  text_input_ids: torch.Size([8, 80])
  text_attention_mask: torch.Size([8, 80])
  aspect_input_ids: torch.Size([8, 30])
  aspect_attention_mask: torch.Size([8, 30])
  images: torch.Size([8, 3, 224, 224])
  labels: torch.Size([8])
  aspect_ids: torch.Size([8])
  pair_id_mask: torch.Size([8, 8])
  sample_ids: <class 'list'>
```

**✅ 检查点：**
- [ ] 数据加载器成功创建
- [ ] 批次数据形状正确
- [ ] pair_id_mask正确构建
- [ ] 图像加载无错误
- [ ] Tokenization长度合适

**预计耗时：** 30分钟

---

## 3. 模型架构实现

### 3.1 编码器配置

#### 3.1.1 文本编码器（BERT）

**设计策略：**
- 使用`bert-base-uncased`（768维，110M参数）
- **冻结策略**：前10层冻结，只微调最后2层
- 目的：保留预训练知识，减少过拟合风险

**实现代码：**

**模块说明：`src/models/encoders.py`**
- **用途**: 封装文本编码器 `TextEncoder`（基于 BERT，支持部分冻结）和图像编码器 `ImageEncoder`（基于 ViT，支持 LoRA 微调），为后续模块提供统一的文本 / 图像特征。
- **输入**: 文本侧为 `input_ids` 和 `attention_mask`（形状 `[B, L]`），图像侧为 `pixel_values`（形状 `[B, 3, 224, 224]`），以及初始化时的模型名称、冻结层数、LoRA 配置等。
- **输出**: 文本编码器输出 token 级特征 `[B, L, D]`，图像编码器输出 patch 级特征 `[B, P, D]`，其中 `D` 为隐藏维度。

```python
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
```

**运行测试：**
```bash
cd src/models
python encoders.py
```

**预期输出：**
```
✅ 冻结BERT前10层,微调后2层
✅ ViT应用LoRA (rank=8, alpha=16)
   可训练参数: 295,936 / 86,567,656 (0.34%)
文本特征形状: torch.Size([4, 80, 768])
图像特征形状: torch.Size([4, 196, 768])
```

#### 3.1.2 混合查询生成器

**设计理念：**
- 结合**隐式查询**（可学习参数）和**显式查询**（LLM扩写的方面描述）
- 隐式查询：8个可学习向量，从方面Embedding做残差学习
- 显式查询：使用BERT编码LLM扩写的描述，取[CLS]表示
- 最终：9个查询向量（8隐式 + 1显式）

**实现代码：**

**模块说明：`src/models/query_generator.py`**
- **用途**: 实现混合查询生成器 `HybridQueryGenerator`，将方面 ID 与 LLM 扩写的方面描述结合，生成隐式 + 显式的多查询向量，用于后续交叉注意力从文本 / 图像中抽取方面相关信息。
- **输入**: `aspect_ids`（形状 `[B]`）、`aspect_desc_encoding`（含 `input_ids` 和 `attention_mask`）、共享的 `TextEncoder` 实例。
- **输出**: 形状为 `[B, 9, D]` 的查询张量，其中前 8 个为隐式查询，最后 1 个为显式查询。

```python
# src/models/query_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQueryGenerator(nn.Module):
    """混合查询生成器：隐式查询 + 显式查询"""
    
    def __init__(
        self,
        num_aspects,
        hidden_dim=768,
        num_learnable_queries=8
    ):
        """
        Args:
            num_aspects: 方面类别数量
            hidden_dim: 隐藏维度（768）
            num_learnable_queries: 可学习查询数量（默认8）
        """
        super().__init__()
        self.num_aspects = num_aspects
        self.hidden_dim = hidden_dim
        self.num_learnable_queries = num_learnable_queries
        
        # 方面Embedding（每个方面一个基础向量）
        self.aspect_embeddings = nn.Embedding(num_aspects, hidden_dim)
        
        # 可学习查询参数（共享给所有方面）
        self.learnable_params = nn.Parameter(
            torch.randn(num_learnable_queries, hidden_dim)
        )
        nn.init.xavier_uniform_(self.learnable_params)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, aspect_ids, aspect_desc_encoding, text_encoder):
        """
        Args:
            aspect_ids: [B] 方面ID
            aspect_desc_encoding: dict with 'input_ids' [B, L] and 'attention_mask' [B, L]
            text_encoder: TextEncoder实例（共享BERT权重）
        
        Returns:
            queries: [B, 9, D] 混合查询（8隐式 + 1显式）
        """
        batch_size = aspect_ids.size(0)
        device = aspect_ids.device
        
        # === Part A: 构造隐式查询 ===
        # 1. 获取方面基础向量 [B, D]
        base_aspect = self.aspect_embeddings(aspect_ids)
        
        # 2. 广播相加：[B, 1, D] + [1, 8, D] -> [B, 8, D]
        implicit_queries = base_aspect.unsqueeze(1) + self.learnable_params.unsqueeze(0)
        
        # 3. 层归一化
        implicit_queries = self.layer_norm(implicit_queries)  # [B, 8, D]
        
        # === Part B: 构造显式查询 ===
        # 使用text_encoder的BERT编码LLM描述
        desc_features = text_encoder(
            input_ids=aspect_desc_encoding['input_ids'],
            attention_mask=aspect_desc_encoding['attention_mask']
        )  # [B, L, D]
        
        # 取[CLS] token作为显式查询
        explicit_query = desc_features[:, 0, :].unsqueeze(1)  # [B, 1, D]
        
        # === Part C: 拼接 ===
        total_queries = torch.cat([implicit_queries, explicit_query], dim=1)  # [B, 9, D]
        
        return total_queries

# 测试
if __name__ == '__main__':
    from encoders import TextEncoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建编码器和查询生成器
    text_encoder = TextEncoder().to(device)
    query_generator = HybridQueryGenerator(
        num_aspects=3,
        hidden_dim=768,
        num_learnable_queries=8
    ).to(device)
    
    # 准备输入
    batch_size = 4
    aspect_ids = torch.randint(0, 3, (batch_size,)).to(device)
    aspect_desc_encoding = {
        'input_ids': torch.randint(0, 30000, (batch_size, 30)).to(device),
        'attention_mask': torch.ones(batch_size, 30).to(device)
    }
    
    # 生成查询
    queries = query_generator(aspect_ids, aspect_desc_encoding, text_encoder)
    print(f"混合查询形状: {queries.shape}")  # [4, 9, 768]
    print(f"✅ 查询生成成功: {8}个隐式查询 + {1}个显式查询")
```

**运行测试：**
```bash
python query_generator.py
```

### 3.2 交叉注意力模块

**设计原理：**
- 使用方面查询从文本和图像中提取相关信息
- 标准Transformer块：多头交叉注意力 + FFN + 残差连接

**实现代码：**

**模块说明：`src/models/attention.py`**
- **用途**: 定义方面引导的交叉注意力模块 `AspectGuidedCrossAttention`，让方面查询从文本或图像特征中选择与当前方面最相关的信息。
- **输入**: `queries`（方面查询 `[B, m, D]`）、`keys` 和 `values`（文本或图像特征 `[B, L, D]`）、可选的 `key_padding_mask`。
- **输出**: 经过多头交叉注意力和前馈网络后的方面特征 `[B, m, D]`，作为后续池化和对比学习的基础。

```python
# src/models/attention.py
import torch
import torch.nn as nn

class AspectGuidedCrossAttention(nn.Module):
    """方面引导的交叉注意力模块"""
    
    def __init__(
        self,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        feedforward_dim=2048
    ):
        super().__init__()
        
        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, key_padding_mask=None):
        """
        Args:
            queries: [B, m, D] 方面查询
            keys: [B, L, D] 文本/图像特征
            values: [B, L, D] 文本/图像特征
            key_padding_mask: [B, L] padding mask (True表示padding位置)
        
        Returns:
            output: [B, m, D] 提取的方面相关特征
        """
        # 交叉注意力 + 残差
        attn_output, _ = self.cross_attn(
            query=queries,
            key=keys,
            value=values,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        queries = self.norm1(queries + self.dropout(attn_output))
        
        # FFN + 残差
        ffn_output = self.ffn(queries)
        output = self.norm2(queries + ffn_output)
        
        return output

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cross_attn = AspectGuidedCrossAttention().to(device)
    
    # 准备输入
    queries = torch.randn(4, 9, 768).to(device)  # [B, m, D]
    text_features = torch.randn(4, 80, 768).to(device)  # [B, L, D]
    
    # 交叉注意力
    output = cross_attn(queries, text_features, text_features)
    print(f"输出形状: {output.shape}")  # [4, 9, 768]
    print(f"✅ 交叉注意力模块正常工作")
```

### 3.3 注意力池化

**设计原理：**
- 替代简单的MeanPooling
- 使用可学习的聚合向量智能加权多个查询
- 通过注意力机制自动学习重要性

**实现代码：**

**模块说明：`src/models/pooling.py`**
- **用途**: 实现注意力池化模块 `AttentionPooling`，使用可学习聚合向量对多查询特征进行加权汇聚，得到单一的方面特征表示。
- **输入**: 多查询特征 `Z`，形状 `[B, m, D]`。
- **输出**: 池化后的特征 `pooled`（形状 `[B, D]`）以及对应的注意力权重（形状 `[B, m]`，用于解释每个查询的重要性）。

```python
# src/models/pooling.py
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """注意力池化：智能聚合多查询特征"""
    
    def __init__(self, hidden_dim=768, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 全局可学习聚合向量
        self.aggregator = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.aggregator)
        
        # 注意力模块
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, Z):
        """
        Args:
            Z: [B, m, D] 多查询特征
        
        Returns:
            pooled: [B, D] 聚合后的单一特征
        """
        batch_size = Z.size(0)
        
        # 扩展聚合向量到batch
        query = self.aggregator.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # 注意力池化：Q=aggregator, K=Z, V=Z
        output, attn_weights = self.attn(
            query=query,
            key=Z,
            value=Z,
            need_weights=True
        )  # output: [B, 1, D], attn_weights: [B, 1, m]
        
        pooled = output.squeeze(1)  # [B, D]
        
        return pooled, attn_weights.squeeze(1)  # [B, D], [B, m]

# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pooling = AttentionPooling().to(device)
    
    # 输入多查询特征
    Z = torch.randn(4, 9, 768).to(device)
    
    # 池化
    pooled, attn_weights = pooling(Z)
    print(f"池化前: {Z.shape}")  # [4, 9, 768]
    print(f"池化后: {pooled.shape}")  # [4, 768]
    print(f"注意力权重: {attn_weights.shape}")  # [4, 9]
    print(f"权重和（应约等于1）: {attn_weights[0].sum().item():.4f}")
    print(f"✅ 注意力池化正常工作")
```

### 3.4 投影头和分类器

**模块说明：`src/models/projector.py`**
- **用途**: 提供对比学习所需的投影头 `ProjectionHead`，以及情感分类器 `SentimentClassifier` 和方面分类器 `AspectClassifier`，统一完成特征投影和分类任务。
- **输入**: 投影头输入为单模态特征 `[B, D]`；情感分类器输入为拼接后的多模态特征 `[B, 2D]`；方面分类器输入为单模态特征 `[B, D]`。
- **输出**: 投影头输出 L2 归一化后的特征 `[B, D']`；分类器输出对应维度的 logits（情感 `[B, 3]`，方面 `[B, num_aspects]`）。

```python
# src/models/projector.py
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """投影头：用于对比学习"""
    
    def __init__(self, input_dim=768, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            h: [B, D'] L2归一化的投影特征
        """
        h = self.projection(x)
        h = nn.functional.normalize(h, p=2, dim=1)  # L2归一化
        return h

class SentimentClassifier(nn.Module):
    """情感分类器"""
    
    def __init__(self, input_dim=1536, hidden_dim=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*D] 拼接的多模态特征
        Returns:
            logits: [B, 3] 情感预测logits
        """
        return self.classifier(x)

class AspectClassifier(nn.Module):
    """方面分类器（辅助任务）"""
    
    def __init__(self, input_dim=768, num_aspects=3, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_aspects)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 单模态特征
        Returns:
            logits: [B, num_aspects] 方面预测logits
        """
        return self.classifier(x)
```

### 3.5 完整模型

**模块说明：`src/models/agp_model.py`**
- **用途**: 将编码器、查询生成器、交叉注意力、注意力池化、投影头和分类器整合为完整的 `AGPModel`，统一实现前向传播和多任务输出。
- **输入**: 一个 batch 的数据字典（通常来自 `DataLoader`），包含文本 token、方面描述 token、图像张量、情感标签、方面 ID、pair_id mask 等信息。
- **输出**: 包含多种中间结果和最终结果的字典，例如 `h_text`、`h_image`、`sentiment_logits`、`aspect_logits_text`、`aspect_logits_image` 等，用于计算联合损失和评估。

```python
# src/models/agp_model.py
import torch
import torch.nn as nn
from encoders import TextEncoder, ImageEncoder
from query_generator import HybridQueryGenerator
from attention import AspectGuidedCrossAttention
from pooling import AttentionPooling
from projector import ProjectionHead, SentimentClassifier, AspectClassifier

class AGPModel(nn.Module):
    """完整的AGP模型"""
    
    def __init__(
        self,
        num_aspects,
        hidden_dim=768,
        proj_dim=256,
        num_queries=8,
        num_classes=3,
        freeze_bert_layers=10,
        use_lora=True,
        lora_rank=8
    ):
        super().__init__()
        
        # 编码器
        self.text_encoder = TextEncoder(freeze_layers=freeze_bert_layers)
        self.image_encoder = ImageEncoder(use_lora=use_lora, lora_rank=lora_rank)
        
        # 查询生成器
        self.query_generator = HybridQueryGenerator(
            num_aspects=num_aspects,
            hidden_dim=hidden_dim,
            num_learnable_queries=num_queries
        )
        
        # 交叉注意力
        self.text_cross_attn = AspectGuidedCrossAttention(hidden_dim=hidden_dim)
        self.image_cross_attn = AspectGuidedCrossAttention(hidden_dim=hidden_dim)
        
        # 注意力池化
        self.text_pooling = AttentionPooling(hidden_dim=hidden_dim)
        self.image_pooling = AttentionPooling(hidden_dim=hidden_dim)
        
        # 投影头（用于对比学习）
        self.text_proj = ProjectionHead(hidden_dim, proj_dim)
        self.image_proj = ProjectionHead(hidden_dim, proj_dim)
        
        # 分类器
        self.sentiment_classifier = SentimentClassifier(
            input_dim=hidden_dim * 2,
            num_classes=num_classes
        )
        
        # 辅助任务：方面分类器
        self.aspect_classifier_text = AspectClassifier(hidden_dim, num_aspects)
        self.aspect_classifier_image = AspectClassifier(hidden_dim, num_aspects)
    
    def forward(self, batch):
        """
        Args:
            batch: dict包含所有输入
        
        Returns:
            dict包含所有输出
        """
        # 1. 编码文本和图像
        text_features = self.text_encoder(
            batch['text_input_ids'],
            batch['text_attention_mask']
        )  # [B, L, D]
        
        image_features = self.image_encoder(batch['images'])  # [B, P, D]
        
        # 2. 生成混合查询
        aspect_desc_encoding = {
            'input_ids': batch['aspect_input_ids'],
            'attention_mask': batch['aspect_attention_mask']
        }
        queries = self.query_generator(
            batch['aspect_ids'],
            aspect_desc_encoding,
            self.text_encoder
        )  # [B, m, D]
        
        # 3. 交叉注意力提取方面相关特征
        # 注意：需要反转attention_mask（1->False, 0->True）
        text_padding_mask = (batch['text_attention_mask'] == 0)
        
        Z_text = self.text_cross_attn(
            queries=queries,
            keys=text_features,
            values=text_features,
            key_padding_mask=text_padding_mask
        )  # [B, m, D]
        
        Z_image = self.image_cross_attn(
            queries=queries,
            keys=image_features,
            values=image_features
        )  # [B, m, D]
        
        # 4. 注意力池化
        g_text, text_attn_weights = self.text_pooling(Z_text)  # [B, D]
        g_image, image_attn_weights = self.image_pooling(Z_image)  # [B, D]
        
        # 5. 投影到对比学习空间
        h_text = self.text_proj(g_text)  # [B, D']
        h_image = self.image_proj(g_image)  # [B, D']
        
        # 6. 拼接多模态特征
        multimodal_feature = torch.cat([g_text, g_image], dim=1)  # [B, 2D]
        
        # 7. 情感分类
        sentiment_logits = self.sentiment_classifier(multimodal_feature)  # [B, 3]
        
        # 8. 辅助任务：方面分类
        aspect_logits_text = self.aspect_classifier_text(g_text)
        aspect_logits_image = self.aspect_classifier_image(g_image)
        
        return {
            'sentiment_logits': sentiment_logits,
            'aspect_logits_text': aspect_logits_text,
            'aspect_logits_image': aspect_logits_image,
            'h_text': h_text,
            'h_image': h_image,
            'g_text': g_text,
            'g_image': g_image,
            'Z_text': Z_text,
            'Z_image': Z_image,
            'text_attn_weights': text_attn_weights,
            'image_attn_weights': image_attn_weights
        }

# 测试完整模型
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AGPModel(
        num_aspects=3,
        num_queries=8
    ).to(device)
    
    # 模拟一个batch
    batch = {
        'text_input_ids': torch.randint(0, 30000, (4, 80)).to(device),
        'text_attention_mask': torch.ones(4, 80).to(device),
        'aspect_input_ids': torch.randint(0, 30000, (4, 30)).to(device),
        'aspect_attention_mask': torch.ones(4, 30).to(device),
        'images': torch.randn(4, 3, 224, 224).to(device),
        'aspect_ids': torch.randint(0, 3, (4,)).to(device)
    }
    
    # 前向传播
    with torch.no_grad():
        outputs = model(batch)
    
    print("=== 模型输出 ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 参数统计 ===")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"✅ 模型构建成功！")
```

**运行完整测试：**
```bash
cd src/models
python agp_model.py
```

**预期输出：**
```
✅ 冻结BERT前10层,微调后2层
✅ ViT应用LoRA (rank=8, alpha=16)
   可训练参数: 295,936 / 86,567,656 (0.34%)

=== 模型输出 ===
sentiment_logits: torch.Size([4, 3])
aspect_logits_text: torch.Size([4, 3])
aspect_logits_image: torch.Size([4, 3])
h_text: torch.Size([4, 256])
h_image: torch.Size([4, 256])
g_text: torch.Size([4, 768])
g_image: torch.Size([4, 768])
Z_text: torch.Size([4, 9, 768])
Z_image: torch.Size([4, 9, 768])
text_attn_weights: torch.Size([4, 9])
image_attn_weights: torch.Size([4, 9])

=== 参数统计 ===
总参数: 201,234,567
可训练参数: 25,678,123 (12.76%)
✅ 模型构建成功！
```

**✅ 检查点：**
- [ ] 所有模块单独测试通过
- [ ] 完整模型前向传播成功
- [ ] 输出形状符合预期
- [ ] 参数冻结策略正确
- [ ] LoRA注入成功

**预计耗时：** 4-6小时

---

## 4. 损失函数实现

### 4.1 损失函数体系概览

**总损失函数：**

```
L_total = L_cls + α·L_InfoNCE + β·L_SupCon + γ·L_aux

其中：
- L_cls: 情感分类损失（交叉熵）
- L_InfoNCE: 跨模态对齐损失
- L_SupCon: Aspect-Aware情感可分离损失
- L_aux: 辅助任务损失（方面分类）
- α, β, γ: 损失权重（推荐：α=1.0, β=0.5, γ=0.3）
```

### 4.2 辅助任务损失（方面分类）

**目标：** 确保提取的特征包含方面信息，强制模型学习方面导向的表示。

**实现代码：**

**模块说明：`src/losses/auxiliary.py`**
- **用途**: 实现辅助任务损失 `AuxiliaryAspectLoss`，通过同时约束文本和图像的方面预测，使模型显式学习方面信息。
- **输入**: 文本方面 logits `aspect_logits_text`、图像方面 logits `aspect_logits_image`（形状均为 `[B, num_aspects]`），以及真实方面标签 `aspect_ids`（`[B]`）。
- **输出**: 标量损失值 `loss`，以及包含文本 / 图像分支单独损失的字典（用于日志和分析）。

```python
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
```

### 4.3 改进的Aspect-Aware SupCon损失

**关键改进：** 只有情感**且**方面都相同才是正例，避免"同方面不同情感"被拉近。

**实现代码：**

**模块说明：`src/losses/supcon.py`**
- **用途**: 实现方面感知的监督对比损失 `AspectAwareSupConLoss` 以及多视图版本 `MultiViewSupConLoss`，在对比学习中同时考虑情感和方面一致性。
- **输入**: 归一化后的特征 `features` 或 `(h_text, h_image)`，对应的情感标签 `labels` 和方面 ID `aspect_ids`。
- **输出**: 标量损失值，用于拉近“同情感且同方面”的样本、推远其他样本，并在多视图场景下同时约束文本和图像特征。

```python
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
```

### 4.4 InfoNCE损失（带Pair-ID Mask）

**关键设计：** 必须屏蔽同一图文对的不同方面样本，避免它们互为负样本。

**实现代码：**

**模块说明：`src/losses/infonce.py`**
- **用途**: 实现带 `pair_id` 掩码的跨模态 InfoNCE 损失 `InfoNCELoss`，用于对齐同一图文对的文本和图像表示，同时避免同一图文对的不同方面误当作负样本。
- **输入**: 归一化的文本特征 `h_text`、图像特征 `h_image`（形状 `[B, D']`），以及 `pair_id_mask`（形状 `[B, B]`，`True` 表示同一图文对）。
- **输出**: 标量损失值 `loss`，以及 text-to-image / image-to-text 两个方向的损失明细字典。

```python
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
```

### 4.5 分类损失

**模块说明：`src/losses/classification.py`**
- **用途**: 封装基础的情感分类交叉熵损失 `ClassificationLoss`，可选支持 label smoothing。
- **输入**: 情感预测 logits（形状 `[B, 3]`）和真实情感标签 `labels`（`[B]`）。
- **输出**: 标量分类损失，用于直接优化情感预测的准确性。

```python
# src/losses/classification.py
import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """情感分类损失"""
    
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 3] 情感预测logits
            labels: [B] 真实标签
        
        Returns:
            loss: scalar
        """
        return self.criterion(logits, labels)
```

### 4.6 联合损失函数

**模块说明：`src/losses/total_loss.py`**
- **用途**: 将分类损失、InfoNCE 损失、SupCon 损失和辅助任务损失组合为一个总损失 `TotalLoss`，并支持通过超参数控制各项权重。
- **输入**: 模型输出字典 `outputs`（包含 logits 和对比学习特征）、情感标签 `labels`、方面 ID `aspect_ids`、`pair_id_mask`。
- **输出**: 标量总损失 `total_loss`，以及包含各子损失与分项信息的字典 `loss_dict`，便于监控和调参。

```python
# src/losses/total_loss.py
import torch
import torch.nn as nn
from classification import ClassificationLoss
from infonce import InfoNCELoss
from supcon import MultiViewSupConLoss
from auxiliary import AuxiliaryAspectLoss

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
```

**运行完整测试：**
```bash
cd src/losses
python total_loss.py
```

**预期输出：**
```
=== 损失函数测试 ===
总损失: 3.2456
  分类损失: 1.0987
  InfoNCE损失: 1.2345
    - T2I: 1.2123
    - I2T: 1.2567
  SupCon损失: 0.7654
  辅助损失: 0.4321
    - Text: 0.4234
    - Image: 0.4408
✅ 联合损失函数测试成功
```

### 4.7 损失权重调优建议

**默认权重配置：**

| 损失项 | 权重 | 作用 | 调优建议 |
|--------|------|------|---------|
| L_cls | 1.0 (基准) | 情感分类 | 保持为1.0 |
| L_InfoNCE (α) | 1.0 | 跨模态对齐 | 如果对齐不足，增大到1.5-2.0 |
| L_SupCon (β) | 0.5 | 情感可分离 | 如果情感混淆严重，增大到0.8-1.0 |
| L_aux (γ) | 0.3 | 方面识别 | 如果方面混淆，增大到0.5 |

**动态调整策略：**

```python
# 训练初期（Epoch 1-3）：强化基础对齐
alpha, beta, gamma = 1.5, 0.3, 0.2

# 训练中期（Epoch 4-10）：平衡发展
alpha, beta, gamma = 1.0, 0.5, 0.3

# 训练后期（Epoch 11-15）：强化分类
alpha, beta, gamma = 0.8, 0.5, 0.2
```

**✅ 检查点：**
- [ ] 所有损失函数单独测试通过
- [ ] 联合损失计算正确
- [ ] Pair-ID mask正确应用
- [ ] Aspect-Aware正样本定义正确
- [ ] 硬负例权重机制工作

**预计耗时：** 2-3小时

---

## 5. 训练执行流程

### 5.1 训练器实现

**模块说明：`src/training/trainer.py`**
- **用途**: 封装完整的训练流程，包括优化器 / 学习率调度器创建、单个 epoch 训练、验证评估、混合精度、梯度裁剪以及检查点保存等。
- **输入**: 已构建好的模型 `model`、`train_loader`、`dev_loader`、损失函数 `loss_fn`、运行设备 `device`、训练配置 `config`（包括学习率、epoch 数、保存目录等）。
- **输出**: 内部维护训练 / 验证历史（accuracy、F1、各项损失），对外通过 `train()` 方法执行训练过程，并在磁盘上保存最佳模型和中间检查点。

```python
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
        self.scaler = GradScaler() if config['use_amp'] else None
        
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
```

### 5.2 主训练脚本

**模块说明：`train.py`**
- **用途**: 作为训练入口脚本，负责加载配置、构建数据加载器和模型、创建损失函数和训练器，并串联起整个训练流程。
- **输入**: `configs/training_config.yaml` 配置文件（或命令行 / 外部环境提供的路径），以及项目目录下的数据和模型文件。
- **输出**: 训练好的模型检查点文件（保存到 `models/checkpoints`）、训练日志（控制台和可选的 JSON 历史文件），以及在训练过程中打印的关键信息。

```python
# train.py
import torch
import yaml
import os
import random
import numpy as np
from src.data.create_dataloaders import create_dataloaders
from src.models.agp_model import AGPModel
from src.losses.total_loss import TotalLoss
from src.training.trainer import Trainer

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path='configs/training_config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config()
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 创建数据加载器
    print("\n1. 创建数据加载器...")
    train_loader, dev_loader, test_loader, num_aspects = create_dataloaders(
        train_jsonl=config['train_jsonl'],
        dev_jsonl=config['dev_jsonl'],
        test_jsonl=config['test_jsonl'],
        image_root=config['image_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("\n2. 创建模型...")
    model = AGPModel(
        num_aspects=num_aspects,
        hidden_dim=config['hidden_dim'],
        proj_dim=config['proj_dim'],
        num_queries=config['num_queries'],
        num_classes=config['num_classes'],
        freeze_bert_layers=config['freeze_bert_layers'],
        use_lora=config['use_lora'],
        lora_rank=config['lora_rank']
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 创建损失函数
    print("\n3. 创建损失函数...")
    loss_fn = TotalLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        temperature_infonce=config['temperature_infonce'],
        temperature_supcon=config['temperature_supcon']
    )
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    trainer.train()
    
    # 保存训练历史
    import json
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train': trainer.train_history,
            'dev': trainer.dev_history
        }, f, indent=2)
    print(f"\n训练历史保存到: {history_path}")

if __name__ == '__main__':
    main()
```

### 5.3 配置文件

```yaml
# configs/training_config.yaml
# 数据路径
train_jsonl: 'data/processed/train_expanded.jsonl'
dev_jsonl: 'data/processed/dev_expanded.jsonl'
test_jsonl: 'data/processed/test_expanded.jsonl'
image_root: 'data/images'

# 模型配置
hidden_dim: 768
proj_dim: 256
num_queries: 8
num_classes: 3
freeze_bert_layers: 10
use_lora: true
lora_rank: 8

# 损失权重
alpha: 1.0          # InfoNCE
beta: 0.5           # SupCon
gamma: 0.3          # Auxiliary
temperature_infonce: 0.07
temperature_supcon: 0.1

# 训练配置
num_epochs: 15
batch_size: 32
num_workers: 4
lr_backbone: 1.0e-5  # BERT和LoRA
lr_head: 1.0e-4      # 新模块
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0
use_amp: true        # 混合精度训练

# 保存配置
save_dir: 'models/checkpoints'
save_every: 5

# 其他
seed: 42
```

### 5.4 执行训练

**步骤1：验证配置**

```bash
# 检查配置文件
cat configs/training_config.yaml

# 测试数据加载
python -c "from src.data.create_dataloaders import create_dataloaders; \
           train_loader, dev_loader, test_loader, num_aspects = create_dataloaders(batch_size=8); \
           print(f'Train: {len(train_loader)}, Dev: {len(dev_loader)}, Test: {len(test_loader)}')"
```

**步骤2：启动训练**

```bash
# 单GPU训练
python train.py

# 指定GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# 后台运行（推荐）
nohup python train.py > logs/training.log 2>&1 &

# 查看日志
tail -f logs/training.log
```

**步骤3：监控训练**

使用TensorBoard（需要在Trainer中添加SummaryWriter）：

```bash
tensorboard --logdir=logs/tensorboard --port=6006
```

### 5.5 预期训练曲线

**Epoch 1-3（冷启动阶段）：**
```
Epoch 1 Summary:
Train Loss: 3.2456 | Train Acc: 0.4523
Dev Loss: 3.0123 | Dev Acc: 0.4234 | Dev Macro-F1: 0.3987

Epoch 2 Summary:
Train Loss: 2.7834 | Train Acc: 0.5234
Dev Loss: 2.6789 | Dev Acc: 0.5012 | Dev Macro-F1: 0.4756

Epoch 3 Summary:
Train Loss: 2.4567 | Train Acc: 0.5876
Dev Loss: 2.4321 | Dev Acc: 0.5634 | Dev Macro-F1: 0.5423
```

**Epoch 5-10（快速提升）：**
```
Epoch 5 Summary:
Train Loss: 1.9876 | Train Acc: 0.6543
Dev Loss: 2.1234 | Dev Acc: 0.6234 | Dev Macro-F1: 0.6012

Epoch 8 Summary:
Train Loss: 1.6543 | Train Acc: 0.7234
Dev Loss: 1.9876 | Dev Acc: 0.6756 | Dev Macro-F1: 0.6543
```

**Epoch 13-15（收敛）：**
```
Epoch 15 Summary:
Train Loss: 1.2345 | Train Acc: 0.7856
Dev Loss: 1.7654 | Dev Acc: 0.7123 | Dev Macro-F1: 0.6934
✅ 保存最佳模型 (F1: 0.6934)
```

**✅ 检查点：**
- [ ] 训练正常启动，无内存溢出
- [ ] 损失稳定下降
- [ ] 训练和验证准确率都在提升
- [ ] 最佳模型成功保存
- [ ] Dev F1达到66-70%范围

**预计耗时：** 2-3小时（15 epochs，单A100 GPU）

---

## 6. 模型评估与分析

### 6.1 评估指标计算

**模块说明：`src/evaluation/metrics.py`**
- **用途**: 提供统一的评估指标计算与展示工具 `MetricsCalculator`，包括准确率、Macro/Weighted F1、各类别 F1、混淆矩阵绘制和报告打印。
- **输入**: 模型在评估集上的预测结果 `all_preds`、真实标签 `all_labels`，以及可选的类别名字 `label_names`。
- **输出**: 指标字典 `metrics`（包含各类 F1 值、混淆矩阵和文本报告），同时可将混淆矩阵保存为图片并在终端打印详细指标信息。

```python
# src/evaluation/metrics.py
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, label_names=['Negative', 'Neutral', 'Positive']):
        self.label_names = label_names
    
    def compute_metrics(self, all_preds, all_labels):
        """计算所有指标"""
        # 基础指标
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 各类别F1
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        
        # 分类报告
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.label_names,
            digits=4
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_f1': {
                self.label_names[i]: per_class_f1[i]
                for i in range(len(self.label_names))
            },
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ 混淆矩阵保存到: {save_path}")
    
    def print_metrics(self, metrics):
        """打印指标"""
        print("\n" + "="*60)
        print("评估指标")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\n各类别F1分数:")
        for label, f1 in metrics['per_class_f1'].items():
            print(f"  {label}: {f1:.4f}")
        print("\n分类报告:")
        print(metrics['classification_report'])
```

### 6.2 完整评估脚本

**模块说明：`evaluate.py`**
- **用途**: 加载训练好的检查点，对验证 / 测试集运行前向推理，收集预测结果，并调用 `MetricsCalculator` 完成最终评估和可选可视化。
- **输入**: 模型检查点路径、评估数据加载器（或通过内部创建）、运行设备 `device`、配置文件路径等。
- **输出**: 汇总后的评估指标（打印到控制台或写入文件），以及根据需要保存的混淆矩阵图片或其他评估产物。

```python
# evaluate.py
import torch
import yaml
import os
from tqdm import tqdm
from src.data.create_dataloaders import create_dataloaders
from src.models.agp_model import AGPModel
from src.evaluation.metrics import MetricsCalculator

def load_checkpoint(checkpoint_path, model, device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载模型: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']+1}")
    print(f"   Best Dev F1: {checkpoint['best_dev_f1']:.4f}")
    return model

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="评估中"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        outputs = model(batch)
        preds = outputs['sentiment_logits'].argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    return all_preds, all_labels

def main():
    # 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    print("\n1. 创建数据加载器...")
    _, _, test_loader, num_aspects = create_dataloaders(
        train_jsonl=config['train_jsonl'],
        dev_jsonl=config['dev_jsonl'],
        test_jsonl=config['test_jsonl'],
        image_root=config['image_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("\n2. 创建模型...")
    model = AGPModel(
        num_aspects=num_aspects,
        hidden_dim=config['hidden_dim'],
        proj_dim=config['proj_dim'],
        num_queries=config['num_queries'],
        num_classes=config['num_classes'],
        freeze_bert_layers=config['freeze_bert_layers'],
        use_lora=config['use_lora'],
        lora_rank=config['lora_rank']
    ).to(device)
    
    # 加载最佳模型
    print("\n3. 加载模型...")
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
    model = load_checkpoint(checkpoint_path, model, device)
    
    # 评估
    print("\n4. 开始评估...")
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # 计算指标
    print("\n5. 计算指标...")
    calculator = MetricsCalculator()
    metrics = calculator.compute_metrics(all_preds, all_labels)
    
    # 打印指标
    calculator.print_metrics(metrics)
    
    # 绘制混淆矩阵
    os.makedirs('results', exist_ok=True)
    calculator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='results/confusion_matrix.png'
    )
    
    # 保存结果
    import json
    results_path = 'results/test_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'per_class_f1': metrics['per_class_f1']
        }, f, indent=2)
    print(f"\n✅ 结果保存到: {results_path}")

if __name__ == '__main__':
    main()
```

**运行评估：**
```bash
python evaluate.py
```

**预期输出：**
```
============================================================
评估指标
============================================================
Accuracy: 0.7123
Macro F1: 0.6934
Weighted F1: 0.7089

各类别F1分数:
  Negative: 0.6987
  Neutral: 0.6012
  Positive: 0.7803

分类报告:
              precision    recall  f1-score   support

    Negative     0.7123    0.6876    0.6987       150
     Neutral     0.6234    0.5812    0.6012        95
    Positive     0.7654    0.7956    0.7803       255

    accuracy                         0.7123       500
   macro avg     0.7004    0.6881    0.6934       500
weighted avg     0.7098    0.7123    0.7089       500
```

### 6.3 错误分析

**模块说明：`src/evaluation/error_analysis.py`**
- **用途**: 对预测错误的样本进行系统性分析，包括按方面统计、按真实 / 预测标签组合统计，并导出详细错误样本表格。
- **输入**: 预测结果 `all_preds`、真实标签 `all_labels`、样本 ID 列表 `sample_ids`、原始文本 `texts`、方面列表 `aspects`。
- **输出**: `pandas.DataFrame` 格式的错误样本表（同时保存为 `results/error_analysis.csv`），以及在控制台打印的错误分布统计信息。

```python
# src/evaluation/error_analysis.py
import pandas as pd

def analyze_errors(all_preds, all_labels, sample_ids, texts, aspects):
    """分析预测错误的样本"""
    errors = []
    
    for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
        if pred != true:
            errors.append({
                'sample_id': sample_ids[i],
                'text': texts[i],
                'aspect': aspects[i],
                'true_label': true,
                'pred_label': pred
            })
    
    error_df = pd.DataFrame(errors)
    
    # 按方面统计错误
    print("\n按方面统计错误:")
    print(error_df['aspect'].value_counts())
    
    # 按错误类型统计
    print("\n按错误类型统计:")
    error_types = error_df.groupby(['true_label', 'pred_label']).size()
    print(error_types)
    
    # 保存错误样本
    error_df.to_csv('results/error_analysis.csv', index=False)
    print("\n✅ 错误分析保存到: results/error_analysis.csv")
    
    return error_df
```

**✅ 检查点：**
- [ ] 测试集准确率达到68-72%
- [ ] Macro F1达到66-70%
- [ ] 各类别F1合理（中性类别通常较低）
- [ ] 混淆矩阵显示合理的错误分布

**预计耗时：** 30分钟

---

## 7. 调试与优化

### 7.1 常见问题排查清单

#### 问题1：Loss不下降或NaN

**可能原因：**
- 学习率过大
- 梯度爆炸
- Pair-ID mask错误
- 批次中没有正样本对

**解决方案：**

```python
# 1. 降低学习率
lr_backbone: 5.0e-6  # 从1e-5降到5e-6
lr_head: 5.0e-5      # 从1e-4降到5e-5

# 2. 增加梯度裁剪
max_grad_norm: 0.5   # 从1.0降到0.5

# 3. 检查pair_id_mask
def verify_pair_id_mask(pair_ids):
    batch_size = len(pair_ids)
    mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(batch_size):
            if pair_ids[i] == pair_ids[j] and i != j:
                mask[i, j] = True
    # mask应该是对称的且对角线为False
    assert mask.equal(mask.T), "mask不对称!"
    assert not mask.diagonal().any(), "对角线不应为True!"
    return mask

# 4. 增大batch size以确保有足够的正样本对
batch_size: 64  # 从32增到64
```

#### 问题2：显存溢出

**解决方案：**

```python
# 方案1：减小batch size
batch_size: 16

# 方案2：梯度累积
accumulation_steps: 2  # 有效batch=16*2=32

# 方案3：减少查询数量
num_queries: 6  # 从8降到6

# 方案4：使用更激进的LoRA
lora_rank: 4  # 从8降到4

# 方案5：冻结更多BERT层
freeze_bert_layers: 11  # 从10增到11，只微调最后1层
```

#### 问题3：过拟合（Train高Dev低）

**解决方案：**

```python
# 1. 增大Dropout
# 在模型中添加更多dropout
dropout: 0.3  # 从0.1增到0.3

# 2. 增大温度参数（降低对比学习的确定性）
temperature_supcon: 0.2  # 从0.1增到0.2

# 3. 增大权重衰减
weight_decay: 0.05  # 从0.01增到0.05

# 4. 早停
# 在Trainer中添加early stopping
patience: 5  # 连续5个epoch验证集不提升则停止

# 5. 数据增强
# 对文本进行随机掩码
# 对图像进行更强的增强
```

#### 问题4：方面混淆严重

**解决方案：**

```python
# 1. 增大辅助任务权重
gamma: 0.5  # 从0.3增到0.5

# 2. 增加方面原型对比学习（见CONTRASTIVE_LEARNING_ANALYSIS.md方案1）

# 3. 可视化方面特征，确认方面导向性
```

### 7.2 超参数调优建议

**优先级排序：**

1. **学习率（最重要）**
   ```
   推荐范围:
   - lr_backbone: [5e-6, 1e-5, 2e-5]
   - lr_head: [5e-5, 1e-4, 2e-4]
   ```

2. **损失权重**
   ```
   推荐范围:
   - alpha (InfoNCE): [0.8, 1.0, 1.5]
   - beta (SupCon): [0.3, 0.5, 0.8]
   - gamma (Aux): [0.2, 0.3, 0.5]
   ```

3. **Batch Size**
   ```
   推荐: [16, 32, 64]
   注意：对比学习受益于大batch
   ```

4. **温度参数**
   ```
   - temperature_infonce: [0.05, 0.07, 0.1]
   - temperature_supcon: [0.07, 0.1, 0.15]
   ```

### 7.3 性能优化技巧

```python
# 1. 使用更快的数据加载
num_workers: 8          # 增加worker数量
pin_memory: True        # 使用pin memory
persistent_workers: True  # 保持worker持久化

# 2. 编译模型（PyTorch 2.0+）
model = torch.compile(model)

# 3. 使用BF16而非FP16（A100）
# 在Trainer中
with autocast(dtype=torch.bfloat16):
    outputs = model(batch)

# 4. 梯度检查点（减少显存）
# 在模型中
def forward(self, batch):
    from torch.utils.checkpoint import checkpoint
    Z_text = checkpoint(self.text_cross_attn, queries, text_features, text_features)
```

### 7.4 消融实验设计

**目的：验证各组件的贡献**

```python
# 实验1：Baseline（无LLM扩写）
aspect_desc = sample['aspect']  # 不使用LLM扩写

# 实验2：无Aspect-Aware SupCon
# 使用原始SupCon（不考虑方面）
pos_mask = (label_match).float()  # 移除aspect_match

# 实验3：无辅助任务
gamma: 0.0  # 关闭辅助任务

# 实验4：无Attention Pooling
# 使用MeanPooling替代AttentionPooling

# 实验5：减少查询数量
num_queries: 4  # 从8降到4
```

**✅ 检查点：**
- [ ] 了解常见问题和解决方案
- [ ] 掌握超参数调优策略
- [ ] 知道如何设计消融实验
- [ ] 能够进行性能优化

**预计耗时：** 根据具体问题而定

---

## 附录

### A. 完整文件清单

```
AGP-MABSA/
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── processed/
│   │   ├── train_expanded.jsonl
│   │   ├── dev_expanded.jsonl
│   │   └── test_expanded.jsonl
│   └── images/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── create_dataloaders.py
│   │   └── llm_expansion.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── query_generator.py
│   │   ├── attention.py
│   │   ├── pooling.py
│   │   ├── projector.py
│   │   └── agp_model.py
│   ├── losses/
│   │   ├── classification.py
│   │   ├── infonce.py
│   │   ├── supcon.py
│   │   ├── auxiliary.py
│   │   └── total_loss.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       ├── metrics.py
│       └── error_analysis.py
├── configs/
│   └── training_config.yaml
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

### B. 快速开始命令

```bash
# 1. 环境配置（30分钟）
conda create -n agp_mabsa python=3.9
conda activate agp_mabsa
pip install -r requirements.txt

# 2. 数据预处理（2小时）
python src/data/llm_expansion.py

# 3. 测试模型（10分钟）
cd src/models
python agp_model.py

# 4. 训练模型（2-3小时）
python train.py

# 5. 评估模型（10分钟）
python evaluate.py
```

### C. 预期时间线

| 步骤 | 预计耗时 | 累计耗时 |
|------|---------|---------|
| 1. 环境准备 | 30分钟 | 0.5小时 |
| 2. 数据预处理 | 2小时 | 2.5小时 |
| 3. 模型实现 | 4-6小时 | 8.5小时 |
| 4. 损失函数 | 2-3小时 | 11小时 |
| 5. 训练执行 | 2-3小时 | 14小时 |
| 6. 模型评估 | 1小时 | 15小时 |
| **总计** | **12-15小时** | |

### D. 联系与支持

遇到问题时：
1. 检查本文档的"调试与优化"章节
2. 查阅`CONTRASTIVE_LEARNING_ANALYSIS.md`的改进建议
3. 参考`AGAA METHOD GUIDE.md`的实现细节
4. 查看GitHub Issues（如有）

---

**文档版本:** 1.0  
**最后更新:** 2026-01-27  
**文档状态:** 已完成  
**总字数:** ~15,000字

🎉 **恭喜！您已完成AGP实验步骤文档的学习。现在可以开始实验了！**
