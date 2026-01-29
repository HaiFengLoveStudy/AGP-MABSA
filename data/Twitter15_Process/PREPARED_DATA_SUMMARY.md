# Twitter15 数据准备 - 完成总结

## ✅ 已完成的工作

### 1. 创建的文件

#### 数据处理脚本
- **`prepare_data.py`**: 主数据准备脚本
  - 读取Twitter15的TSV格式文件
  - 转换为AGP-MABSA所需的JSONL格式
  - 自动创建目录结构
  - 为图像文件创建符号链接

#### 验证和测试脚本
- **`verify_data.py`**: 数据验证脚本
  - 检查JSONL文件完整性
  - 验证所有必需字段
  - 统计标签分布
  - 检查图像文件存在性

- **`test_load_data.py`**: 数据加载测试脚本
  - 演示如何加载JSONL数据
  - 展示数据样本
  - 测试图像加载（可选PIL依赖）
  - 显示详细统计信息

#### 文档
- **`使用说明.md`**: 详细的使用说明文档
  - 数据格式说明
  - 目录结构说明
  - 使用方法
  - 常见问题解答

- **`SUMMARY.md`**: 本文件，项目总结

### 2. 生成的数据

#### 目录结构
```
data/
├── raw/                          # 转换后的JSONL数据
│   ├── train.jsonl              # 3,179 样本
│   ├── dev.jsonl                # 1,122 样本
│   └── test.jsonl               # 1,037 样本
├── processed/                    # 预留给LLM扩写数据
├── images/
│   └── twitter2015_images/      # 8,288 张图像（符号链接）
└── Twitter15/
    ├── twitter2015/             # 原始TSV/TXT文件
    ├── twitter2015_images/      # 原始图像文件
    └── *.py, *.md               # 脚本和文档
```

#### 数据统计
- **总样本数**: 5,338
- **训练集**: 3,179 (59.6%)
- **验证集**: 1,122 (21.0%)
- **测试集**: 1,037 (19.4%)
- **图像总数**: 8,288 张JPG图像
- **唯一图文对**: 3,502 个

#### 标签分布
- **负面 (0)**: 630 样本 (11.8%)
- **中性 (1)**: 3,160 样本 (59.2%)
- **正面 (2)**: 1,548 样本 (29.0%)

### 3. 数据格式

生成的JSONL格式符合 `data/README.md` 中的规范：

```json
{
  "sample_id": "twitter15_train_000001",
  "text": "RT @ ltsChuckBass : Chuck Bass is everything # MCM",
  "aspect": "Chuck Bass",
  "image_paths": ["twitter2015_images/1860693.jpg"],
  "label": 2,
  "pair_id": "twitter15_train_1860693"
}
```

## 🚀 快速开始

### 方法一：直接使用已生成的数据

数据已经准备好，可以直接使用：

```python
import json

# 加载训练数据
with open('data/raw/train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]

# 访问样本
sample = train_data[0]
print(sample['text'])
print(sample['aspect'])
print(sample['label'])
```

### 方法二：重新生成数据

如果需要重新生成数据：

```bash
cd data/Twitter15
python prepare_data.py
```

### 方法三：验证数据

验证数据完整性：

```bash
cd data/Twitter15
python verify_data.py
```

### 方法四：测试数据加载

运行完整的测试：

```bash
cd data/Twitter15
python test_load_data.py
```

## 📊 数据特点

### 1. 多目标实体
- 同一条推文可能包含多个目标实体
- 每个目标实体都是独立的样本
- 平均每张图像对应 1.5 个样本

**示例**：
```
文本: "RT @ ltsChuckBass : Chuck Bass is everything # MCM"

样本1: aspect="Chuck Bass", label=2 (正面)
样本2: aspect="# MCM", label=1 (中性)
```

### 2. 多模态数据
- 每个样本包含文本和图像
- 图像路径相对于 `data/images/`
- 支持通过符号链接访问原始图像

### 3. 情感标签
- 0: 负面情感
- 1: 中性情感
- 2: 正面情感

### 4. 唯一标识
- `sample_id`: 每个样本的唯一ID
- `pair_id`: 同一图文对的样本共享相同的pair_id

## 🔧 下一步工作

### 1. 可选：LLM扩写
如果需要 `aspect_desc` 字段，可以运行LLM扩写脚本：

```bash
python src/data/llm_expansion.py
```

这将生成包含方面描述的扩写数据到 `data/processed/` 目录。

### 2. 训练模型
使用准备好的数据训练模型：

```bash
python train.py --data_dir data/raw
```

### 3. 数据探索
使用 `test_load_data.py` 脚本探索数据特征和分布。

## 📝 注意事项

1. **图像路径**: 使用符号链接避免重复存储图像文件
2. **编码格式**: 所有文本文件使用UTF-8编码
3. **Git忽略**: 数据文件已被 `.gitignore` 忽略，不会提交到仓库
4. **图像格式**: 所有图像都是JPG格式

## 🐛 常见问题

### Q1: 找不到图像文件？
**A**: 检查符号链接是否正确：
```bash
ls -la data/images/twitter2015_images
```

### Q2: 如何修改数据格式？
**A**: 编辑 `prepare_data.py` 中的 `convert_to_jsonl_format()` 函数

### Q3: 如何添加新字段？
**A**: 在 `convert_to_jsonl_format()` 函数的 `jsonl_record` 字典中添加新字段

### Q4: PIL/Pillow未安装？
**A**: 安装图像处理库：
```bash
pip install Pillow
```

## 📚 相关文档

- **原始数据说明**: `README.md`
- **项目数据说明**: `../README.md`
- **详细使用说明**: `使用说明.md`

## ✨ 总结

所有数据已经成功准备完成并通过验证：

✅ 5,338 个样本已转换为JSONL格式  
✅ 3 个数据集文件 (train/dev/test) 已生成  
✅ 8,288 张图像文件可正常访问  
✅ 所有字段格式符合规范  
✅ 数据完整性验证通过  

**可以开始使用数据进行模型训练了！** 🎉
