# 数据目录

此目录用于存放实验所需的数据文件。

## 目录结构

```
data/
├── raw/                    # 原始数据文件
│   ├── train.jsonl        # 训练集（原始）
│   ├── dev.jsonl          # 验证集（原始）
│   └── test.jsonl         # 测试集（原始）
├── processed/              # 处理后的数据文件
│   ├── train_expanded.jsonl  # 训练集（LLM扩写后）
│   ├── dev_expanded.jsonl    # 验证集（LLM扩写后）
│   └── test_expanded.jsonl   # 测试集（LLM扩写后）
└── images/                # 图像文件
    └── twitter2015_images/ # Twitter图像数据集
```

## 数据格式

### JSONL格式示例
```json
{
  "sample_id": "twitter15_train_001",
  "text": "The steak was cold but the ambience was nice",
  "aspect": "food",
  "image_paths": ["twitter2015_images/001.jpg"],
  "label": 0,
  "pair_id": "twitter15_train_001",
  "aspect_desc": "taste presentation portion size and freshness of dishes"
}
```

### 字段说明
- `sample_id`: 唯一样本标识符
- `text`: 评论文本
- `aspect`: 目标方面（food/service/ambience等）
- `image_paths`: 图像路径列表
- `label`: 情感标签（0=负面，1=中性，2=正面）
- `pair_id`: 图文对标识（同一图文对的不同方面共享此ID）
- `aspect_desc`: LLM扩写的方面描述（processed目录中的文件才有）

## 数据准备步骤

1. **放置原始数据**：将 `train.jsonl`, `dev.jsonl`, `test.jsonl` 放入 `data/raw/`
2. **放置图像文件**：将图像文件放入 `data/images/twitter2015_images/`
3. **运行LLM扩写**：执行 `python src/data/llm_expansion.py` 生成扩写后的数据

## 注意事项

- 此目录中的所有数据文件都会被 `.gitignore` 忽略
- 数据文件通常较大，不应提交到Git仓库
- 请确保数据路径与配置文件中的路径一致
- 图像路径应相对于 `data/images/` 目录
