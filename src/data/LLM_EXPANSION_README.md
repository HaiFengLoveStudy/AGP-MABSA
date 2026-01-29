# LLM方面词扩写工具

## 安装依赖

```bash
pip install requests tqdm
```

## 配置（需要修改）

编辑 `llm_expansion.py` 第 16-18 行，配置API信息：

```python
# API配置（必须修改）
API_BASE_URL = "https://api.openai.com/v1"  # API地址
API_KEY = "sk-your-key-here"                 # API密钥
MODEL_NAME = "gpt-4o"                        # 模型名称
```

可选参数（如需调整，修改对应行）：

```python
# 请求参数
TEMPERATURE = 0.3        # 生成温度（0-1）
MAX_TOKENS = 30          # 最大token数
TIMEOUT = 30             # 超时时间（秒）

# 重试配置
MAX_RETRIES = 3          # 最大重试次数
RETRY_DELAY = 1.0        # 重试延迟（秒）

# 并发配置
NUM_THREADS = 5          # 并发线程数（免费API建议1-3）
RATE_LIMIT_DELAY = 0.1   # 请求间隔（秒）

# 输入输出路径（如需修改，编辑第50-59行）
INPUT_FILES = {
    'train': 'data/raw/train.jsonl',
    'dev': 'data/raw/dev.jsonl',
    'test': 'data/raw/test.jsonl'
}
OUTPUT_FILES = {
    'train': 'data/processed/train_expanded.jsonl',
    'dev': 'data/processed/dev_expanded.jsonl',
    'test': 'data/processed/test_expanded.jsonl'
}
```

## 运行

```bash
# 测试配置（可选）
python src/data/test_llm_api.py

# 运行扩写
python src/data/llm_expansion.py
```

## 输入输出

### 输入文件

默认从以下位置读取：
- `data/raw/train.jsonl`
- `data/raw/dev.jsonl`
- `data/raw/test.jsonl`

### 输出文件

自动保存到：
- `data/processed/train_expanded.jsonl`
- `data/processed/dev_expanded.jsonl`
- `data/processed/test_expanded.jsonl`

### 输出格式

输出的JSONL文件会在原有字段基础上添加 `aspect_desc` 字段：

**输入：**
```json
{
  "sample_id": "twitter15_train_000001",
  "text": "The food was amazing",
  "aspect": "food",
  "image_paths": ["twitter2015_images/001.jpg"],
  "label": 2,
  "pair_id": "twitter15_train_001"
}
```

**输出：**
```json
{
  "sample_id": "twitter15_train_000001",
  "text": "The food was amazing",
  "aspect": "food",
  "aspect_desc": "taste presentation portion size and freshness of dishes",
  "image_paths": ["twitter2015_images/001.jpg"],
  "label": 2,
  "pair_id": "twitter15_train_001"
}
```
