# 预训练模型目录

此目录用于存放从HuggingFace下载的预训练模型。

## 模型说明

### BERT文本编码器
- **模型名称**: `bert-base-uncased`
- **来源**: HuggingFace Transformers
- **用途**: 文本特征编码
- **下载方式**: 首次运行时会自动下载

### ViT图像编码器
- **模型名称**: `google/vit-base-patch16-224`
- **来源**: HuggingFace Transformers
- **用途**: 图像特征编码
- **下载方式**: 首次运行时会自动下载

## 使用方式

模型会在首次使用时自动从HuggingFace下载并缓存到此目录。

## 注意事项

- 预训练模型文件较大（BERT约440MB，ViT约344MB）
- 下载需要网络连接
- 模型会自动缓存，无需手动下载
- 此目录内容会被 `.gitignore` 忽略，不会提交到Git仓库
