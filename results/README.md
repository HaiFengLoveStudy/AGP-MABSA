# 实验结果目录

此目录用于保存模型评估和实验分析的结果文件。

## 文件说明

### 评估结果
- `test_results.json` - 测试集评估指标的JSON格式结果
- `confusion_matrix.png` - 混淆矩阵可视化图
- `error_analysis.csv` - 错误样本分析表格

### 结果内容

`test_results.json` 包含：
- `accuracy` - 整体准确率
- `macro_f1` - 宏平均F1分数
- `weighted_f1` - 加权F1分数
- `per_class_f1` - 各类别的F1分数

## 使用方式

### 查看评估结果
```python
import json
with open('results/test_results.json') as f:
    results = json.load(f)
    print(f"Macro F1: {results['macro_f1']:.4f}")
```

### 查看错误分析
```python
import pandas as pd
errors = pd.read_csv('results/error_analysis.csv')
print(errors.groupby('aspect').size())
```

## 注意事项

- 每次评估会覆盖之前的结果文件
- 如需保留历史结果，建议重命名或备份
- 混淆矩阵图片为PNG格式，分辨率300dpi
