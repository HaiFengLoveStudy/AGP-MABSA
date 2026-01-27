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
