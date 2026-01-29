#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 测试脚本
用于测试API配置是否正确
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从llm_expansion导入配置和函数
from llm_expansion import (
    API_BASE_URL, API_KEY, MODEL_NAME, TEMPERATURE,
    MAX_RETRIES, NUM_THREADS,
    call_llm_api, expand_aspect_with_retry
)


def test_api_connection():
    """测试API连接"""
    print("="*60)
    print("LLM API 连接测试")
    print("="*60)
    print(f"API地址: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print(f"温度: {TEMPERATURE}")
    print(f"最大重试: {MAX_RETRIES}")
    print(f"并发线程数: {NUM_THREADS}")
    print("="*60)
    
    # 检查API key
    if API_KEY == "your-api-key-here":
        print("\n⚠️  警告: API_KEY未配置！")
        print("请设置环境变量或修改 llm_expansion.py 中的配置")
        return False
    
    print(f"\nAPI Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    # 测试简单请求
    print("\n正在测试API连接...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, API test successful!'"}
    ]
    
    result = call_llm_api(messages, max_tokens=20)
    
    if result:
        print(f"✅ API连接成功!")
        print(f"响应: {result}")
        return True
    else:
        print(f"❌ API连接失败")
        print("\n故障排查:")
        print("1. 检查API_KEY是否正确")
        print("2. 检查API_BASE_URL是否正确")
        print("3. 检查网络连接")
        print("4. 检查API余额/限额")
        return False


def test_aspect_expansion():
    """测试方面词扩写"""
    print("\n" + "="*60)
    print("测试方面词扩写")
    print("="*60)
    
    test_aspects = ["food", "service", "ambience"]
    
    for aspect in test_aspects:
        print(f"\n测试方面词: '{aspect}'")
        expansion = expand_aspect_with_retry(aspect, max_retries=2)
        print(f"扩写结果: '{expansion}'")
        
        if expansion == aspect:
            print("⚠️  扩写失败，返回了原词")
        else:
            print("✅ 扩写成功")
    
    return True


def main():
    """主函数"""
    print("\n🔧 开始测试 LLM API 配置...\n")
    
    # 测试1: API连接
    success = test_api_connection()
    
    if not success:
        print("\n❌ API连接测试失败，请检查配置后重试")
        return 1
    
    # 测试2: 方面词扩写
    print("\n" + "-"*60)
    test_aspect_expansion()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)
    print("\n💡 下一步:")
    print("   运行: python src/data/llm_expansion.py")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
