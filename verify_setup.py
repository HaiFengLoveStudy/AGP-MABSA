#!/usr/bin/env python3
# verify_setup.py - 验证项目设置和依赖

import sys
import os

def check_directories():
    """检查必要的目录结构"""
    print("检查目录结构...")
    required_dirs = [
        'src/data', 'src/models', 'src/losses', 'src/training', 'src/evaluation',
        'data/raw', 'data/processed', 'data/images',
        'models/pretrained', 'models/checkpoints',
        'configs', 'logs', 'results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"  ❌ 缺失目录: {dir_path}")
        else:
            print(f"  ✅ {dir_path}")
    
    if missing_dirs:
        print(f"\n缺少 {len(missing_dirs)} 个目录")
        return False
    else:
        print("\n✅ 所有目录结构完整")
        return True

def check_dependencies():
    """检查Python依赖"""
    print("\n检查Python依赖...")
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            missing_packages.append(name)
            print(f"  ❌ {name} 未安装")
    
    if missing_packages:
        print(f"\n缺少 {len(missing_packages)} 个依赖包")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖已安装")
        return True

def check_cuda():
    """检查CUDA可用性"""
    print("\n检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False

def check_files():
    """检查关键文件"""
    print("\n检查关键文件...")
    required_files = [
        'train.py',
        'evaluate.py',
        'requirements.txt',
        'configs/training_config.yaml',
        'src/models/agp_model.py',
        'src/losses/total_loss.py',
        'src/training/trainer.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"  ❌ 缺失文件: {file_path}")
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\n缺少 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✅ 所有关键文件存在")
        return True

def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print("""
1. 准备数据:
   - 将JSONL数据放入 data/raw/ 目录
   - 将图像放入 data/images/ 目录

2. 运行LLM扩写（需要OpenAI API密钥）:
   export OPENAI_API_KEY="your-api-key"
   python src/data/llm_expansion.py

3. 开始训练:
   python train.py

4. 评估模型:
   python evaluate.py

5. 查看详细文档:
   - README.md: 项目概述
   - AGP_EXPERIMENT_PROCEDURE.md: 完整实验步骤
""")

def main():
    print("="*60)
    print("AGP-MABSA 项目设置验证")
    print("="*60)
    
    checks = [
        check_directories(),
        check_files(),
        check_dependencies(),
        check_cuda()
    ]
    
    print("\n" + "="*60)
    if all(checks[:3]):  # 前三项必须通过
        print("✅ 项目设置验证通过！")
        if not checks[3]:
            print("⚠️  注意：CUDA不可用，训练速度会较慢")
        print_next_steps()
        return 0
    else:
        print("❌ 项目设置验证失败")
        print("请修复上述问题后重试")
        return 1

if __name__ == '__main__':
    sys.exit(main())
