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
