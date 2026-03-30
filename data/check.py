"""Check dataset with hidden states.

Usage:
python -m eagle.utils.check_dataset_hidden --dataset-path data/scores_rb_hidden/shareGPT-llama3-d7-topk10-t1/
"""
import argparse
import os
import sys
import base64
import io
import numpy as np
import datasets
from pprint import pprint


def resolve_path(path, project_root):
    """将相对路径转换为绝对路径"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def decode_hidden_state(hidden_b64):
    """解码base64编码的hidden state"""
    buffer = io.BytesIO(base64.b64decode(hidden_b64))
    hidden_np = np.load(buffer).astype(np.float32)
    return hidden_np


def check_dataset(dataset_path, num_samples=3):
    """检查数据集样本"""
    print(f"Loading dataset from: {dataset_path}")
    dataset = datasets.load_from_disk(dataset_path)
    
    print(f"\nDataset info:")
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Test samples: {len(dataset['test'])}")
    print(f"  - Total samples: {len(dataset['train']) + len(dataset['test'])}")
    
    print(f"\nFeatures:")
    pprint(dataset['train'].features)
    
    # 检查训练集中的样本
    print(f"\n{'='*80}")
    print(f"Checking {num_samples} sample(s) from training set:")
    print(f"{'='*80}")
    
    for idx in range(min(num_samples, len(dataset['train']))):
        sample = dataset['train'][idx]
        print(f"\nSample {idx + 1}:")
        print(f"  - Keys: {list(sample.keys())}")
        
        # 检查hidden states
        hidden_keys = [k for k in sample.keys() if '_hidden' in k]
        print(f"  - Hidden states found: {len(hidden_keys)}")
        
        if hidden_keys:
            for i, key in enumerate(sorted(hidden_keys)):
                hidden_b64 = sample[key]
                hidden_np = decode_hidden_state(hidden_b64)
                print(f"    {key}: shape={hidden_np.shape}, dtype={hidden_np.dtype}")
                print(f"      Stats: min={hidden_np.min():.4f}, max={hidden_np.max():.4f}, mean={hidden_np.mean():.4f}")
                
                # 检查是否有NaN或Inf
                if np.isnan(hidden_np).any():
                    print(f"      WARNING: Contains NaN values!")
                if np.isinf(hidden_np).any():
                    print(f"      WARNING: Contains Inf values!")
        
        # 检查其他字段
        forward_keys = [k for k in sample.keys() if '_forward' in k]
        print(f"  - Forward scores found: {len(forward_keys)}")
        
        action_keys = [k for k in sample.keys() if 'action_' in k and not 'forward' in k and not 'hidden' in k]
        print(f"  - Action keys found: {len(action_keys)}")
        
        # 打印一个action示例
        if action_keys:
            print(f"  - Action sample ({action_keys[-1]}):")
            pprint(sample[action_keys[-1]], depth=3)
    
    print(f"\n{'='*80}")
    print("Dataset check completed!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Check dataset with hidden states.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to check"
    )
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    print(f"Project root: {project_root}")
    
    # 解析数据集路径
    dataset_path = resolve_path(args.dataset_path, project_root)
    
    check_dataset(dataset_path, args.num_samples)


if __name__ == "__main__":
    main()
