import json
import numpy as np
import argparse
import os


def process_time_rl(json_path, maxlen=7, output_dir=None, base_model_name="llama3.1", 
                    bench_name="shareGPT", td="t1d7", is_hidden=False):
    """
    处理time_rl.jsonl文件，计算强化学习训练所需的时间参数，并生成训练配置文件
    
    Args:
        json_path: time_rl.jsonl文件路径
        maxlen: 最大深度（默认7）
        output_dir: 输出目录（默认与输入文件同目录）
        base_model_name: 基础模型名称（llama3.1, vicuna13, ds等）
        bench_name: 评测数据集名称
        td: 温度和深度标识，如t1d7
        is_hidden: 是否生成hawkeye_hidden的配置
    """
    
    print(f"Processing {json_path}...")
    
    # 读取time_rl.jsonl数据
    data = []
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    
    print(f"Loaded {len(data)} samples")
    
    # 提取所有时间数据
    all_eagenerate_times = []
    all_eaforward_times = []
    all_eye_times = []
    
    for datapoint in data:
        turn_time_dicts = datapoint["turn_time_dicts"]
        for turn_dict in turn_time_dicts:
            # 每个turn_dict是一个time_dicts列表（每个生成步骤一个time_dict）
            for time_dict in turn_dict:
                if "eagenerate_time" in time_dict:
                    all_eagenerate_times.append(time_dict["eagenerate_time"])
                if "eaforward_times" in time_dict:
                    all_eaforward_times.extend(time_dict["eaforward_times"])
                if "eye_times" in time_dict:
                    all_eye_times.extend(time_dict["eye_times"])
    
    print(f"Collected {len(all_eagenerate_times)} eagenerate times")
    print(f"Collected {len(all_eaforward_times)} eaforward times")
    print(f"Collected {len(all_eye_times)} eye times")
    
    # 计算平均值
    avg_eagenerate_time = np.mean(all_eagenerate_times) if all_eagenerate_times else 0
    avg_eaforward_time = np.mean(all_eaforward_times) if all_eaforward_times else 0
    avg_eye_time = np.mean(all_eye_times) if all_eye_times else 0.0004  # 默认值
    
    # 计算eagen_minus_time = avg_eagenerate_time - avg_eaforward_time * maxlen
    eagen_minus_time = avg_eagenerate_time - avg_eaforward_time * maxlen
    
    # 计算beta = avg_eaforward_time + avg_eye_time
    beta = avg_eaforward_time + avg_eye_time
    
    print("\n=== Time Parameters ===")
    print(f"avg_eagenerate_time: {avg_eagenerate_time:.6f}s")
    print(f"avg_eaforward_time:  {avg_eaforward_time:.6f}s")
    print(f"avg_eye_time:        {avg_eye_time:.6f}s")
    print(f"eagen_minus_time:    {eagen_minus_time:.6f}s")
    print(f"beta:                {beta:.6f}s")
    print(f"maxlen:              {maxlen}")
    
    # 生成训练配置
    config = {
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "beta": [round(beta, 6)],
        "alpha": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        "dropout": [0.1],
        "gamma": [0.99],
        "lstm_hidden": 128,
        "mlp_hidden": 128,
        "num_layers": 1,
        "max_len": maxlen,
        "num_epochs": 100,
        "eagen_minus_time": round(eagen_minus_time, 6),
        "eaforward_time": round(avg_eaforward_time, 6),
        "eye_time": round(avg_eye_time, 6),
        "bench_name": bench_name,
        "base_model_name": base_model_name,
        "td": td,
        "batch_size": 64
    }
    
    if is_hidden:
        config["hidden_dim"] = 4096
        config["reduce_dim"] = 256
        config["dataset_path"] = f"data/scores_rb_hidden/{bench_name}-{base_model_name}-d{maxlen}-topk10-t1"
        config_filename = f"train_{base_model_name}_hidden.json"
    else:
        config["state_dim"] = 10
        config["dataset_path"] = f"data/scores_rb/{bench_name}-{base_model_name}-d{maxlen}-topk10-t1"
        config_filename = f"eye_{base_model_name}.json"
    
    # 确定输出路径
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
        if not output_dir:
            output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, config_filename)
    
    # 写入配置文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\nConfig saved to: {output_path}")
    
    # 返回计算结果
    return {
        "avg_eagenerate_time": avg_eagenerate_time,
        "eagen_minus_time": eagen_minus_time,
        "avg_eaforward_time": avg_eaforward_time,
        "avg_eye_time": avg_eye_time,
        "beta": beta,
        "config_path": output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process time_rl.jsonl to generate training config")
    parser.add_argument("--json-path", type=str, required=True, help="Path to time_rl.jsonl")
    parser.add_argument("--maxlen", type=int, default=7, help="Max depth (default: 7)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--base-model-name", type=str, default="llama3.1", 
                       help="Base model name (llama3.1, vicuna13, ds)")
    parser.add_argument("--bench-name", type=str, default="shareGPT", 
                       help="Benchmark name (shareGPT, mt_bench, etc.)")
    parser.add_argument("--td", type=str, default="t1d7", help="Temperature and depth tag, e.g., t1d7")
    parser.add_argument("--hidden", action="store_true", help="Generate config for hawkeye_hidden")
    
    args = parser.parse_args()
    
    results = process_time_rl(
        json_path=args.json_path,
        maxlen=args.maxlen,
        output_dir=args.output_dir,
        base_model_name=args.base_model_name,
        bench_name=args.bench_name,
        td=args.td,
        is_hidden=args.hidden
    )
