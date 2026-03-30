import json
import numpy as np
import pandas as pd
import os

datasets = ['mt_bench', 'alpaca', 'gsm8k', 'mbpp']
methods = ['baseline', 'hass', 'radar']
output_dir_base = "output/{dataset}/llama3.1/t1d7/5090"

results = []

for dataset in datasets:
    out_dir = output_dir_base.format(dataset=dataset)
    for method in methods:
        file_path = f"{out_dir}/{method}.jsonl"
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        speeds = []
        all_action_lengths = []
        all_acc_lengths = []
        
        for datapoint in data:
            for ch in datapoint.get("choices", []):
                tokens = sum(ch.get('new_tokens', []))
                times = sum(ch.get('wall_time', []))
                if times > 0:
                    speeds.append(tokens / times)
                # if method != 'baseline':
                #     if method == 'hass':
                #         accl = ch['accept_length'] 
                #     else:
                #         accl = ch['accept_lengths']
                #     if isinstance(accl, list):
                #         for item in accl:
                #             if isinstance(item, list):
                #                 all_acc_lengths.extend(item)
                #             elif isinstance(item, (int, float)):
                #                 all_acc_lengths.append(item)
                
                # Extract action_lengths
                if method == 'radar' and 'action_lengths' in ch:
                    al = ch['action_lengths']
                    if isinstance(al, list):
                        for item in al:
                            if isinstance(item, list):
                                all_action_lengths.extend(item)
                            elif isinstance(item, (int, float)):
                                all_action_lengths.append(item)
                                
        avg_speed = np.mean(speeds) if speeds else None
        std_speed = np.std(speeds) if speeds else None
        avg_action_length = np.mean(all_action_lengths) if all_action_lengths else None
        # avg_acc_length = np.mean(all_acc_lengths) if all_acc_lengths else None
        
        results.append({
            'Dataset': dataset,
            'Method': method,
            'Avg_Speed (tokens/s)': avg_speed,
            'Speed_Std': std_speed,
            'Avg_Action_Length': avg_action_length if method == 'radar' else None
            # 'Avg_Accept_Length': avg_acc_length if method != 'baseline' else None
        })

df = pd.DataFrame(results)
print(df)
output_csv = "evaluation_results.csv"
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
