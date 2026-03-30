import os
import subprocess

# datasets = ['mt_bench', 'alpaca', 'gsm8k', 'mbpp']
datasets = ['alpaca', 'gsm8k', 'mbpp']
# methods = ['baseline', 'hass', 'radar']
methods = ['radar']

base_model_path = "/root/autodl-tmp/weights/hf/Meta-Llama-3-8B-Instruct"
ea_model_path = "/root/autodl-tmp/weights/eagle/HASS-LLaMA3-Instruct-8B"
eye_model_path = "/root/code/HASS/output/shareGPT/llama3.1/t1d7/pt/b-0.00216-a-0.00100-g-0.99-lr-1e-04-wd-1e-04-dr-0.10-final.pt"

num_gpus_total = 3
num_gpus_per_model = 1
depth = 7
top_k = 10
temperature = 1.0

# Base output dir
output_dir_base = "output/{dataset}/llama3.1/t1d7/5090"

def main():
    for dataset in datasets:
        out_dir = output_dir_base.format(dataset=dataset)
        os.makedirs(out_dir, exist_ok=True)
        
        for method in methods:
            print(f"\n[{method.upper()}] Starting evaluation on {dataset}...")
            if method == 'baseline':
                script = "evaluation.gen_baseline_answer_llama3chat"
                answer_file = f"{out_dir}/baseline.jsonl"
                cmd = [
                    "python", "-m", script,
                    "--base-model-path", base_model_path,
                    "--ea-model-path", ea_model_path,
                    "--bench-name", dataset,
                    "--num-gpus-total", str(num_gpus_total),
                    "--num-gpus-per-model", str(num_gpus_per_model),
                    "--temperature", str(temperature),
                    "--answer-file", answer_file
                ]
            elif method == 'hass':
                script = "evaluation.gen_ea_answer_llama3chat"
                answer_file = f"{out_dir}/hass.jsonl"
                cmd = [
                    "python", "-m", script,
                    "--base-model-path", base_model_path,
                    "--ea-model-path", ea_model_path,
                    "--bench-name", dataset,
                    "--num-gpus-total", str(num_gpus_total),
                    "--num-gpus-per-model", str(num_gpus_per_model),
                    "--depth", str(depth),
                    "--top-k", str(top_k),
                    "--temperature", str(temperature),
                    "--answer-file", answer_file
                ]
            elif method == 'radar':
                script = "evaluation.gen_ea_we_answer_llama3chat"
                answer_file = f"{out_dir}/radar.jsonl"
                cmd = [
                    "python", "-m", script,
                    "--base-model-path", base_model_path,
                    "--ea-model-path", ea_model_path,
                    "--eye-model-path", eye_model_path,
                    "--bench-name", dataset,
                    "--num-gpus-total", str(num_gpus_total),
                    "--num-gpus-per-model", str(num_gpus_per_model),
                    "--depth", str(depth),
                    "--top-k", str(top_k),
                    "--temperature", str(temperature),
                    "--answer-file", answer_file
                ]
            
            # Print the command being executed
            print(" ".join(cmd))
            
            # Execute the command
            subprocess.run(cmd)

if __name__ == "__main__":
    main()
