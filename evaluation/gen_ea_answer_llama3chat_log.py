"""Generate time data for reinforcement learning.

Usage:
python3 gen_ea_answer_llama3chat_log.py --ea-model-path ../weights/eagle/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path ../weights/hf/Meta-Llama-3.1-8B-Instruct --bench-name shareGPT --num-gpus-total 1 --depth 7 --top-k 10 --temperature 0.0 --time-file output/shareGPT/llama3.1/t1d7/time_rl.jsonl --question-begin 0 --question-end 10
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from ..model.eye import Hawkeye,HawkeyeHidden
except:
    from model.ea_model import EaModel
    from model.kv_cache import initialize_past_key_values
    from model.utils import *
    from model.eye import Hawkeye,HawkeyeHidden



def run_eval(
        base_model_path,
        ea_model_path,
        eye_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        time_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    
    shuffled_ids = [q["question_id"] for q in questions]

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                eye_model_path,
                model_id,
                questions[i: i + chunk_size],
                time_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        eye_model_path,
        model_id,
        questions,
        time_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        eye_model_path=eye_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    eye = Hawkeye()
    eye.to(model.base_model.dtype).to(model.base_model.model.layers[-1].self_attn.q_proj.weight.device)
    eye.eval()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    print('Warming up...')
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant."},
        ]
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt], add_special_tokens=False,).input_ids

            # try:
            output_ids, new_token, idx, time_dicts = model.eagenerate_log(
                torch.as_tensor(input_ids).cuda(),
                eye=eye,
                temperature=temperature,
                log=True,
                is_llama3=True,
            )
            # except Exception as e:
            #     print(f'Warmup error: {e}')
            #     continue

            messages.append({
                "role": "assistant",
                "content": "test"
            })
    print('Warmup done')

    os.makedirs(os.path.dirname(time_file), exist_ok=True)
    if os.path.exists(time_file):
        os.remove(time_file)

    for question in tqdm(questions):
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant."},
            ]
            all_turn_time_dicts = []
            
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                try:
                    output_ids, new_token, idx, time_dicts = model.eagenerate_log(
                        torch.as_tensor(input_ids).cuda(),
                        eye=eye,
                        temperature=temperature,
                        log=True,
                        is_llama3=True,
                        max_new_tokens=max_new_token,
                    )
                    
                    all_turn_time_dicts.append(time_dicts)
                    
                    output_ids = output_ids[0][len(input_ids[0]):]
                    stop_token_ids = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    if stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    messages.append({
                        "role": "assistant",
                        "content": output
                    })
                    
                except Exception as e:
                    print(f'Error processing question {question["question_id"]}, turn {j}: {e}')
                    continue

            time_data_entry = {
                "question_id": question["question_id"],
                "model_id": model_id,
                "choice_index": i,
                "turn_time_dicts": all_turn_time_dicts
            }
            
            with open(time_file, "a") as fout:
                fout.write(json.dumps(time_data_entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.3",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.3",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--eye-model-path",
        type=str,
        default=None,
        help="The path to the eye model weights.",
    )
    parser.add_argument(
        "--model-id", type=str, default="eagle3"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--time-file",
        type=str,
        help="The output time data file.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
        default=0,
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total token for draft model",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="total token for draft model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="total token for draft model",
    )
    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.time_file is None:
        args.time_file = f"output/{args.bench_name}/time_rl.jsonl"

    print(f"Time data file: {args.time_file}")
    print(f"Parameters: total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.eye_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        args.time_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )
