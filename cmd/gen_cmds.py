#!/usr/bin/env python3
"""
gen_cmds.py — 根据命令行参数自动生成测试命令，写入 cmd.json。

用法示例：
    python gen_cmds.py \
        --eagle /path/to/eagle \
        --llm /path/to/llm \
        --llm-alias llama3.1 \
        --benches mt_bench gsm8k mbpp \
        --radars /path/radar1 /path/radar2 \
        --num-choices 3 \
        --out cmd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

ngt = 6
ngpm = 2
# ──────────────────────────────────────────────
# 命令模板
# ──────────────────────────────────────────────

TEMPLATE_BASELINE = (
    "python -m eagle.evaluation.gen_baseline_answer_llama3chat"
    " --ea-model-path {eagle}"
    " --base-model-path {llm}"
    " --bench-name {bench}"
    " --num-gpus-total {ngt}"
    " --num-gpus-per-model {ngpm}"
    " --depth 7"
    " --top-k 10"
    " --temperature 1.0"
    " --num-choices {num_choices}"
    " --answer-file {output_file}"
)

TEMPLATE_EAGLE = (
    "python -m eagle.evaluation.gen_ea_answer_llama3chat"
    " --ea-model-path {eagle}"
    " --base-model-path {llm}"
    " --bench-name {bench}"
    " --num-gpus-total {ngt}"
    " --num-gpus-per-model {ngpm}"
    " --depth 7"
    " --top-k 10"
    " --temperature 1.0"
    " --num-choices {num_choices}"
    " --answer-file {output_file}"
)

TEMPLATE_RADAR = (
    "python -m eagle.evaluation.gen_ea_we_answer_llama3chat"
    " --ea-model-path {eagle}"
    " --base-model-path {llm}"
    " --eye-model-path {radar}"
    " --bench-name {bench}"
    " --num-gpus-total {ngt}"
    " --num-gpus-per-model {ngpm}"
    " --depth 7"
    " --top-k 10"
    " --temperature 1.0"
    " --num-choices {num_choices}"
    " --answer-file {output_file}"
)


def radar_tag(radar_path):
    # type: (str) -> str
    """从 radar 路径中提取一个简短标签，用于输出文件名去重。"""
    return os.path.basename(radar_path.rstrip("/"))


def build_output_path(bench, llm_alias, prefix, num_choices):
    # type: (str, str, str, int) -> str
    return "output/{bench}/{alias}/t1d7/A800/{prefix}-speedtest-{nc}.jsonl".format(
        bench=bench, alias=llm_alias, prefix=prefix, nc=num_choices,
    )


def generate_commands(eagle, llm, llm_alias, benches, radars, num_choices):
    # type: (str, str, str, List[str], List[str], int) -> List[Dict]
    """按 bench → baseline → eagle → radar* 的顺序生成命令列表。"""
    commands = []  # type: List[Dict]
    idx = 0
    global ngt, ngpm

    for bench in sorted(benches):
        # --- baseline ---
        out_file = build_output_path(bench, llm_alias, "baseline", num_choices)
        cmd = TEMPLATE_BASELINE.format(
            eagle=eagle, llm=llm, bench=bench, ngt=ngt, ngpm=ngpm,
            num_choices=num_choices, output_file=out_file,
        )
        commands.append({
            "idx": str(idx),
            "bench": bench,
            "type": "baseline",
            "cmd": cmd,
            "output_file": out_file,
        })
        idx += 1

        # --- eagle ---
        out_file = build_output_path(bench, llm_alias, "eagle3", num_choices)
        cmd = TEMPLATE_EAGLE.format(
            eagle=eagle, llm=llm, bench=bench, ngt=ngt, ngpm=ngpm,
            num_choices=num_choices, output_file=out_file,
        )
        commands.append({
            "idx": str(idx),
            "bench": bench,
            "type": "eagle",
            "cmd": cmd,
            "output_file": out_file,
        })
        idx += 1

        # --- radar (0..N) ---
        for radar_path in radars:
            tag = radar_tag(radar_path)
            out_file = build_output_path(bench, llm_alias, "radar-{}".format(tag), num_choices)
            cmd = TEMPLATE_RADAR.format(
                eagle=eagle, llm=llm, radar=radar_path, bench=bench, ngt=ngt, ngpm=ngpm,
                num_choices=num_choices, output_file=out_file,
            )
            commands.append({
                "idx": str(idx),
                "bench": bench,
                "type": "radar",
                "radar_path": radar_path,
                "radar_tag": tag,
                "cmd": cmd,
                "output_file": out_file,
            })
            idx += 1

    return commands


def parse_args():
    p = argparse.ArgumentParser(description="Generate test commands -> cmd.json")
    p.add_argument("--eagle", required=True, help="Eagle model path")
    p.add_argument("--llm", required=True, help="LLM base model path")
    p.add_argument("--llm-alias", required=True, help="LLM alias (for output path)")
    p.add_argument("--benches", nargs="+", required=True, help="bench list")
    p.add_argument("--radars", nargs="*", default=[], help="radar model path list (can be empty)")
    p.add_argument("--num-choices", type=int, default=3, help="num_choices (default 3)")
    p.add_argument("--out", default="cmd.json", help="output cmd.json path (default cmd.json)")
    return p.parse_args()


def main():
    args = parse_args()
    commands = generate_commands(
        eagle=args.eagle,
        llm=args.llm,
        llm_alias=args.llm_alias,
        benches=args.benches,
        radars=args.radars,
        num_choices=args.num_choices,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2, ensure_ascii=False)

    print("[gen_cmds] generated {} commands -> {}".format(len(commands), args.out))
    for c in commands:
        print("  [{}] {:12s} | {:10s} | {}".format(c["idx"], c["bench"], c["type"], c["output_file"]))


if __name__ == "__main__":
    main()
