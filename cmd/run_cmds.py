#!/usr/bin/env python3
"""
run_cmds.py — 一体化入口：生成 / 执行 / 分析。

模式（--mode）：
    gen+run       先生成 cmd.json 再执行
    gen-only      仅生成 cmd.json
    run-only      从已有 cmd.json 读取并执行（默认）
    analyze-only  仅做速度统计分析
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List


# ──────────────────────────────────────────────
# conda 初始化片段（兼容非交互式 shell）
# ──────────────────────────────────────────────

def build_conda_init_snippet(conda_env):
    # type: (str) -> str
    """
    返回一段 shell 代码，依次尝试多种方式初始化 conda 并 activate。
    兼容 miniconda / anaconda / autodl 等不同安装路径。
    """
    return (
        '# --- conda init ---\n'
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then\n'
        '    source "$HOME/miniconda3/etc/profile.d/conda.sh"\n'
        'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then\n'
        '    source "$HOME/anaconda3/etc/profile.d/conda.sh"\n'
        'elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then\n'
        '    source "/opt/conda/etc/profile.d/conda.sh"\n'
        'elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then\n'
        '    source "$HOME/miniconda/etc/profile.d/conda.sh"\n'
        'elif command -v conda &>/dev/null; then\n'
        '    eval "$(conda shell.bash hook)"\n'
        'else\n'
        '    echo "[ERROR] conda not found"; exit 1\n'
        'fi\n'
        'conda activate {env}\n'
        '# --- end conda init ---\n'
    ).format(env=conda_env)


def load_commands(cmd_file):
    # type: (str) -> List[Dict]
    if not os.path.isfile(cmd_file):
        print("[ERROR] cmd file not found: {}".format(cmd_file), file=sys.stderr)
        sys.exit(1)
    with open(cmd_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_command(entry, workdir, conda_env, cuda_devices, dry_run=False):
    # type: (Dict, str, str, str, bool) -> bool
    idx = entry.get("idx", "?")
    bench = entry.get("bench", "?")
    cmd_type = entry.get("type", "?")
    cmd_str = entry["cmd"]

    print("\n" + "=" * 72)
    print("[{}] bench={}  type={}".format(idx, bench, cmd_type))
    print("  CMD: {}".format(cmd_str))
    print("=" * 72)

    if dry_run:
        print("  (dry-run, skipped)")
        return True

    output_file = entry.get("output_file", "")
    if output_file:
        out_dir = os.path.join(workdir, os.path.dirname(output_file))
        os.makedirs(out_dir, exist_ok=True)

    # 拼接完整 shell 脚本：conda init + activate + 实际命令
    shell_script = build_conda_init_snippet(conda_env) + cmd_str

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    t0 = time.time()
    try:
        result = subprocess.run(
            ["bash", "-e", "-c", shell_script],
            cwd=workdir, env=env,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print("  [FAIL] return code {}  ({:.1f}s)".format(result.returncode, elapsed))
            return False
        print("  [OK] ({:.1f}s)".format(elapsed))
        return True
    except Exception as e:
        print("  [EXCEPTION] {}".format(e))
        return False


def execute_all(commands, workdir, conda_env, cuda_devices, dry_run=False, start_from=0):
    # type: (List[Dict], str, str, str, bool, int) -> None
    total = len(commands)
    ok = fail = skip = 0

    for entry in commands:
        entry_idx = int(entry.get("idx", 0))
        if entry_idx < start_from:
            skip += 1
            continue
        success = run_single_command(entry, workdir, conda_env, cuda_devices, dry_run)
        if success:
            ok += 1
        else:
            fail += 1

    print("\n" + "-" * 72)
    print("Done: ok={}  fail={}  skip={}  total={}".format(ok, fail, skip, total))


def build_analyze_config_from_cmd(commands):
    # type: (List[Dict]) -> Dict[str, Dict[str, str]]
    paths = {}  # type: Dict[str, Dict[str, str]]
    for entry in commands:
        bench = entry["bench"]
        cmd_type = entry["type"]
        output_file = entry["output_file"]
        if bench not in paths:
            paths[bench] = {}
        if cmd_type == "baseline":
            paths[bench]["baseline"] = output_file
        elif cmd_type == "eagle":
            paths[bench]["eagle"] = output_file
        elif cmd_type == "radar":
            tag = entry.get("radar_tag", "radar")
            paths[bench]["radar-{}".format(tag)] = output_file
    return paths


def invoke_gen_cmds(args):
    gen_argv = [
        sys.executable, "gen_cmds.py",
        "--eagle", args.eagle,
        "--llm", args.llm,
        "--llm-alias", args.llm_alias,
        "--num-choices", str(args.num_choices),
        "--out", args.cmd_file,
        "--benches",
    ] + args.benches
    if args.radars:
        gen_argv += ["--radars"] + args.radars
    print("[run_cmds] invoking gen_cmds.py ...")
    result = subprocess.run(gen_argv, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print("[ERROR] gen_cmds.py failed", file=sys.stderr)
        sys.exit(1)


def invoke_analyze(args):
    analyze_argv = [
        sys.executable, "analyze_speed.py",
        "--csv", args.analyze_csv,
    ]

    if args.analyze_config:
        analyze_argv += ["--config", args.analyze_config]
    elif args.cmd_file and os.path.isfile(args.cmd_file):
        commands = load_commands(args.cmd_file)
        auto_config = build_analyze_config_from_cmd(commands)
        auto_path = args.cmd_file.replace(".json", "") + "_analyze_paths.json"
        with open(auto_path, "w", encoding="utf-8") as f:
            json.dump(auto_config, f, indent=2, ensure_ascii=False)
        print("[run_cmds] auto-generated analyze config -> {}".format(auto_path))
        analyze_argv += ["--config", auto_path]
    else:
        print("[ERROR] analyze-only needs --analyze-config or --cmd-file", file=sys.stderr)
        sys.exit(1)

    if args.workdir and args.workdir != ".":
        analyze_argv += ["--base-dir", args.workdir]

    print("[run_cmds] invoking analyze_speed.py ...")
    result = subprocess.run(analyze_argv, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print("[ERROR] analyze_speed.py failed", file=sys.stderr)
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Test command runner / unified entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["gen+run", "run-only", "gen-only", "analyze-only"],
                    default="run-only", help="run mode (default run-only)")

    p.add_argument("--cmd-file", default="cmd.json", help="cmd.json path")

    # execution
    p.add_argument("--workdir", default=".", help="working directory")
    p.add_argument("--conda-env", default="radar", help="conda env name (default radar)")
    p.add_argument("--cuda-visible-devices", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--dry-run", action="store_true", help="print commands only, do not execute")
    p.add_argument("--start-from", type=int, default=0, help="start from command idx N (skip earlier)")

    # gen params
    p.add_argument("--eagle", help="Eagle model path")
    p.add_argument("--llm", help="LLM base model path")
    p.add_argument("--llm-alias", help="LLM alias")
    p.add_argument("--benches", nargs="*", default=[], help="bench list")
    p.add_argument("--radars", nargs="*", default=[], help="radar model path list")
    p.add_argument("--num-choices", type=int, default=3, help="num_choices")

    # analyze params
    p.add_argument("--analyze-config", default="", help="analyze paths JSON config")
    p.add_argument("--analyze-csv", default="speed_comparison.csv", help="output CSV path")

    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    if mode in ("gen-only", "gen+run"):
        missing = []
        for field in ("eagle", "llm", "llm_alias", "benches"):
            if not getattr(args, field, None):
                missing.append("--{}".format(field.replace("_", "-")))
        if missing:
            print("[ERROR] gen mode missing required args: {}".format(", ".join(missing)), file=sys.stderr)
            sys.exit(1)
        invoke_gen_cmds(args)

    if mode == "gen-only":
        return

    if mode in ("run-only", "gen+run"):
        commands = load_commands(args.cmd_file)
        execute_all(
            commands,
            workdir=args.workdir,
            conda_env=args.conda_env,
            cuda_devices=args.cuda_visible_devices,
            dry_run=args.dry_run,
            start_from=args.start_from,
        )

    if mode == "analyze-only":
        invoke_analyze(args)


if __name__ == "__main__":
    main()