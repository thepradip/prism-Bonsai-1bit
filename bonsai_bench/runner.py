"""Core benchmark runner — runs questions against models via llama.cpp."""

import re
import subprocess
import time

from .models import get_llama_cli, get_model_path


def run_llama(model_path, prompt, max_tokens=400):
    """Run a single prompt through llama-cli. Returns (answer, prompt_tps, gen_tps, wall_time)."""
    cli = get_llama_cli()
    cmd = [
        cli, "-fa", "1", "-ngl", "999",
        "-m", model_path, "-p", prompt,
        "-n", str(max_tokens), "--single-turn", "--no-display-prompt",
    ]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    wall_time = time.perf_counter() - t_start

    full = result.stdout + "\n" + result.stderr

    # Parse timing
    prompt_tps = gen_tps = 0.0
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
    if m:
        prompt_tps, gen_tps = float(m.group(1)), float(m.group(2))

    # Parse memory
    memory_mib = 0
    mem_match = re.search(r'\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)', full)
    if mem_match:
        memory_mib = int(mem_match.group(1))

    # Clean answer
    answer = result.stdout
    answer = re.sub(r'\x1b\[[0-9;]*m', '', answer)
    answer = re.sub(r'\[ Prompt:.*?]', '', answer)
    for noise in ["Loading model...", "Exiting...", "llama_memory", "ggml_metal",
                   "ggml_cuda", "build      :", "model      :", "modalities :",
                   "available commands:", "/exit", "/regen", "/clear", "/read"]:
        answer = "\n".join(l for l in answer.split("\n") if noise not in l)
    answer = re.sub(r'^[|/\-\\]+\s*', '', answer)

    return answer.strip(), prompt_tps, gen_tps, wall_time, memory_mib


def run_benchmark(model_names, questions, max_tokens=400, verbose=True):
    """Run all questions on all models. Returns {model_name: [results]}."""
    all_results = {}

    for name in model_names:
        model_path = get_model_path(name)
        results = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"  {name}")
            print(f"{'='*70}")

        for qa in questions:
            answer, pp, tg, wall, mem = run_llama(model_path, qa["question"], max_tokens)
            accepted = qa["accept_fn"](answer.strip())

            result = {
                "id": qa["id"],
                "category": qa["category"],
                "difficulty": qa["difficulty"],
                "description": qa["description"],
                "question": qa["question"],
                "answer": answer[:800],
                "prompt_tps": round(pp, 1),
                "gen_tps": round(tg, 1),
                "wall_time_s": round(wall, 2),
                "memory_mib": mem,
                "accepted": accepted,
            }
            results.append(result)

            if verbose:
                tag = "PASS" if accepted else "FAIL"
                print(f"  Q{qa['id']:2d} [{qa['category']:>15s}] "
                      f"Wall={wall:>5.1f}s  Gen={tg:>5.0f}t/s  [{tag}]", flush=True)

        all_results[name] = results

    return all_results
