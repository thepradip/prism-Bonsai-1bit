"""Memory & throughput benchmark at various context sizes."""

import json
import re
import subprocess
import time

from ..models import get_llama_cli, get_llama_bench, get_model_path

CONTEXT_SIZES = [1024, 2048, 4096, 16384, 32768]
CONTEXT_LABELS = ["1K", "2K", "4K", "16K", "32K"]


def probe_memory(model_path, context_size):
    """Run llama-cli with a context size and parse memory breakdown."""
    cli = get_llama_cli()
    prompt = "Hello " * min(context_size, 500)

    result = subprocess.run(
        [cli, "-m", model_path, "-ngl", "999", "-fa", "1",
         "-c", str(context_size), "-p", prompt, "-n", "1", "--single-turn"],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + "\n" + result.stderr

    info = {"total_mib": 0, "model_mib": 0, "context_mib": 0, "compute_mib": 0, "self_mib": 0}
    for line in output.split("\n"):
        if "memory_breakdown" not in line:
            continue
        if "MTL" in line or "CUDA" in line or "GPU" in line:
            m = re.search(r'(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)', line)
            if m:
                info["total_mib"] = int(m.group(1))
                info["self_mib"] = int(m.group(3))
                info["model_mib"] = int(m.group(4))
                info["context_mib"] = int(m.group(5))
                info["compute_mib"] = int(m.group(6))
    return info


def bench_throughput(model_path):
    """Run llama-bench and return (pp_tps, tg_tps)."""
    bench = get_llama_bench()
    result = subprocess.run(
        [bench, "-m", model_path, "-ngl", "999", "-fa", "1",
         "-p", "512", "-n", "32", "-r", "1", "-o", "json"],
        capture_output=True, text=True, timeout=300,
    )
    full = result.stdout + "\n" + result.stderr
    pp_tps = tg_tps = 0.0
    try:
        start = full.find("[")
        end = full.rfind("]")
        if start >= 0 and end > start:
            data = json.loads(full[start:end + 1])
            for entry in data:
                if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                    pp_tps = entry.get("avg_ts", 0.0)
                elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                    tg_tps = entry.get("avg_ts", 0.0)
    except (json.JSONDecodeError, TypeError):
        pass
    return pp_tps, tg_tps


def run_memory_benchmark(model_names, verbose=True):
    """Run memory benchmark on all models. Returns {model_name: [results_per_ctx]}."""
    all_results = {}

    for name in model_names:
        model_path = get_model_path(name)
        results = []

        if verbose:
            print(f"\n  {name}")

        pp_tps, tg_tps = bench_throughput(model_path)

        for ctx, label in zip(CONTEXT_SIZES, CONTEXT_LABELS):
            mem = probe_memory(model_path, ctx)
            entry = {
                "context_label": label,
                "context_size": ctx,
                **mem,
                "prompt_tps": round(pp_tps, 1),
                "gen_tps": round(tg_tps, 1),
            }
            results.append(entry)

            if verbose:
                print(f"    {label:>5}: Used={mem['self_mib']}MiB  "
                      f"(Model={mem['model_mib']}  KV={mem['context_mib']}  "
                      f"Compute={mem['compute_mib']})", flush=True)

        all_results[name] = results

    return all_results
