#!/usr/bin/env python3
"""
Needle-in-Haystack Test for Bonsai + TurboQuant
================================================
Inserts a hidden fact ("needle") at various positions in a long context
("haystack") and tests if the model can retrieve it.

Tests at 4K, 16K, 32K, 64K, 128K contexts with FP16/TQ-Q8/TQ-Q4 KV cache.
"""

import os
import re
import subprocess
import time
import json
import random
import sys

from ..models import get_llama_cli, get_model_path, MODELS

# ── Needles: unique facts to hide in context ───────────────────────
NEEDLES = [
    {
        "id": "N1",
        "fact": "The secret project code name is Operation Crimson Falcon and it launched on July 7, 2019.",
        "question": "What is the secret project code name and when did it launch?",
        "check": lambda a: "crimson falcon" in a.lower() and ("july" in a.lower() or "2019" in a),
    },
    {
        "id": "N2",
        "fact": "The company's backup data center is located at 742 Evergreen Terrace, Springfield, with a capacity of exactly 8,421 servers.",
        "question": "Where is the backup data center located and how many servers does it have?",
        "check": lambda a: ("742" in a or "evergreen" in a.lower()) and "8,421" in a or "8421" in a,
    },
    {
        "id": "N3",
        "fact": "Dr. Elena Vasquez holds the patent for the TurboCache algorithm, filed under patent number US-2024-7734291.",
        "question": "Who holds the patent for the TurboCache algorithm and what is the patent number?",
        "check": lambda a: "vasquez" in a.lower() and ("7734291" in a or "7734" in a),
    },
]

# ── Haystack: filler paragraphs to pad context ────────────────────
FILLER_PARAGRAPHS = [
    "Cloud computing has revolutionized how organizations deploy and manage their IT infrastructure. By leveraging shared pools of configurable computing resources, businesses can achieve economies of scale while maintaining flexibility. The adoption of cloud services continues to grow across all industry sectors, driven by the need for scalability and cost optimization.",
    "Database management systems are critical components of modern software architectures. Relational databases use structured query language for data manipulation, while NoSQL databases offer flexible schemas for unstructured data. The choice between these approaches depends on factors such as data consistency requirements, query patterns, and scalability needs.",
    "Network security remains a top priority for enterprise organizations. Implementing defense-in-depth strategies involves multiple layers of security controls including firewalls, intrusion detection systems, and encryption protocols. Regular security audits and penetration testing help identify vulnerabilities before they can be exploited.",
    "Software development methodologies have evolved significantly over the past decades. Agile frameworks emphasize iterative development and continuous feedback, while DevOps practices bridge the gap between development and operations teams. Continuous integration and continuous deployment pipelines automate the build, test, and release process.",
    "Machine learning algorithms learn patterns from data to make predictions or decisions. Supervised learning uses labeled training data, while unsupervised learning discovers hidden structures in unlabeled data. Deep learning architectures with multiple neural network layers have achieved state-of-the-art results in computer vision and natural language processing.",
    "Microservices architecture decomposes applications into small, independent services that communicate through well-defined APIs. Each service can be developed, deployed, and scaled independently, enabling teams to work autonomously. Container orchestration platforms like Kubernetes manage the lifecycle of these distributed services.",
    "Data engineering pipelines transform raw data into actionable insights through extraction, transformation, and loading processes. Modern data platforms support both batch and stream processing paradigms. Data quality monitoring and governance frameworks ensure the reliability and trustworthiness of analytical outputs.",
    "Operating systems manage hardware resources and provide services for application software. Process scheduling algorithms determine the order in which processes execute on the CPU. Memory management subsystems use techniques like virtual memory and paging to efficiently allocate limited physical memory.",
    "Distributed systems coordinate multiple computers to work together as a single coherent system. Consensus protocols like Raft and Paxos ensure agreement among nodes despite failures. CAP theorem states that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance.",
    "Web application frameworks provide structure and tools for building dynamic websites and APIs. Frontend frameworks handle user interface rendering and state management, while backend frameworks manage business logic and data persistence. Modern full-stack development often combines both approaches with shared data formats like JSON.",
    "Cryptographic protocols protect data confidentiality and integrity during transmission and storage. Public key infrastructure enables secure communication between parties who have not previously established trust. Hash functions create fixed-size digests of arbitrary data, useful for password storage and data integrity verification.",
    "Version control systems track changes to source code over time, enabling collaboration among developers. Branching strategies define how teams manage parallel development efforts. Code review processes improve quality by having peers examine changes before they are integrated into the main codebase.",
    "Load testing evaluates system performance under expected and peak usage conditions. Metrics such as response time, throughput, and error rate indicate whether the system meets its performance requirements. Capacity planning uses these measurements to determine the resources needed to support projected growth.",
    "API design principles guide the creation of interfaces that are intuitive, consistent, and maintainable. RESTful APIs use standard HTTP methods and status codes to represent operations on resources. GraphQL offers an alternative approach where clients specify exactly what data they need in a single request.",
    "Observability encompasses logging, metrics, and distributed tracing to understand system behavior. Structured logging formats enable efficient search and analysis of log data. Service mesh technologies provide observability features transparently to application code.",
]


def generate_haystack(target_tokens, needle_text, needle_position=0.5):
    """Generate a haystack with a needle inserted at the specified position.

    needle_position: 0.0 = beginning, 0.5 = middle, 1.0 = end
    """
    # Rough estimate: 1 token ≈ 4 characters
    target_chars = target_tokens * 4

    # Build filler text
    rng = random.Random(42)
    filler_parts = []
    chars_so_far = 0

    while chars_so_far < target_chars:
        para = rng.choice(FILLER_PARAGRAPHS)
        filler_parts.append(para)
        chars_so_far += len(para) + 2  # +2 for newlines

    # Insert needle at target position
    insert_idx = max(0, min(int(len(filler_parts) * needle_position), len(filler_parts) - 1))
    filler_parts.insert(insert_idx, f"\n{needle_text}\n")

    haystack = "\n\n".join(filler_parts)

    # Trim to approximate target
    if len(haystack) > target_chars:
        # Keep the needle — find its position and trim around it
        needle_pos = haystack.find(needle_text)
        if needle_position < 0.3:
            haystack = haystack[:target_chars]
        elif needle_position > 0.7:
            haystack = haystack[len(haystack) - target_chars:]
        else:
            half = target_chars // 2
            center = needle_pos + len(needle_text) // 2
            start = max(0, center - half)
            haystack = haystack[start:start + target_chars]

    return haystack


KV_CONFIGS = [
    {"name": "FP16", "ctk": "f16", "ctv": "f16"},
    {"name": "TQ-Q8", "ctk": "q8_0", "ctv": "q8_0"},
    {"name": "TQ-Q4", "ctk": "q4_0", "ctv": "q4_0"},
]

# Context sizes to test (in tokens)
CONTEXT_SIZES = [
    (4096, "4K"),
    (16384, "16K"),
    (32768, "32K"),
    (65536, "64K"),
    (131072, "128K"),
]

# Needle positions: where in the context to hide the fact
POSITIONS = [
    (0.0, "Start"),
    (0.25, "25%"),
    (0.5, "Middle"),
    (0.75, "75%"),
    (1.0, "End"),
]


def run_needle_test(model_path, needle, haystack, question, kv_config, ctx_size):
    """Run a single needle-in-haystack test."""
    cli = get_llama_cli()

    prompt = f"""Read the following document carefully and answer the question at the end.

<DOCUMENT>
{haystack}
</DOCUMENT>

Question: {question}
Answer precisely based on the document above."""

    cmd = [
        cli, "-fa", "1", "-ngl", "999",
        "-m", model_path,
        "-ctk", kv_config["ctk"], "-ctv", kv_config["ctv"],
        "-c", str(ctx_size),
        "-p", prompt,
        "-n", "100",
        "--single-turn", "--no-display-prompt",
    ]

    t_start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall_time = time.perf_counter() - t_start

        full = result.stdout + "\n" + result.stderr

        # Check for OOM
        if result.returncode != 0 and ("memory" in full.lower() or "alloc" in full.lower()):
            return {"status": "OOM", "answer": "", "wall_time_s": 0, "prompt_tps": 0, "gen_tps": 0}

        # Parse timing
        prompt_tps = gen_tps = 0.0
        m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
        if m:
            prompt_tps, gen_tps = float(m.group(1)), float(m.group(2))

        # Clean answer
        answer = result.stdout
        answer = re.sub(r'\x1b\[[0-9;]*m', '', answer)
        answer = re.sub(r'\[ Prompt:.*?]', '', answer)
        for noise in ["Loading model...", "Exiting...", "llama_memory", "ggml_metal",
                       "ggml_cuda", "build      :", "model      :", "modalities :",
                       "available commands:", "/exit", "/regen", "/clear", "/read"]:
            answer = "\n".join(l for l in answer.split("\n") if noise not in l)
        answer = answer.strip()

        return {
            "status": "OK",
            "answer": answer[:500],
            "wall_time_s": round(wall_time, 2),
            "prompt_tps": round(prompt_tps, 1),
            "gen_tps": round(gen_tps, 1),
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "answer": "", "wall_time_s": 300, "prompt_tps": 0, "gen_tps": 0}
    except Exception as e:
        return {"status": f"ERROR:{e}", "answer": "", "wall_time_s": 0, "prompt_tps": 0, "gen_tps": 0}


def run_needle_benchmark(model_names, verbose=True):
    """Run full needle-in-haystack benchmark."""
    all_results = {
        "benchmark": "Needle-in-Haystack: Bonsai + TurboQuant",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {},
    }

    for model_name in model_names:
        model_path = get_model_path(model_name)
        all_results["results"][model_name] = {}

        for cfg in KV_CONFIGS:
            cfg_name = cfg["name"]
            all_results["results"][model_name][cfg_name] = []

            if verbose:
                print(f"\n{'='*80}")
                print(f"  {model_name} | {cfg_name}")
                print(f"{'='*80}")

            for ctx_tokens, ctx_label in CONTEXT_SIZES:
                for pos_val, pos_label in POSITIONS:
                    for needle in NEEDLES:
                        # Generate haystack
                        haystack = generate_haystack(
                            ctx_tokens - 200,  # leave room for prompt wrapper + answer
                            needle["fact"],
                            pos_val
                        )

                        result = run_needle_test(
                            model_path, needle["fact"], haystack,
                            needle["question"], cfg, ctx_tokens
                        )

                        found = False
                        if result["status"] == "OK":
                            found = needle["check"](result["answer"])

                        entry = {
                            "needle_id": needle["id"],
                            "ctx_label": ctx_label,
                            "ctx_tokens": ctx_tokens,
                            "position": pos_label,
                            "found": found,
                            **result,
                        }
                        all_results["results"][model_name][cfg_name].append(entry)

                        if verbose:
                            if result["status"] == "OOM":
                                tag = " OOM"
                            elif result["status"] == "TIMEOUT":
                                tag = "T/O "
                            elif found:
                                tag = "FOUND"
                            else:
                                tag = "MISS"
                            print(f"  {needle['id']} ctx={ctx_label:>5} pos={pos_label:>6} "
                                  f"Wall={result['wall_time_s']:>6.1f}s "
                                  f"PP={result['prompt_tps']:>6.0f}t/s "
                                  f"[{tag}]", flush=True)

    return all_results


def print_needle_report(results):
    """Print needle-in-haystack results as a heatmap-style table."""
    models = list(results["results"].keys())
    sep = "=" * 100

    print(f"\n{sep}")
    print(f"  NEEDLE-IN-HAYSTACK RESULTS")
    print(f"  {results['timestamp']}")
    print(sep)

    for model in models:
        print(f"\n  {model}")
        print(f"  {'─'*90}")

        for cfg_name in ["FP16", "TQ-Q8", "TQ-Q4"]:
            entries = results["results"][model].get(cfg_name, [])
            if not entries:
                continue

            print(f"\n    Config: {cfg_name}")

            # Build heatmap: context x position
            ctx_labels = []
            for e in entries:
                if e["ctx_label"] not in ctx_labels:
                    ctx_labels.append(e["ctx_label"])

            pos_labels = ["Start", "25%", "Middle", "75%", "End"]

            print(f"    {'Context':>8}", end="")
            for p in pos_labels:
                print(f" | {p:>8}", end="")
            print(f" | {'Score':>6}")
            print(f"    {'─'*8}", end="")
            for _ in pos_labels:
                print(f"─┼{'─'*8}", end="")
            print(f"─┼{'─'*6}")

            for ctx_l in ctx_labels:
                print(f"    {ctx_l:>8}", end="")
                ctx_entries = [e for e in entries if e["ctx_label"] == ctx_l]
                total = 0
                found = 0
                for pos in pos_labels:
                    pos_entries = [e for e in ctx_entries if e["position"] == pos]
                    if not pos_entries:
                        print(f" | {'--':>8}", end="")
                        continue
                    # Count found across all needles at this position
                    n_found = sum(1 for e in pos_entries if e["found"])
                    n_total = len(pos_entries)
                    total += n_total
                    found += n_found
                    if pos_entries[0]["status"] == "OOM":
                        print(f" | {'OOM':>8}", end="")
                    elif n_found == n_total:
                        print(f" | {n_found}/{n_total} OK", end="")
                    elif n_found == 0:
                        print(f" | {n_found}/{n_total} --", end="")
                    else:
                        print(f" | {n_found}/{n_total}   ", end="")

                score = f"{found}/{total}" if total > 0 else "--"
                print(f" | {score:>6}")

        print()

    # Summary table
    print(f"\n  SUMMARY: Total Retrieval Rate")
    print(f"  {'─'*70}")
    print(f"  {'Model':<15} {'Config':<8}", end="")
    for _, ctx_l in CONTEXT_SIZES:
        print(f" | {ctx_l:>6}", end="")
    print(f" | {'Total':>6}")
    print(f"  {'─'*15} {'─'*8}", end="")
    for _ in CONTEXT_SIZES:
        print(f"─┼{'─'*6}", end="")
    print(f"─┼{'─'*6}")

    for model in models:
        for cfg_name in ["FP16", "TQ-Q8", "TQ-Q4"]:
            entries = results["results"][model].get(cfg_name, [])
            if not entries:
                continue
            print(f"  {model:<15} {cfg_name:<8}", end="")
            grand_found = grand_total = 0
            for _, ctx_l in CONTEXT_SIZES:
                ctx_e = [e for e in entries if e["ctx_label"] == ctx_l]
                if not ctx_e:
                    print(f" | {'--':>6}", end="")
                    continue
                if ctx_e[0]["status"] == "OOM":
                    print(f" |   OOM", end="")
                    continue
                f = sum(1 for e in ctx_e if e["found"])
                t = len(ctx_e)
                grand_found += f
                grand_total += t
                pct = f / t * 100 if t else 0
                print(f" | {pct:>4.0f}%", end="")

            gpct = grand_found / grand_total * 100 if grand_total else 0
            print(f" | {gpct:>4.0f}%")

    print(sep)
