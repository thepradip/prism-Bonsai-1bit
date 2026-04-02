"""CLI entry point for bonsai-bench."""

import argparse
import json
import sys
import time

from . import __version__
from .models import MODELS, download_llama_binary, download_model, resolve_model_names
from .questions import QA_QUESTIONS
from .runner import run_benchmark
from .benchmarks.memory import run_memory_benchmark
from .reporting.pdf import generate_pdf


def cmd_run(args):
    """Run QA benchmark."""
    model_names = resolve_model_names(args.models)
    if not model_names:
        print("No valid models specified."); sys.exit(1)

    print(f"Bonsai-Bench v{__version__}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Questions: {len(QA_QUESTIONS)}")

    # Ensure binaries + models are downloaded
    print("\nSetup:")
    download_llama_binary()
    for name in model_names:
        download_model(name)

    # Run QA benchmark
    print("\nRunning QA Benchmark...")
    qa_results = run_benchmark(model_names, QA_QUESTIONS, max_tokens=args.max_tokens)

    # Run memory benchmark if requested
    mem_results = None
    if args.memory:
        print("\nRunning Memory Benchmark...")
        mem_results = run_memory_benchmark(model_names)

    # Print summary
    print_summary(qa_results)

    # Save JSON
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    output = {
        "benchmark": "Bonsai 1-Bit QA Benchmark",
        "version": __version__,
        "timestamp": timestamp,
        "models": model_names,
        "results": qa_results,
    }
    if mem_results:
        output["memory"] = mem_results

    json_path = args.output or "bonsai_bench_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON results saved to {json_path}")

    # Generate PDF if requested
    if args.pdf:
        pdf_path = args.pdf
        generate_pdf(qa_results, mem_results, pdf_path, timestamp)
        print(f"PDF report saved to {pdf_path}")


def cmd_download(args):
    """Download models and binaries."""
    model_names = resolve_model_names(args.models)
    print("Downloading llama.cpp binaries...")
    download_llama_binary()
    for name in model_names:
        download_model(name)
    print("Done.")


def cmd_list(args):
    """List available models."""
    print("Available Bonsai Models:")
    print(f"  {'Name':<15} {'Params':<8} {'GGUF Repo'}")
    print(f"  {'-'*15} {'-'*8} {'-'*40}")
    for name, info in MODELS.items():
        print(f"  {name:<15} {info['params']:<8} {info['gguf_repo']}")


def print_summary(qa_results):
    """Print pass/fail summary to console."""
    models = list(qa_results.keys())

    # Gather categories
    cats = {}
    for r in qa_results[models[0]]:
        cats.setdefault(r["category"], []).append(r["id"])

    n = len(qa_results[models[0]])
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  RESULTS SUMMARY")
    print(sep)

    print(f"\n  {'Category':<20} {'Count':>5}", end="")
    for m in models:
        print(f"  | {m:>14}", end="")
    print()
    print(f"  {'-'*20} {'-'*5}", end="")
    for _ in models:
        print(f"--+{'-'*14}", end="")
    print()

    for cat, ids in cats.items():
        print(f"  {cat:<20} {len(ids):>5}", end="")
        for m in models:
            p = sum(1 for r in qa_results[m] if r["category"] == cat and r["accepted"])
            print(f"  | {p}/{len(ids)} ({p/len(ids)*100:.0f}%)", end="")
        print()

    print(f"  {'-'*20} {'-'*5}", end="")
    for _ in models:
        print(f"--+{'-'*14}", end="")
    print()

    print(f"  {'TOTAL':<20} {n:>5}", end="")
    for m in models:
        p = sum(1 for r in qa_results[m] if r["accepted"])
        print(f"  | {p}/{n} ({p/n*100:.0f}%)", end="")
    print()
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        prog="bonsai-bench",
        description="QA & Memory benchmark toolkit for Bonsai 1-bit models",
    )
    parser.add_argument("--version", action="version", version=f"bonsai-bench {__version__}")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run QA benchmark on models")
    p_run.add_argument("-m", "--models", default="all",
                       help="Model(s) to test: 'all', '8B', 'Bonsai-4B', or comma-separated (default: all)")
    p_run.add_argument("--max-tokens", type=int, default=400, help="Max tokens to generate (default: 400)")
    p_run.add_argument("--memory", action="store_true", help="Also run memory benchmark at 1K/2K/4K/16K/32K")
    p_run.add_argument("--pdf", type=str, default=None, help="Generate PDF report at this path")
    p_run.add_argument("-o", "--output", type=str, default=None, help="JSON output path (default: bonsai_bench_results.json)")

    # download
    p_dl = sub.add_parser("download", help="Download models and binaries")
    p_dl.add_argument("-m", "--models", default="all", help="Model(s) to download (default: all)")

    # list
    sub.add_parser("list", help="List available models")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
