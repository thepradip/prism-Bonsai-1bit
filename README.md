# bonsai-bench

QA & Memory benchmarking toolkit for [PrismML Bonsai 1-bit models](https://huggingface.co/collections/prism-ml/bonsai).

Runs 20 production-quality questions across 4 categories — RAG context comprehension, finance document QA, reasoning & logic, instruction following — and generates PDF reports with per-model comparisons.

## Install

```bash
pip install bonsai-bench
```

Or install from source:

```bash
git clone https://github.com/PrismML-Eng/bonsai-bench.git
cd bonsai-bench
pip install -e .
```

## Quick Start

```bash
# Run full benchmark on all 3 models (downloads everything automatically)
bonsai-bench run

# Run on a specific model
bonsai-bench run -m 8B

# Run on multiple models
bonsai-bench run -m "Bonsai-4B,Bonsai-8B"

# Include memory benchmark + PDF report
bonsai-bench run --memory --pdf report.pdf

# Save JSON results to custom path
bonsai-bench run -o my_results.json
```

## Commands

| Command | Description |
|---------|-------------|
| `bonsai-bench run` | Run QA benchmark (downloads models + binaries automatically) |
| `bonsai-bench run --memory` | Also run memory benchmark at 1K/2K/4K/16K/32K context |
| `bonsai-bench run --pdf report.pdf` | Generate PDF report |
| `bonsai-bench download` | Download models and llama.cpp binaries only |
| `bonsai-bench download -m 4B` | Download a specific model |
| `bonsai-bench list` | List available models |

## Options for `run`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --models` | Models to test: `all`, `8B`, `4B`, `1.7B`, or comma-separated | `all` |
| `--max-tokens` | Max tokens per response | `400` |
| `--memory` | Run memory benchmark at 1K/2K/4K/16K/32K | off |
| `--pdf PATH` | Generate PDF report | off |
| `-o, --output` | JSON output path | `bonsai_bench_results.json` |

## Available Models

| Model | Params | HuggingFace Repo |
|-------|--------|------------------|
| Bonsai-1.7B | 1.7B | [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) |
| Bonsai-4B | 4B | [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf) |
| Bonsai-8B | 8B | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) |

## What it tests

**20 questions across 4 categories:**

| Category | Questions | Tests |
|----------|-----------|-------|
| RAG Context (5) | Fact extraction, cross-referencing, contradiction detection, refusal when no answer, constrained summarization |
| Finance QA (5) | Ratio calculation, earnings sentiment, data extraction, trend analysis, risk ranking |
| Reasoning (5) | Logic puzzles, root cause analysis, syllogisms, constraint optimization, analogies |
| Instruction (5) | JSON conversion, conditional routing, negative constraints, code review, NL-to-SQL |

## What it measures

- **Wall Time** — total real-world time per question
- **Prompt tok/s (PP)** — prompt processing throughput
- **Generation tok/s (TG)** — text generation throughput
- **Memory** — model weights, KV cache, compute at each context size
- **Quality** — PASS/FAIL per question with keyword + functional acceptance criteria

## How it works

1. Downloads pre-built [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) binaries (Metal GPU on Mac, CUDA on Linux)
2. Downloads GGUF models from HuggingFace
3. Runs each question through `llama-cli` with greedy decoding
4. Parses timing and memory from llama.cpp output
5. Evaluates answers against acceptance criteria
6. Generates JSON + optional PDF report

All downloads are cached in `~/.cache/bonsai-bench/`.

## Requirements

- Python >= 3.10
- macOS (Apple Silicon) or Linux (CUDA GPU)
- ~4GB disk space for all 3 models

## License

MIT
