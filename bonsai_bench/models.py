"""Model registry and download helpers."""

import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

from huggingface_hub import snapshot_download

# ── Model Registry ─────────────────────────────────────────────────
MODELS = {
    "Bonsai-1.7B": {
        "gguf_repo": "prism-ml/Bonsai-1.7B-gguf",
        "gguf_file": "Bonsai-1.7B.gguf",
        "mlx_repo": "prism-ml/Bonsai-1.7B-mlx-1bit",
        "params": "1.7B",
    },
    "Bonsai-4B": {
        "gguf_repo": "prism-ml/Bonsai-4B-gguf",
        "gguf_file": "Bonsai-4B.gguf",
        "mlx_repo": "prism-ml/Bonsai-4B-mlx-1bit",
        "params": "4B",
    },
    "Bonsai-8B": {
        "gguf_repo": "prism-ml/Bonsai-8B-gguf",
        "gguf_file": "Bonsai-8B.gguf",
        "mlx_repo": "prism-ml/Bonsai-8B-mlx-1bit",
        "params": "8B",
    },
}

LLAMA_RELEASE_TAG = "prism-b8194-1179bfc"
LLAMA_RELEASE_BASE = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}"


def get_cache_dir():
    """Return ~/.cache/bonsai-bench, creating it if needed."""
    cache = os.path.join(os.path.expanduser("~"), ".cache", "bonsai-bench")
    os.makedirs(cache, exist_ok=True)
    return cache


def get_models_dir():
    d = os.path.join(get_cache_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def get_bin_dir():
    d = os.path.join(get_cache_dir(), "bin")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_model_names(names):
    """Resolve model name(s) to list of registry keys.
    Accepts: 'all', '8B', 'Bonsai-8B', or comma-separated list."""
    if names is None or names.lower() == "all":
        return list(MODELS.keys())

    result = []
    for n in names.split(","):
        n = n.strip()
        if n in MODELS:
            result.append(n)
        else:
            # Try matching by size suffix
            for key in MODELS:
                if key.endswith(n):
                    result.append(key)
                    break
            else:
                print(f"WARNING: Unknown model '{n}'. Available: {', '.join(MODELS.keys())}")
    return result


def download_model(name):
    """Download a GGUF model from HuggingFace. Returns path to .gguf file."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")

    info = MODELS[name]
    models_dir = get_models_dir()
    gguf_path = os.path.join(models_dir, info["gguf_file"])

    if os.path.isfile(gguf_path):
        print(f"  {name}: already downloaded ({gguf_path})")
        return gguf_path

    print(f"  {name}: downloading from {info['gguf_repo']}...")
    snapshot_download(repo_id=info["gguf_repo"], local_dir=models_dir)
    print(f"  {name}: ready at {gguf_path}")
    return gguf_path


def get_model_path(name):
    """Get path to model file, downloading if needed."""
    info = MODELS[name]
    gguf_path = os.path.join(get_models_dir(), info["gguf_file"])
    if not os.path.isfile(gguf_path):
        download_model(name)
    return gguf_path


def download_llama_binary():
    """Download pre-built llama.cpp binary for current platform. Returns path to llama-cli."""
    bin_dir = get_bin_dir()
    system = platform.system()

    if system == "Darwin":
        asset = f"llama-{LLAMA_RELEASE_TAG}-bin-macos-arm64.tar.gz"
        cli_name = "llama-cli"
    elif system == "Linux":
        # Auto-detect CUDA version
        cuda_tag = "12.8"  # default
        try:
            nvcc_out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
            import re as _re
            m = _re.search(r"release (\d+)\.(\d+)", nvcc_out)
            if m:
                major, minor = int(m.group(1)), int(m.group(2))
                if major >= 13:
                    cuda_tag = "13.1"
                elif major == 12 and minor >= 8:
                    cuda_tag = "12.8"
                else:
                    cuda_tag = "12.4"
                print(f"  Detected CUDA {major}.{minor} -> using build for CUDA {cuda_tag}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"  CUDA version not detected, defaulting to CUDA {cuda_tag}")
        asset = f"llama-{LLAMA_RELEASE_TAG}-bin-linux-cuda-{cuda_tag}-x64.tar.gz"
        cli_name = "llama-cli"
    else:
        raise RuntimeError(f"Unsupported platform: {system}. Build llama.cpp from source.")

    cli_path = os.path.join(bin_dir, cli_name)
    bench_path = os.path.join(bin_dir, "llama-bench")

    if os.path.isfile(cli_path):
        return cli_path

    url = f"{LLAMA_RELEASE_BASE}/{asset}"
    print(f"  Downloading llama.cpp binaries from {url}...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(["curl", "-L", "--progress-bar", url, "-o", tmp_path], check=True)

    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(path=bin_dir)

    os.unlink(tmp_path)

    # Flatten: move all files from nested subdirectories to bin_dir
    for root, dirs, files in os.walk(bin_dir):
        if root == bin_dir:
            continue
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(bin_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Make executable + codesign on macOS
    for f in os.listdir(bin_dir):
        fp = os.path.join(bin_dir, f)
        if os.path.isfile(fp) and f.startswith("llama-"):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC)
            if system == "Darwin":
                subprocess.run(["xattr", "-cr", fp], capture_output=True)
                subprocess.run(["codesign", "-s", "-", "--force", "--timestamp=none", fp], capture_output=True)

    if not os.path.isfile(cli_path):
        raise RuntimeError(f"llama-cli not found after extraction at {cli_path}")

    print(f"  llama.cpp ready at {bin_dir}/")
    return cli_path


def get_llama_cli():
    """Get path to llama-cli, downloading if needed."""
    cli = os.path.join(get_bin_dir(), "llama-cli")
    if not os.path.isfile(cli):
        download_llama_binary()
    return cli


def get_llama_bench():
    """Get path to llama-bench, downloading if needed."""
    bench = os.path.join(get_bin_dir(), "llama-bench")
    if not os.path.isfile(bench):
        download_llama_binary()
    return bench
