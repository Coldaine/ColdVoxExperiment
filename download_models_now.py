#!/usr/bin/env python3
"""
Download ASR models to local cache
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, login

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN")

# Login to HuggingFace
print("Logging in to HuggingFace...")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("HF_TOKEN not set, skipping login. Some models may be inaccessible.")

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Models to download (transformers-based)
MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "distil-whisper/distil-large-v3",
        "name": "Distil-Whisper Large V3",
        "cache_dir": models_dir / "distil-whisper",
    },
    {
        "repo_id": "ibm-granite/granite-speech-3.3-2b",
        "name": "Granite Speech 2B",
        "cache_dir": models_dir / "granite-2b",
    },
    {
        "repo_id": "ibm-granite/granite-speech-3.3-8b",
        "name": "Granite Speech 8B",
        "cache_dir": models_dir / "granite-8b",
    },
    {
        "repo_id": "mistralai/Voxtral-Mini-3B-2507",
        "name": "Voxtral Mini 3B",
        "cache_dir": models_dir / "voxtral",
    },
    {
        "repo_id": "microsoft/Phi-4-multimodal-instruct",
        "name": "Phi-4 Multimodal (Experimental)",
        "cache_dir": models_dir / "phi4",
    },
]

# Whisper.cpp quantized models
WHISPER_CPP_MODELS = [
    {
        "repo_id": "ggerganov/whisper.cpp",
        "filename": "ggml-large-v3-turbo-q5_0.bin",
        "name": "Whisper Large V3 Turbo Q5_0",
        "cache_dir": models_dir / "whisper-cpp",
    }
]

# Moonshine ONNX models
MOONSHINE_MODELS = [
    {
        "repo_id": "UsefulSensors/moonshine",
        "name": "Moonshine Base ONNX",
        "cache_dir": models_dir / "moonshine",
    }
]

def download_model(repo_id, name, cache_dir, allow_patterns=None):
    """Download a model from HuggingFace"""
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Repository: {repo_id}")
        print(f"Cache dir: {cache_dir}")
        print(f"{'='*60}")

        cache_dir.mkdir(parents=True, exist_ok=True)

        result = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=cache_dir / "files",
            token=HF_TOKEN,
            allow_patterns=allow_patterns,
        )

        print(f"[OK] Successfully downloaded {name} to {result}")
        return True

    except Exception as e:
        print(f"[FAIL] Failed to download {name}: {e}")
        return False

def download_single_file(repo_id, filename, name, cache_dir):
    """Download a single file from HuggingFace"""
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Repository: {repo_id}")
        print(f"File: {filename}")
        print(f"Cache dir: {cache_dir}")
        print(f"{'='*60}")

        cache_dir.mkdir(parents=True, exist_ok=True)

        result = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=cache_dir,
            token=HF_TOKEN,
        )

        print(f"[OK] Successfully downloaded {name} to {result}")
        return True

    except Exception as e:
        print(f"[FAIL] Failed to download {name}: {e}")
        return False

def main():
    print("="*60)
    print("ASR Models Download Script")
    print("="*60)

    success_count = 0
    fail_count = 0

    # Download transformers-based models
    print("\n## Downloading Transformers-based Models ##\n")
    for model in MODELS_TO_DOWNLOAD:
        if download_model(**model):
            success_count += 1
        else:
            fail_count += 1

    # Download Whisper.cpp quantized models
    print("\n## Downloading Whisper.cpp Quantized Models ##\n")
    for model in WHISPER_CPP_MODELS:
        if download_single_file(**model):
            success_count += 1
        else:
            fail_count += 1

    # Download Moonshine ONNX models
    print("\n## Downloading Moonshine ONNX Models ##\n")
    for model in MOONSHINE_MODELS:
        if download_model(**model):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"[OK] Successful: {success_count}")
    print(f"[FAIL] Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")
    print("="*60)

    return fail_count == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
