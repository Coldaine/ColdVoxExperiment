# Documentation Index

## Overview
This project benchmarks local speech-to-text (ASR) models on AMD hardware without NVIDIA GPUs.

## Quick Links

### Project Vision
- [North Star](./northstar.md) - Why this project exists and what we're trying to achieve

### Implementation
- [Implementation Plan](./plans/implementation-plan.md) - Full technical plan with corrections and details

### Development
- [Testing Guide](./dev/testing.md) - How to run benchmarks, add models, interpret results

### Model Documentation
Each model family has its own documentation:

| Model | Docs | Status |
|-------|------|--------|
| Moonshine | [docs/models/moonshine/](./models/moonshine/README.md) | Primary |
| Whisper (whisper.cpp) | [docs/models/whisper/](./models/whisper/README.md) | Primary |
| IBM Granite | [docs/models/granite/](./models/granite/README.md) | Primary |
| Distil-Whisper | [docs/models/distil-whisper/](./models/distil-whisper/README.md) | Primary |
| Phi-4 Multimodal | [docs/models/phi4/](./models/phi4/README.md) | Experimental |
| Voxtral | [docs/models/voxtral/](./models/voxtral/README.md) | Primary |

## Getting Started

```bash
# 1. Setup environment (installs ROCm PyTorch + dependencies)
python setup_env.py

# 2. Download all model weights
python download_models.py

# 3. Add your test audio files
cp your_audio.wav snippets/
cp your_transcript.txt snippets/  # Same name, .txt extension

# 4. Run benchmark
python benchmark.py

# 5. Check results
cat results/results.csv
```

## Hardware Requirements

- **Recommended**: AMD Ryzen AI MAX+ (Strix Halo) with 64GB+ RAM
- **Minimum**: Any x86_64 CPU with 32GB RAM (CPU-only mode)
- **GPU**: AMD Radeon with ROCm support (optional, improves speed)
