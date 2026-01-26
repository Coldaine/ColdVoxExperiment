# ColdVox ASR Benchmark

Benchmarking local speech-to-text models on AMD Strix Halo hardware (no NVIDIA GPUs).

## Project Status

**Last Updated**: 2026-01-20

- ✅ Documentation and planning complete
- ✅ Models downloaded (~75GB from HuggingFace)
- ⏳ Environment setup (pending)
- ⏳ Benchmark implementation (pending)

## Quick Links

- **[Full Documentation](./docs/README.md)** - Start here
- **[Implementation Plan](./docs/plans/implementation-plan.md)** - Phases, PRs, workflow
- **[Model Insights](./docs/plans/model_insights_deep_dive.md)** - Gotchas and recommendations

## Models Ready for Benchmarking

All 7 models downloaded and cached locally:

| Model | Size | Type | Location |
|-------|------|------|----------|
| Distil-Whisper Large V3 | 12 GB | Baseline | `models/distil-whisper/files` |
| Granite Speech 2B | 11 GB | Primary | `models/granite-2b/files` |
| Granite Speech 8B | 17 GB | Primary | `models/granite-8b/files` |
| Voxtral Mini 3B | 18 GB | Primary | `models/voxtral/files` |
| Phi-4 Multimodal | 12 GB | Experimental | `models/phi4/files` |
| Whisper V3 Turbo (Q5_0) | 548 MB | Primary | `models/whisper-cpp/` |
| Moonshine Base ONNX | 4.3 GB | Primary | `models/moonshine/files` |

**Total**: ~75 GB

## Hardware

**Target Device**: ASUS ROG Flow Z13 (2025)
- CPU: AMD Ryzen AI MAX+ 395 (16 cores, 32 threads)
- GPU: AMD Radeon 8060S (integrated)
- RAM: 128 GB unified memory
- OS: Windows 11 Pro
- ROCm: 7.x (native Windows support)

## Next Steps

1. Run `python setup_env.py` to configure ROCm PyTorch
2. Add test audio to `snippets/` directory
3. Run `python benchmark.py` to evaluate all models
4. Review results in `results/results.csv`

See [Implementation Plan](./docs/plans/implementation-plan.md) for detailed instructions.
