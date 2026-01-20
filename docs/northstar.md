# North Star: Local ASR Benchmark

## Why This Exists

The goal of this repository is to **empirically test which speech-to-text models can actually run on consumer AMD hardware** (specifically the Ryzen AI MAX+ 395 with 128GB RAM, no NVIDIA GPU).

The AI/ML ecosystem is heavily optimized for NVIDIA CUDA. This project explores what's actually possible on AMD hardware using ROCm, DirectML, and CPU fallbacks.

## The Question We're Answering

> "Which open-source ASR model gives the best accuracy-to-speed tradeoff on AMD/CPU hardware for real-world speech samples?"

## What We're Testing

### Models (7 candidates)
1. **useful-sensors/moonshine-base** - Lightweight ONNX model
2. **openai/whisper-large-v3-turbo** - Via whisper.cpp with quantization
3. **ibm-granite/granite-speech-3.3-2b** - IBM's speech model (small)
4. **ibm-granite/granite-speech-3.3-8b** - IBM's speech model (large)
5. **distil-whisper/distil-large-v3** - Distilled Whisper
6. **microsoft/Phi-4-multimodal-instruct** - Multimodal LLM (experimental)
7. **mistralai/Voxtral-Mini-3B-2507** - Mistral's speech model

### Hardware
- **CPU**: AMD Ryzen AI MAX+ 395 (16 cores / 32 threads)
- **GPU**: AMD Radeon 8060S (integrated, gfx1151)
- **RAM**: 128GB unified memory
- **Acceleration**: ROCm 6.4.4+ / ROCm 7 nightlies

### Metrics
- **WER (Word Error Rate)**: Transcription accuracy against reference
- **RTF (Real-Time Factor)**: Speed relative to audio duration
- **Peak RAM**: Memory usage during inference
- **Load Time**: Model initialization time

## Success Criteria

1. **At least 3 models** run successfully on this hardware
2. **One model** achieves RTF < 0.5 (2x faster than real-time) with acceptable accuracy
3. **Clear winner** identified for production use in local STT pipeline

## Non-Goals

- **Cloud API comparisons** - This is local-only
- **Training or fine-tuning** - Inference benchmarking only
- **Windows vs Linux comparison** - Windows-first, Linux later if needed
- **Comprehensive WER benchmarking** - Focus on relative comparison, not absolute scores

## Why 128GB RAM Matters

The Strix Halo APU has a wide memory interface to system RAM. Unlike discrete GPUs with limited VRAM (8-24GB typically), this system can:
- Load the full 8B Granite model into GPU-accessible memory
- Avoid model sharding or quantization compromises
- Run models that would OOM on typical consumer GPUs

## Expected Outcome

A ranked list of models by usability on AMD hardware:

| Tier | Criteria |
|------|----------|
| **Tier 1** | RTF < 0.5, WER < 10%, stable |
| **Tier 2** | RTF < 1.0, WER < 15%, works |
| **Tier 3** | Runs but too slow for real-time |
| **Failed** | Doesn't load or crashes |

The winning model(s) will be integrated into the production STT pipeline.
