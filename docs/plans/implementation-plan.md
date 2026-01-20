# Local ASR Benchmark Implementation Plan

## Overview

Benchmark 7 speech-to-text models on AMD Strix Halo hardware to find the best accuracy/speed tradeoff for local inference.

**Hardware**: ASUS ROG Flow Z13 (Ryzen AI MAX+ 395, Radeon 8060S, 128GB RAM, Windows)

**Related Docs**:
- [Architecture & Code Design](./architecture.md) - Interfaces, TDD specs, code examples
- [Environment Verification](./environment_verification_report.md) - ROCm setup, corrected code snippets
- [Model Insights Deep Dive](./model_insights_deep_dive.md) - Gotchas, hallucination risks, practical advice

---

## Quick Start (Execution Flow)

```bash
# 1. One-time setup
python setup_env.py

# 2. Download model weights
python download_models.py

# 3. Add test audio
cp your_audio.wav snippets/
cp your_transcript.txt snippets/   # Same basename

# 4. Run benchmark
python benchmark.py

# 5. Review results
cat results/results.csv
```

---

## Models to Benchmark

Ordered by **implementation priority** (based on stability research):

| Priority | Model | Size | Runtime | Risk Level | Notes |
|----------|-------|------|---------|------------|-------|
| 1 | **distil-large-v3** | 756M | transformers | Low | More stable than Whisper V3, use as baseline |
| 2 | Moonshine-base | 62M | ONNX | Low | English-only, fast, some hallucination on silence |
| 3 | whisper-large-v3-turbo | 809M | whisper.cpp | Medium | Hallucination risk on long audio |
| 4 | Granite-3.3-2B | 2B | transformers | Medium | Two-pass architecture |
| 5 | Granite-3.3-8B | 8B | transformers | Medium | Same as 2B, needs more RAM |
| 6 | Voxtral-Mini-3B | 3B | transformers | **High** | Known bugs, "integration hell" |
| 7 | Phi-4-multimodal | 5.6B | transformers | **High** | 40s limit, no timestamps, experimental |

---

## Technical Approach

### GPU Acceleration: ROCm 7.x

ROCm supports Windows + Strix Halo (gfx1151). Use nightly builds:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
```

**Verify installation:**
```python
import torch
print(f"ROCm Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"HIP Version: {torch.version.hip}")
```

**Fallback chain**: ROCm → DirectML → CPU

**Important**: Avoid WSL2. Use native Windows to access full 128GB unified memory.

### Sources
- [PyTorch ROCm 7 on Strix Halo](https://medium.com/@GenerationAI/pytorch-with-rocm-7-for-windows-on-amd-ryzen-ai-max-395-strix-halo-radeon-8060s-gfx1151-1ba069edc2c4)
- [ROCm 6.4.4 Windows Support](https://wccftech.com/amd-rocm-6-4-4-pytorch-support-windows-radeon-9000-radeon-7000-gpus-ryzen-ai-apus/)
- [ROCm 7.2.2 at CES 2026](https://videocardz.com/newz/amd-highlights-rocm-7-2-2-at-ces-2026-with-ryzen-ai-400-support-and-a-single-windows-plus-linux-release)

---

## Development Workflow

### TDD Approach
1. **Write tests FIRST** - No implementation without failing tests
2. Red → Green → Refactor
3. Each runner must have unit tests before implementation
4. Integration tests validate end-to-end flow

### Branch Strategy
- `main` - stable, docs and scaffolding
- `feature/scaffolding` - base classes, test framework
- `feature/runner-{model}` - one branch per model

### PR Rules
- **One PR per model runner**
- PRs sit **24 hours minimum** for async review
- Must pass: pytest, mypy, ruff
- Merge only after approval

---

## Implementation Phases

### PR #1: Scaffolding
**Branch**: `feature/scaffolding`

Create project foundation with TDD infrastructure. See [architecture.md](./architecture.md) for detailed specs.

- `pyproject.toml`, `setup_env.py`, requirements
- `runners/base.py` - Abstract BaseRunner interface
- `benchmark.py` - Main harness
- `tests/` - pytest framework with fixtures
- `.github/workflows/ci.yml`

### PR #2: Distil-Whisper (Baseline)
**Branch**: `feature/runner-distil-whisper`

Start here - most stable model, establishes the transformers pattern.
- Use transformers pipeline (NOT faster-whisper, which is CUDA-only)
- This becomes the accuracy baseline for comparing other models

### PR #3: Moonshine
**Branch**: `feature/runner-moonshine`

- Use `transformers` with `trust_remote_code=True` (see environment_verification_report.md)
- English-only, watch for hallucinations on short/silent segments

### PR #4: whisper.cpp
**Branch**: `feature/runner-whisper-cpp`

- `pywhispercpp>=1.4.1`
- Download Q5_0 quantized model (sweet spot for speed/accuracy)
- Set `condition_on_previous_text=False` to reduce hallucinations
- Optional: build with HIPBLAS for 7x speedup

### PR #5: Granite
**Branch**: `feature/runner-granite`

- Implement 2B first, then 8B
- Two-pass architecture - be aware of this
- Use `torch.bfloat16`, requires `transformers>=4.52.4`
- Add silence padding at end of audio to avoid EOF hallucinations

### PR #6: Voxtral
**Branch**: `feature/runner-voxtral`

- **High risk** - known bugs with tokenizer and HF pipelines
- May need custom inference code (standard `pipeline()` may fail)
- `VoxtralForConditionalGeneration`, may need transformers from git

### PR #7: Phi-4 (Experimental)
**Branch**: `feature/runner-phi4`

- 40 second audio limit, no timestamps
- FlashAttention not available on ROCm - use `attn_implementation='eager'`
- See environment_verification_report.md for corrected code
- Allow graceful skip if fails

### PR #8: Integration
**Branch**: `feature/integration`

- End-to-end tests with real audio
- CI/CD finalization
- Final documentation review

---

## Output Format

### results.csv
| Model | Snippet | Transcript | Reference | WER | Time_Sec | RTF | Peak_RAM_MB | Load_Time_Sec | Error |

### Metrics
- **WER**: Word Error Rate (0.0 = perfect, <10% = good, >20% = poor)
- **RTF**: Real-Time Factor (<1.0 = faster than real-time)

---

## Verification Checklist

1. **Environment**: `python setup_env.py` → ROCm detected, all imports work
2. **Models**: `python download_models.py` → all weights cached in `./models/`
3. **Benchmark**: `python benchmark.py` → results.csv generated
4. **Validate**:
   - All 7 models in results (or error entries for failures)
   - WER calculated correctly against reference transcripts
   - RTF values reasonable (<10 for all models)
   - RAM returns to baseline after each model (no leaks)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ROCm nightly instability | Fallback to DirectML, then CPU |
| Whisper V3 hallucinations | Use Distil-Whisper as primary; set `condition_on_previous_text=False` |
| Phi-4 FlashAttn dependency | Use `attn_implementation='eager'` or skip |
| Voxtral integration bugs | Allocate extra debug time; write custom inference if needed |
| Granite EOF hallucination | Add silence padding to audio |
| Moonshine hallucination | English only; avoid short/silent segments |

---

## Alternative Models

If primary models fail, consider:

| Model | Notes |
|-------|-------|
| NVIDIA Canary-Qwen-2.5B | Best WER (5.63%), but CUDA-only |
| Meta Omnilingual ASR | 1600+ languages |
| GLM-ASR-Nano-2512 | 1.5B, noise-robust |
| OLMoASR (AI2) | Fully open training pipeline |
