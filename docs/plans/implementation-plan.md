# Local ASR Benchmark Implementation Plan

## Overview

Benchmark 7 speech-to-text models on AMD Strix Halo hardware to find the best accuracy/speed tradeoff for local inference.

**Hardware**: ASUS ROG Flow Z13 (Ryzen AI MAX+ 395, Radeon 8060S, 128GB RAM, Windows)

---

## Models to Benchmark

| Model | Size | Runtime | Device | Notes |
|-------|------|---------|--------|-------|
| Moonshine-base | 62M | ONNX | CPU/DirectML | Lightweight, edge-optimized |
| whisper-large-v3-turbo | 809M | whisper.cpp | CPU | Quantized (Q5_0), no ROCm bindings |
| Granite-3.3-2B | 2B | transformers | ROCm GPU | IBM, arbitrary-length audio |
| Granite-3.3-8B | 8B | transformers | ROCm GPU | Fits in 128GB unified memory |
| distil-large-v3 | 756M | transformers | ROCm GPU | 6x faster than Whisper |
| Phi-4-multimodal | 5.6B | transformers | CPU | **Experimental** - FlashAttn issues |
| Voxtral-Mini-3B | 3B | transformers | ROCm GPU | Mistral, July 2025 release |

---

## Technical Approach

### GPU Acceleration: ROCm 7.x

ROCm now supports Windows + Strix Halo (gfx1151). Use nightly builds:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
```

**Fallback chain**: ROCm → DirectML → CPU

### Dependencies

```
# requirements.txt

# Core (install PyTorch from ROCm index FIRST - see above)
transformers>=4.52.4
accelerate>=1.3.0
mistral_common

# Model-specific
useful-moonshine-onnx @ git+https://github.com/moonshine-ai/moonshine.git#subdirectory=moonshine-onnx
pywhispercpp>=1.4.1
peft

# Audio
librosa
soundfile

# Metrics
jiwer>=4.0.0
psutil
pandas

# Fallback
torch-directml
onnxruntime-directml
```

```
# requirements-dev.txt
pytest>=8.0
mypy
ruff
```

---

## Project Structure

```
├── pyproject.toml
├── setup_env.py
├── download_models.py
├── benchmark.py
├── requirements.txt
├── requirements-dev.txt
├── runners/
│   ├── __init__.py
│   ├── base.py
│   ├── moonshine.py
│   ├── whisper_cpp.py
│   ├── granite.py
│   ├── distil_whisper.py
│   ├── phi4.py
│   └── voxtral.py
├── tests/
│   ├── conftest.py
│   ├── test_base.py
│   ├── test_benchmark.py
│   └── test_*.py (per runner)
├── docs/
├── models/          # Downloaded weights (gitignored)
├── snippets/        # Test audio + transcripts
└── results/         # Output CSVs
```

---

## Development Workflow

### TDD Approach
1. Write tests first
2. Red → Green → Refactor
3. Each runner has unit tests before implementation

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

Create:
- `pyproject.toml`, `setup_env.py`, requirements
- `runners/base.py` (abstract interface)
- `benchmark.py` (main harness)
- `tests/` framework
- `.github/workflows/ci.yml`

Tests:
```python
def test_runner_interface(): ...
def test_transcription_result_dataclass(): ...
def test_benchmark_discovers_snippets(): ...
def test_benchmark_calculates_wer(): ...
def test_benchmark_handles_runner_failure(): ...
```

### PR #2: Moonshine
**Branch**: `feature/runner-moonshine`

- ONNX-based, simplest implementation
- Uses `moonshine_onnx` package

### PR #3: whisper.cpp
**Branch**: `feature/runner-whisper-cpp`

- Uses `pywhispercpp>=1.4.1`
- Download Q5_0 quantized model
- Optional: build with HIPBLAS for 7x speedup

### PR #4: Distil-Whisper
**Branch**: `feature/runner-distil-whisper`

- Use transformers pipeline (NOT faster-whisper)
- CTranslate2 is CUDA-only, avoid

### PR #5: Granite
**Branch**: `feature/runner-granite`

- Implement 2B first, then 8B
- Requires `transformers>=4.52.4`
- Use `torch.bfloat16`

### PR #6: Voxtral
**Branch**: `feature/runner-voxtral`

- `VoxtralForConditionalGeneration`
- May need transformers from git

### PR #7: Phi-4 (Experimental)
**Branch**: `feature/runner-phi4`

- FlashAttention not available on ROCm
- Use `attn_implementation='eager'`
- Allow graceful skip if fails

### PR #8: Integration
**Branch**: `feature/integration`

- End-to-end tests with real audio
- CI/CD finalization
- Documentation review

---

## Output Format

### results.csv
| Model | Snippet | Transcript | Reference | WER | Time_Sec | RTF | Peak_RAM_MB | Load_Time_Sec | Error |

### Metrics
- **WER**: Word Error Rate (0.0 = perfect)
- **RTF**: Real-Time Factor (< 1.0 = faster than real-time)

### Test Data Structure
```
snippets/
├── sample1.wav
├── sample1.txt    # Reference transcript
├── sample2.wav
├── sample2.txt
└── ...
```

---

## Verification

1. **Environment**: `python setup_env.py` → ROCm detected
2. **Models**: `python download_models.py` → all weights cached
3. **Benchmark**: `python benchmark.py` → results.csv generated
4. **Validate**: 7 models in results (or error entries), reasonable RTF values

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ROCm nightly instability | Fallback to DirectML, then CPU |
| Phi-4 FlashAttn dependency | Use eager attention or skip |
| Voxtral import issues | Update transformers from git |
| Granite-8B OOM | 128GB unified memory should suffice; use float16 |

---

## Alternative Models

If primary models fail, consider:

| Model | Notes |
|-------|-------|
| NVIDIA Canary-Qwen-2.5B | Best WER (5.63%), but CUDA-only |
| Meta Omnilingual ASR | 1600+ languages |
| GLM-ASR-Nano-2512 | 1.5B, noise-robust |
| OLMoASR (AI2) | Fully open training |
