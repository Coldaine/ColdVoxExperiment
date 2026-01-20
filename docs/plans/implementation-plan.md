# AMD Z13 Local ASR Benchmark Plan - CORRECTED

## Overview
Build a Python-based benchmarking harness to evaluate 7 local speech-to-text models on an AMD Z13 (Ryzen AI MAX+ 395, 128GB RAM, no NVIDIA GPU).

---

## Critical Corrections to Original Plan

### 1. System Specs
- **Issue**: Original says "64GB RAM" but your system has **128GB RAM**
- **Impact**: The 8B Granite model is more feasible than expected

### 2. Model Corrections

| Model | Original Assumption | Correction |
|-------|---------------------|------------|
| **Moonshine** | "Use claude-stt logic" | Use official `moonshine` package from useful-sensors (pip install useful-sensors-moonshine) or direct ONNX inference |
| **whisper.cpp** | `pywhispercpp` or `whisper_cpp_python` | Use `pywhispercpp` (more maintained) or the newer `whisper-cpp-python` package. Note: quantized models (Q5_0) require manual download from ggerganov/whisper.cpp releases |
| **faster-whisper** | "CTranslate2 CPU mode" | CTranslate2 CPU mode works but is **slower than transformers** on CPU. Consider using `transformers` directly for distil-whisper instead |
| **Phi-4 multimodal** | "Audio transcription mode" | Phi-4-multimodal requires **specific audio preprocessing** (speech_encoder). This is NOT a drop-in ASR model - it's a multimodal LLM. May need significant custom code |
| **Voxtral-Mini** | "transformers or vLLM" | Very new model (released mid-2025). Verify HuggingFace availability. **Fallback**: Use `moshi` from Kyutai if Voxtral unavailable |

### 3. Runtime Recommendations (ROCm-first)

| Model | Recommended Runtime | Device | Why |
|-------|---------------------|--------|-----|
| Moonshine-base | ONNX Runtime | ROCm or DirectML | Lightweight, ONNX optimized |
| whisper-large-v3-turbo | whisper.cpp | CPU | Optimized C++ with quantization, no ROCm bindings yet |
| Granite-3.3-2B | transformers + ROCm | GPU (gfx1151) | Native PyTorch ROCm support |
| Granite-3.3-8B | transformers + ROCm | GPU | 128GB unified memory = can fit full model! |
| distil-large-v3 | transformers + ROCm | GPU | Native PyTorch ROCm support |
| Phi-4-multimodal | transformers | CPU (experimental) | Complex pipeline, skip gracefully if fails |
| Voxtral-Mini-3B | transformers + ROCm | GPU | Standard HF pipeline |

### 4. GPU Acceleration: ROCm on Windows (UPDATED Jan 2026)

**ROCm NOW supports Windows + Strix Halo!**

| ROCm Version | Status | Strix Halo Support |
|--------------|--------|-------------------|
| **ROCm 6.4.4** | Released | Windows PyTorch for Ryzen AI MAX "Strix Halo" APUs |
| **ROCm 7.2.2** | CES 2026 | Single Windows + Linux release, WinML readiness |
| **ROCm 7 nightlies** | Available | gfx1151 (Radeon 8060S) builds at `rocm.nightlies.amd.com` |

**Why Strix Halo is uniquely powerful for this:**
- Wide memory interface to **128GB system RAM** (not limited to small VRAM)
- Can run models that wouldn't fit in typical discrete GPU VRAM
- fp32 and fp64 support

**Recommended Acceleration Strategy:**
1. **Primary**: ROCm + PyTorch (native AMD stack)
2. **Fallback**: DirectML via `onnxruntime-directml` (if ROCm has issues)
3. **Fallback**: CPU (always works)

**Installation (ROCm 7 on Windows):**
```bash
# Via conda + pip with nightly builds
pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

Sources:
- [PyTorch with ROCm 7 for Windows on Strix Halo](https://medium.com/@GenerationAI/pytorch-with-rocm-7-for-windows-on-amd-ryzen-ai-max-395-strix-halo-radeon-8060s-gfx1151-1ba069edc2c4)
- [ROCm 6.4.4: AMD's Official PyTorch for Windows](https://wccftech.com/amd-rocm-6-4-4-pytorch-support-windows-radeon-9000-radeon-7000-gpus-ryzen-ai-apus/)
- [AMD highlights ROCm 7.2.2 at CES 2026](https://videocardz.com/newz/amd-highlights-rocm-7-2-2-at-ces-2026-with-ryzen-ai-400-support-and-a-single-windows-plus-linux-release)

---

## Revised Deliverables

### File 1: `requirements.txt`
```
# Core (installed via ROCm index for gfx1151)
# torch  -- installed separately via: pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
# torchaudio -- installed separately via ROCm index
transformers>=4.40
accelerate

# Fallback AMD support (if ROCm has issues)
torch-directml
onnxruntime-directml

# ASR-specific
openai-whisper
pywhispercpp
librosa
soundfile

# Metrics
jiwer           # Word Error Rate calculation
psutil          # RAM monitoring
pandas          # Results CSV handling

# Model-specific
sentencepiece
protobuf
```

**Note**: PyTorch must be installed separately from the ROCm nightly index for gfx1151 support.

### File 2: `setup_env.py`
- Create venv
- Install PyTorch from ROCm nightly index (gfx1151)
- Install remaining packages from requirements.txt
- Attempt DirectML installation as fallback (graceful failure)
- Verify imports and detect available acceleration (ROCm > DirectML > CPU)

### File 3: `download_models.py`
- Pre-download all model weights to `./models/` cache
- Handle whisper.cpp GGUF files separately (direct URL download)
- Verify checksums where available

### File 4: `benchmark.py`
Main `ASRBench` class with:

```python
class ASRBench:
    def __init__(self, snippets_dir: Path, models_cache: Path):
        self.models = [
            MoonshineRunner(),      # ONNX
            WhisperCppRunner(),     # whisper.cpp
            GraniteRunner("2b"),    # transformers
            GraniteRunner("8b"),    # transformers + aggressive gc
            DistilWhisperRunner(),  # transformers
            Phi4Runner(),           # transformers multimodal
            VoxtralRunner(),        # transformers
        ]

    def run(self) -> pd.DataFrame:
        results = []
        for model in self.models:
            try:
                model.load()
                model.warmup()
                for snippet in self.snippets:
                    metrics = model.transcribe(snippet)
                    results.append(metrics)
            except Exception as e:
                results.append(error_row(model, e))
            finally:
                model.unload()
                gc.collect()
        return pd.DataFrame(results)
```

### File 5: `runners/` module
Individual runner classes per model family:
- `runners/moonshine.py`
- `runners/whisper_cpp.py`
- `runners/granite.py`
- `runners/distil_whisper.py`
- `runners/phi4.py`
- `runners/voxtral.py`

---

## Output Format

### `results.csv`
| Model | Snippet | Transcript | Reference | WER | Time_Sec | RTF | Peak_RAM_MB | Load_Time_Sec | Error |
|-------|---------|------------|-----------|-----|----------|-----|-------------|---------------|-------|

### Metrics Explained
- **WER (Word Error Rate)**: `(substitutions + insertions + deletions) / reference_words` - lower is better (0.0 = perfect)
- **RTF (Real-Time Factor)**: `processing_time / audio_duration`
  - RTF < 1.0 = faster than real-time
  - RTF = 1.0 = real-time
  - RTF > 1.0 = slower than real-time

### Snippet File Structure
```
snippets/
├── sample1.wav
├── sample1.txt      # Reference transcript for WER
├── sample2.wav
├── sample2.txt
└── ...
```

---

## Execution Flow

```
1. setup_env.py          # One-time setup
2. download_models.py    # Pre-fetch all weights
3. Place .wav files in ./snippets/
4. python benchmark.py   # Run benchmark
5. Review results.csv
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ROCm nightly instability | Fallback to DirectML, then CPU |
| Phi-4 complexity | Marked as "experimental", skip gracefully if audio encoder unavailable |
| Voxtral unavailability | Fallback to Kyutai Moshi or skip |
| OOM on Granite-8B | Leverage 128GB unified memory; use `torch.float16` if needed |

---

## Project Structure

```
asr-benchmark/
├── setup_env.py
├── download_models.py
├── benchmark.py
├── requirements.txt
├── runners/
│   ├── __init__.py
│   ├── base.py
│   ├── moonshine.py
│   ├── whisper_cpp.py
│   ├── granite.py
│   ├── distil_whisper.py
│   ├── phi4.py
│   └── voxtral.py
├── docs/
│   ├── northstar.md             # Project vision and goals
│   ├── dev/
│   │   └── testing.md           # Testing guide, how to run benchmarks, add new models
│   ├── models/
│   │   ├── moonshine/
│   │   │   └── README.md        # Moonshine setup, quirks, performance notes
│   │   ├── whisper/
│   │   │   └── README.md        # whisper.cpp, quantization, GGUF formats
│   │   ├── granite/
│   │   │   └── README.md        # IBM Granite 2B/8B, memory requirements
│   │   ├── distil-whisper/
│   │   │   └── README.md        # Distil-whisper specifics
│   │   ├── phi4/
│   │   │   └── README.md        # Phi-4 multimodal audio handling
│   │   └── voxtral/
│   │       └── README.md        # Voxtral/Mistral setup
│   └── README.md                # Documentation index
├── models/              # Downloaded weights cache
├── snippets/            # Input .wav files
└── results/             # Output CSVs
```

---

## Verification Plan

### Step 1: Environment Setup
```bash
python setup_env.py
# Expected: venv created, ROCm PyTorch installed, imports verified
```

### Step 2: Model Download
```bash
python download_models.py
# Expected: All 7 models downloaded to ./models/, progress shown
```

### Step 3: Prepare Test Data
```
snippets/
├── test1.wav    # Short clip (~5s)
├── test1.txt    # Reference: "Hello world this is a test"
├── test2.wav    # Medium clip (~30s)
├── test2.txt
└── test3.wav    # Long clip (~2min)
    test3.txt
```

### Step 4: Run Benchmark
```bash
python benchmark.py
# Expected: Each model loads, transcribes, unloads
# Progress bar shown, results.csv generated
```

### Step 5: Validate Results
- Check `results/results.csv` contains all 7 models (or error entries for failed ones)
- Verify WER calculated correctly against reference transcripts
- Confirm RTF values are reasonable (< 10 for all models)
- Check no memory leaks (RAM returns to baseline after each model)

---

## Phased Implementation (TDD + One PR Per Model)

### Development Philosophy

**Test-Driven Development (TDD)**:
- Write tests FIRST for each component
- Red → Green → Refactor cycle
- Each runner must have unit tests before implementation
- Integration tests validate end-to-end flow

**Branch Strategy**:
- `main` - stable, documentation and scaffolding only initially
- `feature/scaffolding` - project structure, base classes, test framework
- `feature/runner-moonshine` - Moonshine implementation
- `feature/runner-whisper-cpp` - whisper.cpp implementation
- `feature/runner-granite` - Granite 2B/8B implementation
- `feature/runner-distil-whisper` - Distil-Whisper implementation
- `feature/runner-phi4` - Phi-4 (experimental)
- `feature/runner-voxtral` - Voxtral implementation

**PR Workflow**:
- One PR per model runner
- PRs should sit for review before merge (allows async review)
- Each PR must pass: unit tests, type checks, linting
- Merge to `main` only after approval

---

### Phase 1: Scaffolding (Branch: `feature/scaffolding`)

**PR #1: Project Foundation**

Files to create:
```
├── pyproject.toml           # Modern Python packaging
├── setup_env.py
├── requirements.txt
├── requirements-dev.txt     # pytest, mypy, ruff
├── runners/
│   ├── __init__.py
│   └── base.py             # Abstract BaseRunner
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # pytest fixtures
│   └── test_base.py        # BaseRunner tests
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions
```

Tests first:
```python
# tests/test_base.py
def test_runner_interface():
    """BaseRunner must define load, transcribe, unload."""

def test_transcription_result_dataclass():
    """TranscriptionResult has required fields."""

def test_runner_context_manager():
    """Runners work as context managers for cleanup."""
```

---

### Phase 2: Benchmark Harness (Same PR as Phase 1)

Files:
```
├── benchmark.py
├── tests/test_benchmark.py
```

Tests first:
```python
# tests/test_benchmark.py
def test_benchmark_discovers_snippets():
    """ASRBench finds .wav files with matching .txt."""

def test_benchmark_calculates_wer():
    """WER calculated correctly using jiwer."""

def test_benchmark_handles_runner_failure():
    """Failed runners logged, benchmark continues."""
```

---

### Phase 3-8: Model Runners (One PR Each)

Each model gets its own branch and PR. PRs should sit for review.

#### PR #2: Moonshine Runner (Branch: `feature/runner-moonshine`)

```python
# tests/test_moonshine.py
def test_moonshine_loads_onnx():
    """Moonshine loads ONNX model."""

def test_moonshine_transcribes_wav():
    """Moonshine transcribes test audio."""

def test_moonshine_unloads_cleanly():
    """Memory freed after unload."""
```

Implementation pattern (from research):
```python
# runners/moonshine.py
import moonshine_onnx

class MoonshineRunner(BaseRunner):
    name = "moonshine-base"

    def load(self):
        # Uses useful-moonshine-onnx package
        self.model = moonshine_onnx.load_model('moonshine/base')
```

---

#### PR #3: whisper.cpp Runner (Branch: `feature/runner-whisper-cpp`)

Tests + implementation for pywhispercpp (v1.4.1+)

**Note from research**: Consider building whisper.cpp with HIPBLAS for 7x speedup on AMD:
```bash
make clean && WHISPER_HIPBLAS=1 make -j
```

---

#### PR #4: Granite Runner (Branch: `feature/runner-granite`)

Tests + implementation for IBM Granite Speech 3.3 (2B and 8B)

**From research**:
- Requires `transformers>=4.52.4`
- Python 3.11 or 3.12 (numba issues with 3.14)
- Uses `torch.bfloat16` for optimal performance

---

#### PR #5: Distil-Whisper Runner (Branch: `feature/runner-distil-whisper`)

Tests + implementation using transformers (NOT faster-whisper)

**From research**:
- faster-whisper (CTranslate2) is CUDA-only, avoid for AMD
- Use HuggingFace transformers pipeline directly with ROCm

---

#### PR #6: Phi-4 Runner (Branch: `feature/runner-phi4`) - EXPERIMENTAL

Tests + implementation (may be skipped)

**From research**:
- FlashAttention required but not available for ROCm
- Workaround: `attn_implementation='eager'` (slower)
- Mark as experimental, allow graceful skip

---

#### PR #7: Voxtral Runner (Branch: `feature/runner-voxtral`)

Tests + implementation for Mistral Voxtral-Mini-3B-2507

**From research**:
- Released July 2025, fully available
- Requires `transformers` from git (bleeding edge)
- Uses `VoxtralForConditionalGeneration`

---

### Phase 9: Integration (Branch: `feature/integration`)

**PR #8: Full Integration**

- Integration tests with real audio
- CI/CD pipeline finalization
- Final documentation review

---

## PR Review Guidelines

Each PR must include:

1. **Tests passing** (pytest)
2. **Type hints** (mypy clean)
3. **Linting** (ruff)
4. **Documentation** updated if needed
5. **Model-specific README** updated with findings

PRs should sit open for at least **24 hours** for async review before merge.

---

## Updated Dependencies (From Research - Jan 2026)

```
# requirements.txt (UPDATED)

# Core - Install PyTorch FIRST from ROCm index:
# pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision

transformers>=4.52.4          # Required for Granite, Voxtral
accelerate>=1.3.0
mistral_common                # For Voxtral

# Moonshine
useful-moonshine-onnx @ git+https://github.com/moonshine-ai/moonshine.git#subdirectory=moonshine-onnx

# whisper.cpp
pywhispercpp>=1.4.1

# Audio processing
librosa
soundfile
peft                          # For Granite

# Metrics
jiwer>=4.0.0                  # Updated to v4 with RapidFuzz backend
psutil
pandas

# Fallback (if ROCm fails)
torch-directml
onnxruntime-directml
```

```
# requirements-dev.txt
pytest>=8.0
pytest-asyncio
mypy
ruff
```

---

## Alternative Models to Consider (From Research)

If primary models fail, consider:

| Model | WER | Notes |
|-------|-----|-------|
| NVIDIA Canary-Qwen-2.5B | 5.63% | #1 on leaderboard, but CUDA-only |
| Meta Omnilingual ASR | - | 1600+ languages, Nov 2025 |
| GLM-ASR-Nano-2512 | - | 1.5B, noise-robust, Dec 2025 |
| OLMoASR (AI2) | - | Fully open training pipeline |

---

## Implementation Order

1. **Phase 1**: Project scaffolding + `setup_env.py` + base classes (PR #1)
2. **Phase 2**: Benchmark harness with TDD (same PR)
3. **Phase 3**: Moonshine runner (PR #2) - easiest, ONNX
4. **Phase 4**: whisper.cpp runner (PR #3) - well-documented
5. **Phase 5**: Distil-Whisper runner (PR #4) - transformers pattern
6. **Phase 6**: Granite runner (PR #5) - 2B first, then 8B
7. **Phase 7**: Voxtral runner (PR #6) - newest model
8. **Phase 8**: Phi-4 runner (PR #7) - experimental, may skip
9. **Phase 9**: Integration + final testing (PR #8)
