# Architecture & Code Design

This document defines the code structure, interfaces, and TDD specifications for the ASR benchmark project.

---

## Project Structure

```
asr-benchmark/
├── pyproject.toml           # Project metadata, dependencies
├── setup_env.py             # Environment setup script
├── download_models.py       # Model weight downloader
├── benchmark.py             # Main benchmark harness
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Dev dependencies (pytest, mypy, ruff)
│
├── runners/
│   ├── __init__.py          # Exports all runners
│   ├── base.py              # BaseRunner ABC, TranscriptionResult
│   ├── distil_whisper.py    # Distil-Whisper (baseline)
│   ├── moonshine.py         # Moonshine ONNX
│   ├── whisper_cpp.py       # whisper.cpp via pywhispercpp
│   ├── granite.py           # IBM Granite 2B/8B
│   ├── voxtral.py           # Mistral Voxtral
│   └── phi4.py              # Phi-4 multimodal (experimental)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── test_base.py         # BaseRunner interface tests
│   ├── test_benchmark.py    # Benchmark harness tests
│   ├── test_distil_whisper.py
│   ├── test_moonshine.py
│   ├── test_whisper_cpp.py
│   ├── test_granite.py
│   ├── test_voxtral.py
│   └── test_phi4.py
│
├── docs/                    # Documentation
├── models/                  # Downloaded weights (gitignored)
├── snippets/                # Test audio + transcripts
├── results/                 # Output CSVs
└── .github/workflows/ci.yml # GitHub Actions
```

---

## Dependencies

### requirements.txt
```
# Core (install PyTorch from ROCm index FIRST)
# pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision

transformers>=4.52.4
accelerate>=1.3.0
mistral_common

# Model-specific
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

### requirements-dev.txt
```
pytest>=8.0
pytest-cov
mypy
ruff
```

---

## Core Interfaces

### TranscriptionResult

```python
# runners/base.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TranscriptionResult:
    """Result from a single transcription."""
    transcript: str
    time_sec: float
    peak_ram_mb: float
    audio_duration_sec: float
    error: Optional[str] = None

    @property
    def rtf(self) -> float:
        """Real-Time Factor: processing_time / audio_duration."""
        if self.audio_duration_sec <= 0:
            return float('inf')
        return self.time_sec / self.audio_duration_sec
```

### BaseRunner (Abstract Base Class)

```python
# runners/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import gc

class BaseRunner(ABC):
    """Abstract base class for ASR model runners."""

    name: str  # e.g., "distil-whisper-large-v3"

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._model = None
        self._processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a single audio file."""
        pass

    def unload(self) -> None:
        """Free model from memory."""
        self._model = None
        self._processor = None
        gc.collect()
        # If using PyTorch with CUDA/ROCm:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
```

---

## Benchmark Harness

### ASRBench Class

```python
# benchmark.py
from pathlib import Path
from typing import List, Optional
import pandas as pd
import gc
from jiwer import wer

from runners.base import BaseRunner, TranscriptionResult
from runners import (
    DistilWhisperRunner,
    MoonshineRunner,
    WhisperCppRunner,
    GraniteRunner,
    VoxtralRunner,
    Phi4Runner,
)


class ASRBench:
    """Main benchmark harness for ASR models."""

    def __init__(
        self,
        snippets_dir: Path,
        models_dir: Path,
        results_dir: Path,
    ):
        self.snippets_dir = Path(snippets_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Runners in priority order (most stable first)
        self.runners: List[BaseRunner] = [
            DistilWhisperRunner(models_dir),
            MoonshineRunner(models_dir),
            WhisperCppRunner(models_dir),
            GraniteRunner(models_dir, variant="2b"),
            GraniteRunner(models_dir, variant="8b"),
            VoxtralRunner(models_dir),
            Phi4Runner(models_dir),
        ]

    def discover_snippets(self) -> List[tuple[Path, Path]]:
        """Find .wav files with matching .txt transcripts."""
        snippets = []
        for wav in self.snippets_dir.glob("*.wav"):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                snippets.append((wav, txt))
        return sorted(snippets)

    def run(self) -> pd.DataFrame:
        """Run all benchmarks and return results DataFrame."""
        snippets = self.discover_snippets()
        if not snippets:
            raise ValueError(f"No snippets found in {self.snippets_dir}")

        results = []

        for runner in self.runners:
            print(f"\n{'='*50}")
            print(f"Model: {runner.name}")
            print('='*50)

            try:
                # Measure load time
                import time
                load_start = time.perf_counter()
                runner.load()
                load_time = time.perf_counter() - load_start
                print(f"Loaded in {load_time:.2f}s")

                # Warmup (1 dry run)
                if snippets:
                    print("Warming up...")
                    _ = runner.transcribe(snippets[0][0])

                # Process all snippets
                for wav_path, txt_path in snippets:
                    reference = txt_path.read_text(encoding="utf-8").strip()

                    result = runner.transcribe(wav_path)

                    # Calculate WER
                    word_error_rate = wer(reference, result.transcript) if reference else None

                    results.append({
                        "Model": runner.name,
                        "Snippet": wav_path.name,
                        "Transcript": result.transcript,
                        "Reference": reference,
                        "WER": word_error_rate,
                        "Time_Sec": result.time_sec,
                        "RTF": result.rtf,
                        "Peak_RAM_MB": result.peak_ram_mb,
                        "Load_Time_Sec": load_time,
                        "Error": result.error,
                    })

                    print(f"  {wav_path.name}: RTF={result.rtf:.2f}, WER={word_error_rate:.2%}")

            except Exception as e:
                # Log error for this model, continue to next
                print(f"ERROR: {e}")
                for wav_path, txt_path in snippets:
                    results.append({
                        "Model": runner.name,
                        "Snippet": wav_path.name,
                        "Transcript": "",
                        "Reference": txt_path.read_text(encoding="utf-8").strip(),
                        "WER": None,
                        "Time_Sec": None,
                        "RTF": None,
                        "Peak_RAM_MB": None,
                        "Load_Time_Sec": None,
                        "Error": str(e),
                    })

            finally:
                runner.unload()
                gc.collect()

        df = pd.DataFrame(results)
        return df

    def save_results(self, df: pd.DataFrame, filename: str = "results.csv") -> Path:
        """Save results to CSV."""
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)
        return output_path


def main():
    bench = ASRBench(
        snippets_dir=Path("./snippets"),
        models_dir=Path("./models"),
        results_dir=Path("./results"),
    )
    df = bench.run()
    output = bench.save_results(df)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
```

---

## TDD Test Specifications

### tests/conftest.py (Shared Fixtures)

```python
import pytest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio(temp_dir) -> Path:
    """Create a simple test audio file (1 second of silence)."""
    audio_path = temp_dir / "test.wav"
    # 1 second of silence at 16kHz
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(audio_path, samples, 16000)
    return audio_path


@pytest.fixture
def sample_snippet(temp_dir, sample_audio) -> tuple[Path, Path]:
    """Create a test snippet with audio and transcript."""
    txt_path = sample_audio.with_suffix(".txt")
    txt_path.write_text("test transcript")
    return sample_audio, txt_path


@pytest.fixture
def snippets_dir(temp_dir, sample_snippet) -> Path:
    """Create a snippets directory with test files."""
    snippets = temp_dir / "snippets"
    snippets.mkdir()
    wav, txt = sample_snippet
    (snippets / wav.name).write_bytes(wav.read_bytes())
    (snippets / txt.name).write_text(txt.read_text())
    return snippets
```

### tests/test_base.py

```python
"""Tests for BaseRunner interface - write BEFORE implementation."""
import pytest
from pathlib import Path
from runners.base import BaseRunner, TranscriptionResult


class TestTranscriptionResult:
    """TranscriptionResult dataclass tests."""

    def test_has_required_fields(self):
        """TranscriptionResult must have transcript, time_sec, peak_ram_mb, audio_duration_sec."""
        result = TranscriptionResult(
            transcript="hello world",
            time_sec=1.5,
            peak_ram_mb=100.0,
            audio_duration_sec=3.0,
        )
        assert result.transcript == "hello world"
        assert result.time_sec == 1.5
        assert result.peak_ram_mb == 100.0
        assert result.audio_duration_sec == 3.0

    def test_rtf_calculation(self):
        """RTF = time_sec / audio_duration_sec."""
        result = TranscriptionResult(
            transcript="test",
            time_sec=1.5,
            peak_ram_mb=100.0,
            audio_duration_sec=3.0,
        )
        assert result.rtf == 0.5  # 1.5 / 3.0

    def test_rtf_handles_zero_duration(self):
        """RTF should handle zero audio duration gracefully."""
        result = TranscriptionResult(
            transcript="test",
            time_sec=1.0,
            peak_ram_mb=100.0,
            audio_duration_sec=0.0,
        )
        assert result.rtf == float('inf')

    def test_error_field_optional(self):
        """Error field should be optional, defaulting to None."""
        result = TranscriptionResult(
            transcript="test",
            time_sec=1.0,
            peak_ram_mb=100.0,
            audio_duration_sec=1.0,
        )
        assert result.error is None


class TestBaseRunner:
    """BaseRunner ABC tests."""

    def test_is_abstract(self):
        """BaseRunner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRunner(Path("."))

    def test_subclass_requires_load(self):
        """Subclasses must implement load()."""
        class IncompleteRunner(BaseRunner):
            name = "incomplete"
            def transcribe(self, audio_path):
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Path("."))

    def test_subclass_requires_transcribe(self):
        """Subclasses must implement transcribe()."""
        class IncompleteRunner(BaseRunner):
            name = "incomplete"
            def load(self):
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Path("."))

    def test_context_manager_calls_load_and_unload(self, temp_dir):
        """Context manager should call load() on enter, unload() on exit."""
        load_called = False
        unload_called = False

        class TestRunner(BaseRunner):
            name = "test"

            def load(self):
                nonlocal load_called
                load_called = True

            def transcribe(self, audio_path):
                return TranscriptionResult("", 0, 0, 0)

            def unload(self):
                nonlocal unload_called
                unload_called = True

        with TestRunner(temp_dir):
            assert load_called
            assert not unload_called

        assert unload_called

    def test_is_loaded_property(self, temp_dir):
        """is_loaded should reflect model state."""
        class TestRunner(BaseRunner):
            name = "test"

            def load(self):
                self._model = "loaded"

            def transcribe(self, audio_path):
                return TranscriptionResult("", 0, 0, 0)

        runner = TestRunner(temp_dir)
        assert not runner.is_loaded
        runner.load()
        assert runner.is_loaded
        runner.unload()
        assert not runner.is_loaded
```

### tests/test_benchmark.py

```python
"""Tests for ASRBench harness - write BEFORE implementation."""
import pytest
from pathlib import Path
import pandas as pd
from benchmark import ASRBench
from runners.base import BaseRunner, TranscriptionResult


class MockRunner(BaseRunner):
    """Mock runner for testing."""
    name = "mock-runner"

    def load(self):
        self._model = "loaded"

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            transcript="mock transcript",
            time_sec=0.5,
            peak_ram_mb=100.0,
            audio_duration_sec=1.0,
        )


class FailingRunner(BaseRunner):
    """Runner that fails on load."""
    name = "failing-runner"

    def load(self):
        raise RuntimeError("Simulated load failure")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        raise RuntimeError("Should not be called")


class TestASRBench:
    """ASRBench harness tests."""

    def test_discover_snippets_finds_wav_with_txt(self, snippets_dir):
        """discover_snippets() finds .wav files with matching .txt."""
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=Path("."),
            results_dir=Path("."),
        )
        snippets = bench.discover_snippets()
        assert len(snippets) >= 1
        wav, txt = snippets[0]
        assert wav.suffix == ".wav"
        assert txt.suffix == ".txt"
        assert wav.stem == txt.stem

    def test_discover_snippets_ignores_wav_without_txt(self, temp_dir):
        """discover_snippets() ignores .wav files without matching .txt."""
        snippets = temp_dir / "snippets"
        snippets.mkdir()
        # Create wav without txt
        (snippets / "orphan.wav").write_bytes(b"fake")

        bench = ASRBench(
            snippets_dir=snippets,
            models_dir=Path("."),
            results_dir=Path("."),
        )
        assert bench.discover_snippets() == []

    def test_run_returns_dataframe(self, snippets_dir, temp_dir):
        """run() returns a pandas DataFrame."""
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=temp_dir,
            results_dir=temp_dir,
        )
        bench.runners = [MockRunner(temp_dir)]
        df = bench.run()
        assert isinstance(df, pd.DataFrame)

    def test_run_includes_required_columns(self, snippets_dir, temp_dir):
        """Results DataFrame has required columns."""
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=temp_dir,
            results_dir=temp_dir,
        )
        bench.runners = [MockRunner(temp_dir)]
        df = bench.run()

        required_columns = [
            "Model", "Snippet", "Transcript", "Reference",
            "WER", "Time_Sec", "RTF", "Peak_RAM_MB", "Load_Time_Sec", "Error"
        ]
        for col in required_columns:
            assert col in df.columns

    def test_run_calculates_wer(self, snippets_dir, temp_dir):
        """run() calculates WER against reference transcript."""
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=temp_dir,
            results_dir=temp_dir,
        )
        bench.runners = [MockRunner(temp_dir)]
        df = bench.run()

        # WER should be calculated (not None)
        assert df["WER"].notna().all()

    def test_run_handles_runner_failure(self, snippets_dir, temp_dir):
        """run() logs error and continues when runner fails."""
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=temp_dir,
            results_dir=temp_dir,
        )
        bench.runners = [FailingRunner(temp_dir), MockRunner(temp_dir)]
        df = bench.run()

        # Should have results from both runners
        assert len(df["Model"].unique()) == 2
        # Failing runner should have error recorded
        failing_rows = df[df["Model"] == "failing-runner"]
        assert failing_rows["Error"].notna().all()

    def test_run_raises_on_no_snippets(self, temp_dir):
        """run() raises ValueError if no snippets found."""
        empty_snippets = temp_dir / "empty"
        empty_snippets.mkdir()

        bench = ASRBench(
            snippets_dir=empty_snippets,
            models_dir=temp_dir,
            results_dir=temp_dir,
        )
        with pytest.raises(ValueError, match="No snippets found"):
            bench.run()

    def test_save_results_creates_csv(self, snippets_dir, temp_dir):
        """save_results() creates a CSV file."""
        results_dir = temp_dir / "results"
        bench = ASRBench(
            snippets_dir=snippets_dir,
            models_dir=temp_dir,
            results_dir=results_dir,
        )
        bench.runners = [MockRunner(temp_dir)]
        df = bench.run()
        output = bench.save_results(df)

        assert output.exists()
        assert output.suffix == ".csv"
        # Verify it's readable
        loaded = pd.read_csv(output)
        assert len(loaded) == len(df)
```

### tests/test_distil_whisper.py (Example Runner Test)

```python
"""Tests for Distil-Whisper runner - write BEFORE implementation."""
import pytest
from pathlib import Path
from runners.distil_whisper import DistilWhisperRunner
from runners.base import TranscriptionResult


class TestDistilWhisperRunner:
    """Distil-Whisper runner tests."""

    def test_has_correct_name(self, temp_dir):
        """Runner should have descriptive name."""
        runner = DistilWhisperRunner(temp_dir)
        assert "distil" in runner.name.lower()
        assert "whisper" in runner.name.lower()

    def test_load_initializes_model(self, temp_dir):
        """load() should initialize model and processor."""
        runner = DistilWhisperRunner(temp_dir)
        runner.load()
        assert runner.is_loaded
        runner.unload()

    def test_transcribe_returns_result(self, temp_dir, sample_audio):
        """transcribe() returns TranscriptionResult."""
        runner = DistilWhisperRunner(temp_dir)
        runner.load()
        try:
            result = runner.transcribe(sample_audio)
            assert isinstance(result, TranscriptionResult)
            assert isinstance(result.transcript, str)
            assert result.time_sec > 0
            assert result.audio_duration_sec > 0
        finally:
            runner.unload()

    def test_unload_frees_memory(self, temp_dir):
        """unload() should free model from memory."""
        runner = DistilWhisperRunner(temp_dir)
        runner.load()
        assert runner.is_loaded
        runner.unload()
        assert not runner.is_loaded

    def test_context_manager(self, temp_dir, sample_audio):
        """Runner works as context manager."""
        with DistilWhisperRunner(temp_dir) as runner:
            result = runner.transcribe(sample_audio)
            assert isinstance(result, TranscriptionResult)
```

---

## CI/CD Configuration

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest  # Note: No AMD GPU in CI, CPU-only tests

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linter
        run: ruff check .

      - name: Run type checker
        run: mypy runners/ benchmark.py

      - name: Run tests
        run: pytest tests/ -v --cov=runners --cov=benchmark

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## Runner Implementation Pattern

Each runner follows this pattern:

```python
# runners/distil_whisper.py
from pathlib import Path
import time
import psutil
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

from .base import BaseRunner, TranscriptionResult


class DistilWhisperRunner(BaseRunner):
    name = "distil-whisper-large-v3"

    def __init__(self, models_dir: Path):
        super().__init__(models_dir)
        self.model_id = "distil-whisper/distil-large-v3"
        self._pipe = None

    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.models_dir,
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.models_dir,
        )

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        self._model = model  # For is_loaded check

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr

        # Measure inference
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        start = time.perf_counter()
        result = self._pipe(audio)
        elapsed = time.perf_counter() - start

        mem_after = process.memory_info().rss / 1024 / 1024
        peak_ram = max(mem_after - mem_before, 0)

        return TranscriptionResult(
            transcript=result["text"].strip(),
            time_sec=elapsed,
            peak_ram_mb=peak_ram,
            audio_duration_sec=audio_duration,
        )
```
