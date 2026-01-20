# Testing Guide

## Running the Benchmark

### Quick Start
```bash
python benchmark.py
```

### With Options
```bash
# Specify snippets directory
python benchmark.py --snippets ./my_audio_files

# Run only specific models
python benchmark.py --models moonshine,granite-2b

# Skip warmup (faster but less accurate timing)
python benchmark.py --no-warmup

# Verbose output
python benchmark.py -v
```

## Preparing Test Snippets

### File Structure
Each audio file needs a matching transcript file:

```
snippets/
├── interview_clip.wav
├── interview_clip.txt      # Reference transcript
├── podcast_intro.wav
├── podcast_intro.txt
└── noisy_recording.wav
    noisy_recording.txt
```

### Audio Requirements
- **Format**: WAV (16-bit PCM recommended)
- **Sample Rate**: 16kHz (will be resampled if different)
- **Channels**: Mono (will be converted if stereo)
- **Duration**: 5 seconds to 5 minutes recommended

### Transcript Format
Plain text, one file per audio:
```
Hello world, this is a test of the speech recognition system.
```

Tips:
- No timestamps needed
- Punctuation optional (normalized during WER calculation)
- Case insensitive comparison

## Adding a New Model

### 1. Create Runner Class

Create `runners/your_model.py`:

```python
from runners.base import BaseRunner, TranscriptionResult
from pathlib import Path

class YourModelRunner(BaseRunner):
    name = "your-model-name"

    def load(self):
        """Load model weights into memory."""
        # Import and initialize your model
        pass

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a single audio file."""
        # Return TranscriptionResult with transcript and timing
        pass

    def unload(self):
        """Free model from memory."""
        # Delete model, call gc.collect()
        pass
```

### 2. Register in benchmark.py

```python
from runners.your_model import YourModelRunner

# Add to model list
self.models.append(YourModelRunner())
```

### 3. Add Documentation

Create `docs/models/your-model/README.md` with:
- Model card info
- Installation requirements
- Known issues
- Performance expectations

## Interpreting Results

### results.csv Columns

| Column | Description |
|--------|-------------|
| `Model` | Model identifier |
| `Snippet` | Audio filename |
| `Transcript` | Model's transcription |
| `Reference` | Ground truth text |
| `WER` | Word Error Rate (0.0 = perfect) |
| `Time_Sec` | Inference time in seconds |
| `RTF` | Real-Time Factor (< 1.0 = faster than real-time) |
| `Peak_RAM_MB` | Peak memory usage |
| `Load_Time_Sec` | Model load time |
| `Error` | Error message if failed |

### WER Interpretation

| WER | Quality |
|-----|---------|
| 0-5% | Excellent |
| 5-10% | Good |
| 10-20% | Acceptable |
| 20-30% | Poor |
| 30%+ | Unusable |

### RTF Interpretation

| RTF | Meaning |
|-----|---------|
| < 0.1 | 10x faster than real-time |
| 0.1-0.5 | 2-10x faster than real-time |
| 0.5-1.0 | Faster than real-time |
| 1.0 | Real-time |
| > 1.0 | Slower than real-time |

## Troubleshooting

### Model fails to load
1. Check RAM availability: `psutil.virtual_memory()`
2. Try CPU-only mode if GPU OOM
3. Check model-specific docs for requirements

### ROCm not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True with ROCm
print(torch.version.hip)          # Should show HIP version
```

If not working:
1. Verify ROCm installation: `rocminfo`
2. Check PyTorch was installed from ROCm index
3. Fallback to DirectML or CPU

### WER seems wrong
- Check transcript encoding (UTF-8)
- Verify audio and transcript match
- Check for silence/empty transcripts

## CI/Automation

### GitHub Actions Example
```yaml
name: ASR Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: self-hosted  # Needs AMD hardware
    steps:
      - uses: actions/checkout@v4
      - run: python setup_env.py
      - run: python download_models.py
      - run: python benchmark.py
      - uses: actions/upload-artifact@v4
        with:
          name: results
          path: results/
```

Note: CI runners typically don't have AMD GPUs, so expect CPU-only results.
