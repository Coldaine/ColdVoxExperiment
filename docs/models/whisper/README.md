# Whisper (whisper.cpp / whisper-large-v3-turbo)

## Model Card

| Property | Value |
|----------|-------|
| **Model** | openai/whisper-large-v3-turbo |
| **Parameters** | ~809M |
| **Architecture** | Encoder-decoder transformer |
| **License** | MIT |
| **Languages** | 99+ languages |
| **Original** | [OpenAI Whisper](https://github.com/openai/whisper) |

## Overview

Whisper is OpenAI's robust speech recognition model. We use `whisper.cpp` for efficient CPU inference with quantization support, rather than the Python implementation.

## Why whisper.cpp?

- **Quantization**: Q5_0, Q8_0 formats reduce memory and improve speed
- **CPU optimized**: AVX2/AVX512 acceleration
- **No Python overhead**: Native C++ inference
- **Memory efficient**: Can run large models on limited RAM

## Installation

### Python Bindings
```bash
pip install pywhispercpp
# or
pip install whisper-cpp-python
```

### Download Quantized Models
```bash
# From ggerganov/whisper.cpp releases
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
```

### Available Quantizations

| Format | Size | Quality | Speed |
|--------|------|---------|-------|
| f16 | 3.1GB | Best | Slowest |
| q8_0 | 1.6GB | Excellent | Fast |
| q5_0 | 1.1GB | Very Good | Faster |
| q4_0 | 0.8GB | Good | Fastest |

**Recommended**: `q5_0` for best quality/speed tradeoff

## Usage

```python
from pywhispercpp.model import Model

model = Model("ggml-large-v3-turbo-q5_0.bin")
result = model.transcribe("audio.wav")
print(result)
```

## AMD-Specific Notes

### ROCm
whisper.cpp doesn't have ROCm GPU acceleration (uses CUDA or CPU). On AMD:
- **CPU inference only** (still fast with quantization)
- AVX2/AVX512 used automatically if available

### Expected Performance on Z13

| Quantization | RTF (est.) | RAM |
|--------------|------------|-----|
| q5_0 | ~0.3-0.5 | ~2GB |
| q8_0 | ~0.4-0.6 | ~3GB |
| f16 | ~0.8-1.0 | ~6GB |

## Known Issues

1. **No ROCm support**: CPU only on AMD (still performant)
2. **Long audio**: May need chunking for files > 30 minutes
3. **Hallucination**: Can hallucinate on silence or low-quality audio

## Recommended Settings

```python
# Best accuracy
model.transcribe(
    audio_path,
    language="en",           # Specify language if known
    beam_size=5,
    best_of=5,
)

# Best speed
model.transcribe(
    audio_path,
    language="en",
    beam_size=1,
    greedy=True,
)
```

## References

- [whisper.cpp GitHub](https://github.com/ggerganov/whisper.cpp)
- [Quantized Models](https://huggingface.co/ggerganov/whisper.cpp)
- [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp)
