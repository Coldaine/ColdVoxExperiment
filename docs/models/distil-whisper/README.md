# Distil-Whisper (distil-whisper/distil-large-v3)

## Model Card

| Property | Value |
|----------|-------|
| **Model** | distil-whisper/distil-large-v3 |
| **Parameters** | ~756M (49% of Whisper large) |
| **Architecture** | Distilled encoder-decoder |
| **License** | MIT |
| **Languages** | 99+ languages |
| **HuggingFace** | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |

## Overview

Distil-Whisper is a distilled version of Whisper that's 6x faster while maintaining comparable accuracy. It uses knowledge distillation to compress the large Whisper model.

## Why Distil-Whisper over faster-whisper?

For AMD/CPU:
- **transformers** pipeline works better than CTranslate2 on CPU
- CTranslate2 (used by faster-whisper) is CUDA-optimized
- Direct ROCm support via PyTorch

## Installation

```bash
pip install transformers accelerate torch
# PyTorch should be from ROCm index for GPU acceleration
```

## Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device_map="auto",
)

result = pipe("audio.wav")
print(result["text"])
```

## AMD-Specific Notes

### ROCm (Recommended)
```python
import torch
# With ROCm PyTorch:
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Why NOT faster-whisper on AMD
faster-whisper uses CTranslate2 which:
- Has CUDA-optimized kernels
- CPU mode is slower than native transformers
- No ROCm support

**Recommendation**: Use transformers directly for AMD.

### Expected Performance on Z13

| Device | RTF (est.) | RAM |
|--------|------------|-----|
| ROCm GPU | ~0.15-0.3 | ~3GB |
| CPU | ~0.5-0.8 | ~4GB |

## Known Issues

1. **Slightly lower accuracy**: ~1-2% WER increase vs full Whisper
2. **Long-form degradation**: May struggle with very long audio (>10 min)
3. **Hallucination inherited**: Same hallucination issues as Whisper

## Recommended Settings

### Best Speed
```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device_map="auto",
    chunk_length_s=30,
    batch_size=8,  # If RAM allows
)
```

### Best Accuracy
```python
result = pipe(
    "audio.wav",
    return_timestamps=True,
    generate_kwargs={"language": "english"},
)
```

## Comparison: Whisper vs Distil-Whisper

| Metric | Whisper Large v3 | Distil Large v3 |
|--------|------------------|-----------------|
| Parameters | 1.5B | 756M |
| Speed | 1x | 6x |
| WER (LibriSpeech) | 2.7% | 2.9% |
| Memory | ~6GB | ~3GB |

## References

- [Distil-Whisper Paper](https://arxiv.org/abs/2311.00430)
- [HuggingFace Model](https://huggingface.co/distil-whisper/distil-large-v3)
- [Distil-Whisper GitHub](https://github.com/huggingface/distil-whisper)
