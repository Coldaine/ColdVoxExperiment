# Voxtral Mini (mistralai/Voxtral-Mini-3B-2507)

## Model Card

| Property | Value |
|----------|-------|
| **Model** | mistralai/Voxtral-Mini-3B-2507 |
| **Parameters** | ~3B |
| **Architecture** | Speech-to-text transformer |
| **License** | Apache 2.0 |
| **Languages** | Multilingual |
| **HuggingFace** | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) |

## Overview

Voxtral is Mistral AI's speech recognition model, released mid-2025. It's designed for efficient multilingual transcription with a relatively compact 3B parameter footprint.

## Installation

```bash
pip install transformers accelerate torch
# PyTorch should be from ROCm index for GPU acceleration
```

## Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load audio
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)

# Transcribe
inputs = processor(audio, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs)
transcript = processor.decode(generated_ids[0], skip_special_tokens=True)
```

## AMD-Specific Notes

### ROCm (Recommended)
Standard PyTorch model, should work with ROCm:
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Expected Performance on Z13

| Device | RTF (est.) | RAM |
|--------|------------|-----|
| ROCm GPU | ~0.2-0.4 | ~6GB |
| CPU | ~1.0-1.5 | ~12GB |

## Known Issues

1. **New model**: Released mid-2025, less community testing
2. **Availability**: Verify HuggingFace availability before benchmarking
3. **Documentation**: May have limited documentation compared to Whisper

## Fallback: Kyutai Moshi

If Voxtral is unavailable or fails, consider Kyutai Moshi as an alternative:

```bash
pip install moshi
```

```python
from moshi import Moshi
model = Moshi()
transcript = model.transcribe("audio.wav")
```

## Recommended Settings

### Best Speed
```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Greedy decoding
generated_ids = model.generate(
    **inputs,
    do_sample=False,
    num_beams=1,
)
```

### Best Accuracy
```python
generated_ids = model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    length_penalty=1.0,
)
```

## Comparison with Other 3B Models

| Model | Parameters | Focus |
|-------|------------|-------|
| Voxtral Mini | 3B | Multilingual ASR |
| Granite 2B | 2B | Enterprise ASR |
| Distil-Whisper | 756M | Speed-optimized ASR |

Voxtral sits in the middle ground: larger than Distil-Whisper but smaller than Granite 8B.

## References

- [Mistral AI](https://mistral.ai)
- [Voxtral Announcement](https://mistral.ai/news/voxtral/) (if available)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) (fallback)
