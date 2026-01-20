# IBM Granite Speech (granite-speech-3.3-2b / 8b)

## Model Card

| Property | 2B Model | 8B Model |
|----------|----------|----------|
| **Model** | ibm-granite/granite-speech-3.3-2b | ibm-granite/granite-speech-3.3-8b |
| **Parameters** | ~2B | ~8B |
| **Architecture** | Speech-to-text transformer |
| **License** | Apache 2.0 |
| **Languages** | English, multilingual |
| **HuggingFace** | [Link](https://huggingface.co/ibm-granite/granite-speech-3.3-2b) | [Link](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) |

## Overview

IBM Granite Speech models are enterprise-grade ASR models from IBM Research. They offer strong accuracy with good multilingual support.

## Installation

```bash
pip install transformers accelerate torch
# PyTorch should be from ROCm index for GPU acceleration
```

## Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "ibm-granite/granite-speech-3.3-2b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Transcribe
inputs = processor(audio_array, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs)
transcript = processor.decode(generated_ids[0], skip_special_tokens=True)
```

## AMD-Specific Notes

### ROCm (Recommended)
With ROCm PyTorch installed:
```python
import torch
print(torch.cuda.is_available())  # True with ROCm
model = model.to("cuda")  # Uses ROCm backend
```

### Memory Requirements

| Model | FP16 VRAM | FP32 RAM | Strix Halo Fit? |
|-------|-----------|----------|-----------------|
| 2B | ~4GB | ~8GB | Yes (GPU) |
| 8B | ~16GB | ~32GB | Yes (unified memory) |

The 128GB unified memory on Strix Halo means even the 8B model can run on "GPU" since it shares system RAM.

### Expected Performance on Z13

| Model | Device | RTF (est.) | RAM |
|-------|--------|------------|-----|
| 2B | ROCm GPU | ~0.2-0.4 | ~5GB |
| 2B | CPU | ~0.8-1.2 | ~8GB |
| 8B | ROCm GPU | ~0.4-0.8 | ~18GB |
| 8B | CPU | ~2.0-3.0 | ~32GB |

## Known Issues

1. **8B memory pressure**: May need `gc.collect()` after unloading
2. **First inference slow**: JIT compilation on first run
3. **Audio preprocessing**: Requires specific sample rate (16kHz)

## Recommended Settings

### 2B Model (Speed)
```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "ibm-granite/granite-speech-3.3-2b",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
```

### 8B Model (Accuracy)
```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "ibm-granite/granite-speech-3.3-8b",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# After use, aggressively free memory
del model
import gc
gc.collect()
torch.cuda.empty_cache()
```

## References

- [IBM Granite Models](https://huggingface.co/ibm-granite)
- [Granite Speech Paper](https://arxiv.org/abs/...) (check HuggingFace for latest)
