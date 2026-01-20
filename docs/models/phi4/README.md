# Phi-4 Multimodal (microsoft/Phi-4-multimodal-instruct)

## Model Card

| Property | Value |
|----------|-------|
| **Model** | microsoft/Phi-4-multimodal-instruct |
| **Parameters** | ~14B |
| **Architecture** | Multimodal LLM (text + audio + vision) |
| **License** | MIT |
| **Languages** | English (primarily) |
| **HuggingFace** | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |

## Status: EXPERIMENTAL

This model is marked as **experimental** in our benchmark because:
1. It's a multimodal LLM, not a dedicated ASR model
2. Requires custom audio preprocessing pipeline
3. May have compatibility issues with current transformers versions

## Overview

Phi-4 Multimodal is Microsoft's multimodal language model that can process audio, images, and text. For ASR, it can transcribe audio but requires specific prompt engineering.

## Installation

```bash
pip install transformers accelerate torch
# Additional dependencies may be required for audio encoding
pip install soundfile librosa
```

## Usage (Theoretical)

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_id = "microsoft/Phi-4-multimodal-instruct"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load and process audio
audio = processor.audio_processor(audio_array, return_tensors="pt")

# Create prompt
prompt = "<audio>Transcribe this audio:</audio>"

# Generate
inputs = processor(text=prompt, audio=audio, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
transcript = processor.decode(outputs[0], skip_special_tokens=True)
```

## AMD-Specific Notes

### Memory Requirements
| Precision | VRAM/RAM |
|-----------|----------|
| FP16 | ~28GB |
| FP32 | ~56GB |

**Strix Halo**: May fit with 128GB unified memory, but will be slow.

### ROCm Status
- Should work via PyTorch ROCm if model loads
- Custom operators may have issues

### Expected Performance on Z13

| Device | RTF (est.) | RAM | Status |
|--------|------------|-----|--------|
| ROCm GPU | Unknown | ~30GB | Untested |
| CPU | Very slow | ~56GB | May work |

## Known Issues

1. **Not a dedicated ASR model**: Designed for multimodal understanding, not optimized for transcription
2. **Complex pipeline**: Audio preprocessing is non-trivial
3. **Large memory footprint**: 14B parameters
4. **trust_remote_code**: Requires trusting remote code
5. **Transformers version**: May need specific version

## Fallback Behavior

In the benchmark, if Phi-4 fails to load or process audio:
1. Log the error
2. Record "SKIPPED" in results
3. Continue to next model

```python
try:
    phi4_runner.load()
    result = phi4_runner.transcribe(audio)
except Exception as e:
    result = TranscriptionResult(
        transcript="",
        error=f"Phi-4 failed: {str(e)}",
        skipped=True,
    )
```

## When to Use Phi-4 for ASR

Consider Phi-4 only if you need:
- Combined audio + image understanding
- Conversational follow-up about audio content
- Audio summarization (not just transcription)

For pure transcription, use dedicated ASR models (Whisper, Granite, etc.).

## References

- [Phi-4 Technical Report](https://arxiv.org/abs/...)
- [Microsoft Phi Models](https://huggingface.co/microsoft)
- [Multimodal LLM Audio](https://huggingface.co/docs/transformers/main/en/model_doc/phi4)
