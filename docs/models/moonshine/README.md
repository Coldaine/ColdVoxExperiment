# Moonshine (useful-sensors/moonshine-base)

## Model Card

| Property | Value |
|----------|-------|
| **Model** | useful-sensors/moonshine-base |
| **Parameters** | ~100M |
| **Architecture** | Encoder-decoder, ONNX optimized |
| **License** | Apache 2.0 |
| **Languages** | English (primarily) |
| **HuggingFace** | [useful-sensors/moonshine-base](https://huggingface.co/useful-sensors/moonshine-base) |

## Overview

Moonshine is a lightweight ASR model designed for edge deployment. It's optimized for ONNX Runtime, making it ideal for CPU and DirectML inference without requiring full PyTorch.

## Installation

### Option 1: Official Package
```bash
pip install useful-sensors-moonshine
```

### Option 2: Direct ONNX
```bash
pip install onnxruntime onnxruntime-directml
# Download ONNX weights from HuggingFace
```

## Usage

```python
from moonshine import Moonshine

model = Moonshine()
transcript = model.transcribe("audio.wav")
```

## AMD-Specific Notes

### ROCm
ONNX Runtime doesn't have native ROCm support, but you can use:
- `onnxruntime-directml` on Windows (recommended)
- CPU fallback (still fast due to model size)

### DirectML
```python
import onnxruntime as ort
sess = ort.InferenceSession("moonshine.onnx", providers=['DmlExecutionProvider'])
```

### Expected Performance on Z13

| Mode | RTF (est.) | RAM |
|------|------------|-----|
| DirectML | ~0.1 | ~500MB |
| CPU | ~0.3 | ~500MB |

## Known Issues

1. **VAD dependency**: May need external Voice Activity Detection for long audio
2. **English only**: Limited multilingual support compared to Whisper
3. **Preprocessing**: Requires specific audio preprocessing (16kHz mono)

## Recommended Settings

```python
# Best accuracy
model = Moonshine(beam_size=5)

# Best speed
model = Moonshine(beam_size=1, greedy=True)
```

## References

- [Moonshine GitHub](https://github.com/usefulsensors/moonshine)
- [ONNX Runtime DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
