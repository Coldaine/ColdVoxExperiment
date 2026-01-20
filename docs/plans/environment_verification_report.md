# Environment & Model Verification Report

**Date:** January 20, 2026
**Target Environment:** Windows 11, AMD Ryzen AI MAX+ 395 (Strix Halo), 128GB RAM
**Purpose:** Verify feasibility of the "North Star" ASR benchmark plan.

## 1. Environment Compatibility: Strix Halo + Windows

### ✅ ROCm 7.1.1+ Support
*   **Status:** **Supported**. Native PyTorch on Windows is available for gfx1151 (Strix Halo).
*   **Verification:** Run this script to confirm valid ROCm detection:
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"ROCm Version: {torch.version.hip}")
else:
    print("❌ ROCm not detected. Check PyTorch installation index.")
```

## 2. Critical Fixes for Implementation Plan

### 🚨 Issue 1: "useful-sensors-moonshine" Package Does Not Exist
The plan suggests `pip install useful-sensors-moonshine`. This is incorrect.
**Solution:** Use standard `transformers` or the `useful-moonshine-onnx` repo.
**Recommended Code (Transformers):**
```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa

# Load from Hugging Face
model_id = "UsefulSensors/moonshine-base"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Transcribe
audio, sr = librosa.load("test.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
generated_ids = model.generate(**inputs)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### 🚨 Issue 2: Phi-4 Multimodal "Audio Preprocessing" Missing
The plan notes "requires specific audio preprocessing" but gives no details.
**Solution:** Use the `AutoProcessor` which handles the multimodal inputs (text + audio).
**Recommended Code:**
```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import librosa

model_id = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

# Load Audio
audio, sr = librosa.load("test.wav", sr=16000)

# Create Prompt with Audio
messages = [
    {"role": "user", "content": "<|audio|>Transcribe this audio clip verbatim."},
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process Inputs
inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=1000)
print(processor.batch_decode(generate_ids, skip_special_tokens=True)[0])
```

## 3. Dependency Recommendations

| Dependency | Recommendation | Reason |
|:---|:---|:---|
| **Moonshine** | `pip install transformers librosa` | "useful-sensors-moonshine" package is invalid. |
| **PyTorch** | `pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/` | **Strix Halo (gfx1151)** requires nightly index or ROCm 7.1+ specifically. Standard PyTorch index typically trails behind latest hardware support. |
| **Visual C++** | **Install Latest Redistributable** | `onnxruntime` and `torch` on Windows depend on recent MSVC runtimes. Often missing on fresh Windows installs. |

## 4. Conclusion
The plan structure is solid, but the specific implementation details for **Moonshine** and **Phi-4** were broken or vague. The code snippets above provide the concrete "how-to" needed to actually run these models.
