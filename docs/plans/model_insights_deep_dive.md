# Deep Dive: ASR Model Insights & "Gotchas"

**Date:** January 20, 2026
**Purpose:** Provide "unvarnished" truths about the 7 candidate models to prevent wasted time on known pitfalls.

## 1. 🚨 Critical "Gotchas" by Model

| Model | The "Gotcha" | Impact | Mitigation |
|:---|:---|:---|:---|
| **Moonshine** | **English-Centric & Hallucinations** | Originally designed for English; can hallucinate repetitive text on short/silence segments. | Use `useful-moonshine-onnx` repo. Avoid for non-English usage unless verified. |
| **Whisper V3** | **Hallucinations** | **High risk** of "poisoned" loops (repeating same phrase) in long audio. Worse than V2 in some cases. | **Use Distil-Whisper instead** for long audio (see below) or strict `condition_on_previous_text=False`. |
| **Distil-Whisper** | **The "Better" Whisper** | Surprisingly **LESS** prone to hallucination than full Whisper V3 on long-form audio. | **Prioritize Distil-Whisper** over standard Whisper V3 for the "reliable" baseline. |
| **Granite 8B** | **Two-Pass Architecture** | It's not just "audio-in, text-out". Often requires distinct ASR + LLM passes. `vLLM` serving has known "end-of-file" hallucination bugs. | Ensure ample silence padding at end of audio. Be wary of `vLLM` implementations; use native Transformers. |
| **Voxtral** | **Integration Hell** | Newest model. Known bugs with "missing tokenizer tokens" in ONNX and broken HuggingFace pipelines in recent versions. | Expect to write custom inference code; standard `pipeline()` may fail contentiously. |
| **Phi-4** | **No Timestamps** | It outputs raw text only. No start/end times for words. 40s audio limit. | Use only for "short segment accuracy" benchmarks. **Cannot** replace Whisper for full-file subtitling. |

## 2. Practical Recommendations

### "The Safe Bet" Category
*   **Distil-Whisper Large V3**: It turns out this isn't just "faster" Whisper; it's often *more stable* Whisper. It should be your primary baseline for long audio.

### "The Speed Demon" Category
*   **Moonshine**: Excellent for sub-1s latency, but treat it as an "English Command/Control" model rather than a general-purpose transcriber.

### "The Wildcards"
*   **Phi-4**: Keep it to prove you *can* run a 14B multimodal model, but don't expect it to be a daily driver for ASR due to the timestamp/length limitations.
*   **Voxtral**: High risk of implementation failure due to immature software support. Allocate extra time for debugging.

## 3. Deployment Advice for Strix Halo (Windows)

*   **Avoid WSL2**: You have native ROCm 7.1.1+ support. Run directly on Windows to access the full 128GB unified memory without virtualization overhead.
*   **Quantization Matters**: For Whisper.cpp on CPU:
    *   `q5_0`: Sweet spot for speed/accuracy.
    *   `q8_0`: Minimal gain over q5_0 but slower.
    *   Avoid `f16` on CPU (too slow).

## 4. Final Verdict for Benchmark

Don't just measure WER/RTF. Measure **"Time to First Headache"**:
1.  **Moonshine**: Low headache (if English).
2.  **Distil-Whisper**: Low headache.
3.  **Whisper V3**: Medium headache (hallucinations).
4.  **Granite**: High headache (architecture).
5.  **Voxtral**: Extreme headache (bugs).
