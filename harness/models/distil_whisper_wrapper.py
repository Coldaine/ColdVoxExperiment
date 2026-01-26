import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from harness.models.base import ASRModel

class DistilWhisperModel(ASRModel):
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        device = next(self.model.parameters()).device
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features, 
                max_new_tokens=128
            )
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
