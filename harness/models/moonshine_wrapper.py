import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from harness.models.base import ASRModel

class MoonshineModel(ASRModel):
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Moonshine usually requires trust_remote_code=True if not in main transformers yet
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        device = next(self.model.parameters()).device
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
