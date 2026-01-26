from abc import ABC, abstractmethod
from typing import Dict, Any

class ASRModel(ABC):
    """Abstract Base Class for all ASR models in the benchmark suite."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        """Load the model and processor into memory."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe the given audio file.
        Returns: The transcription string.
        """
        pass

    def teardown(self):
        """Cleanup resources, move model to CPU or delete to free VRAM/RAM."""
        import torch
        import gc
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
