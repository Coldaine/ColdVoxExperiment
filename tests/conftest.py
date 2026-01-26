import pytest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_audio(temp_dir) -> Path:
    audio_path = temp_dir / "test_silence.wav"
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(audio_path, samples, 16000)
    return audio_path
