import pytest
from harness.models.base import ASRModel

def test_asr_model_is_abstract():
    with pytest.raises(TypeError):
        ASRModel("test-id")

class MockModel(ASRModel):
    def load(self):
        self.model = "loaded"
    def transcribe(self, audio_path):
        return "mock transcript"

def test_mock_model_implementation():
    model = MockModel("test-id")
    model.load()
    assert model.model == "loaded"
    assert model.transcribe("fake.wav") == "mock transcript"
    model.teardown()
    assert model.model is None
