import sys
import os
import traceback
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Path: {sys.path}")
print(f"Env Path: {os.environ.get('PATH')}")

try:
    print("\n--- Testing transformers import ---")
    from transformers import AutoModelForSpeechSeq2Seq
    print("Transformers imported successfully")
    
    print("\n--- Testing Model Pretrained Loading (CPU) ---")
    # Smallest possible model load test
    model_id = "openai/whisper-tiny" 
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, low_cpu_mem_usage=True)
    print("Model loaded successfully")
    
except Exception as e:
    print(f"\nFAILURE DETECTED")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
    traceback.print_exc()
