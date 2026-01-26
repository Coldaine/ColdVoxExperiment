import argparse
import json
import logging
import os
import torch
import librosa
from typing import List, Dict, Any
from harness.metrics import PerformanceTracker, calculate_wer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_manifest(manifest_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest not found at {manifest_path}. Using empty list.")
        return []
    with open(manifest_path, 'r') as f:
        return json.load(f)

def run_benchmark(model_name: str, manifest_path: str, output_path: str):
    # 1. Initialize Model
    logger.info(f"Loading model: {model_name}")
    
    # Dynamic import based on model name
    if model_name == "moonshine":
        from harness.models.moonshine_wrapper import MoonshineModel
        model = MoonshineModel("UsefulSensors/moonshine-base")
    elif model_name == "distil-whisper":
        from harness.models.distil_whisper_wrapper import DistilWhisperModel
        model = DistilWhisperModel("distil-whisper/distil-large-v3")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load()

    # 2. Load Data
    samples = load_manifest(manifest_path)
    if not samples:
        logger.error("No samples to process.")
        return

    results = []
    tracker = PerformanceTracker()

    # 3. Benchmark Loop
    for sample in samples:
        audio_path = sample['audio_filepath']
        reference_text = sample['text']
        
        logger.info(f"Processing: {audio_path}")
        
        # Warm-up (optional, but good for first run)
        # tracker.start() / tracker.stop() around a dummy call if needed
        
        tracker.start()
        try:
            hypothesis = model.transcribe(audio_path)
            tracker.update_peak_mem()
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            hypothesis = ""
        
        perf_metrics = tracker.stop()
        
        # Calculate accuracy
        wer = calculate_wer(reference_text, hypothesis)
        
        # Get audio duration for RTF
        audio_duration = librosa.get_duration(path=audio_path)
        rtf = perf_metrics['duration'] / audio_duration if audio_duration > 0 else 0

        res = {
            "audio": audio_path,
            "reference": reference_text,
            "hypothesis": hypothesis,
            "wer": wer,
            "duration": perf_metrics['duration'],
            "audio_len": audio_duration,
            "rtf": rtf,
            "peak_ram_gb": perf_metrics['peak_ram_gb']
        }
        results.append(res)
        logger.info(f"Result: WER={wer:.2f}, RTF={rtf:.2f}")

    # 4. Save Results
    output_data = {
        "model": model_name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "results": results,
        "summary": {
            "avg_wer": sum(r['wer'] for r in results) / len(results),
            "avg_rtf": sum(r['rtf'] for r in results) / len(results),
            "max_ram_gb": max(r['peak_ram_gb'] for r in results)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"Benchmark complete. Results saved to {output_path}")
    model.teardown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Benchmarking Runner")
    parser.add_argument("--model", type=str, required=True, choices=["moonshine", "distil-whisper"])
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--output", type=str, default="reports/latest_results.json")
    
    args = parser.parse_args()
    run_benchmark(args.model, args.manifest, args.output)
