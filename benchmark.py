import sys
from pathlib import Path

# Add the project root to sys.path so we can import from harness
root_dir = Path(__file__).parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from harness.runner import run_benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASR Benchmarking Wrapper")
    parser.add_argument("--model", type=str, required=True, choices=["moonshine", "distil-whisper"])
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--output", type=str, default="reports/latest_results.json")
    
    args = parser.parse_args()
    run_benchmark(args.model, args.manifest, args.output)
