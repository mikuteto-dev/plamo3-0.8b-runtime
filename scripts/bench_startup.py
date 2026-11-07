"""Compare model initialization without timing a complete translation."""
import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--before-model", type=Path)
    parser.add_argument("--after-first", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/kernel-startup.json")
    args = parser.parse_args()
    binaries = {"before": args.before.resolve(), "after": ROOT / ".build/release/plamo"}
    if not binaries["before"].is_file() or binaries["before"] == binaries["after"].resolve():
        parser.error("Provide a separate baseline binary with its resource bundle")
    report = []
    with tempfile.TemporaryDirectory(prefix="plamo-startup-") as directory:
        dump = Path(directory) / "logits.f32"
        for packed in [False, True]:
            samples = {"before": [], "after": []}
            expected = None
            order = ["before", "after", "after", "before", "after", "before", "before", "after"]
            for label in (reversed(order) if args.after_first else order):
                command = [str(binaries[label]), "--tokens", "1,500,600", "--max-tokens", "0", "--dump-logits", str(dump)]
                if label == "before" and args.before_model:
                    command += ["--model", str(args.before_model)]
                if packed:
                    command += ["--packed"]
                started = time.perf_counter()
                process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=60)
                seconds = time.perf_counter() - started
                metrics = json.loads(process.stderr)
                assert metrics["runs"][0]["input_tokens"] == 3 and metrics["runs"][0]["stop_reason"] == "prefill"
                values = dump.read_bytes()
                assert len(values) == 107520 * 4
                if expected is None:
                    expected = values
                assert values == expected, (packed, label, "logits changed")
                samples[label].append(dict(seconds=seconds, metrics=metrics))
            result = dict(packed=packed, after_first=args.after_first, matched=True,
                          logits_sha256=hashlib.sha256(expected).hexdigest(), samples=samples)
            for label in ["before", "after"]:
                rows = samples[label]
                result[label] = dict(load_seconds=statistics.median(r["metrics"]["load_seconds"] for r in rows),
                                     min_load_seconds=min(r["metrics"]["load_seconds"] for r in rows),
                                     max_load_seconds=max(r["metrics"]["load_seconds"] for r in rows),
                                     process_seconds=statistics.median(r["seconds"] for r in rows),
                                     metal_bytes=max(r["metrics"]["metal_allocated_bytes"] for r in rows))
            report.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
