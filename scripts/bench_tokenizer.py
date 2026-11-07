"""Compare CLI tokenization and retained host memory, with identical outputs."""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from bench import CASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--before-model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tokenizer-lifetime.json")
    args = parser.parse_args()
    binaries = {"before": args.before.resolve(), "after": ROOT / ".build/release/plamo"}
    if not binaries["before"].is_file() or binaries["before"] == binaries["after"].resolve():
        parser.error("Provide a separate baseline binary with its resource bundle")
    text = CASES[2][2]
    report = []
    for case, arguments in [("encode", ["--tokenize", text]),
                            ("translation", ["--packed", "--to", "ja", "--max-tokens", "128", text])]:
        expected = None
        samples = {"before": [], "after": []}
        for label in ["before", "after", "after", "before", "after", "before", "before", "after"]:
            command = [str(binaries[label]), *arguments]
            if label == "before" and args.before_model:
                command += ["--model", str(args.before_model)]
            started = time.perf_counter()
            process = subprocess.run(command, cwd=ROOT, text=True,
                                     capture_output=True, check=True, timeout=60)
            seconds = time.perf_counter() - started
            if case == "encode":
                ids = json.loads(process.stdout)
                value, metrics = ids, None
            else:
                metrics = json.loads(process.stderr)
                assert metrics["runs"][0]["stop_reason"] == "stop"
                ids = metrics["runs"][0]["output_ids"]
                value = (ids, process.stdout)
            if expected is None:
                expected = value
            assert value == expected, (case, label, "output changed")
            samples[label].append(dict(seconds=seconds, metrics=metrics))
        result = dict(case=case, ids=ids, matched=True, samples=samples)
        for label in ["before", "after"]:
            rows = samples[label]
            times = [r["seconds"] for r in rows]
            summary = dict(seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times))
            if case == "translation":
                for key in ["peak_rss_bytes", "physical_footprint_bytes", "peak_physical_footprint_bytes", "metal_allocated_bytes"]:
                    summary[key] = statistics.median(r["metrics"][key] for r in rows)
            result[label] = summary
        report.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "ids")}), flush=True)


if __name__ == "__main__":
    main()
