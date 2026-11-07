"""Compare two native binaries in ABBA order, checking logits and generated IDs."""
import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from bench import CASES
from reference import ROOT
from wmt import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--before-model", type=Path)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/optimization.json")
    args = parser.parse_args()
    before = args.before.resolve()
    after = ROOT / ".build/release/plamo"
    if not before.is_file() or before == after.resolve():
        parser.error("Provide a separate baseline binary with its resource bundle")
    data = [r for r in rows() if r["domain"] == "news"]
    document = data[0]["document_id"]
    segments = sorted([r for r in data if r["document_id"] == document], key=lambda r: r["segment_id"])[:5]
    paragraph = CASES[2][2]
    cases = [("raw-short", False, None, paragraph, 128),
             ("packed-short", True, None, paragraph, 128),
             ("packed-document", True, None, "\n".join(r["source"] for r in segments), 1024),
             ("packed-context", True, 32768, paragraph, 128)]
    report = []
    with tempfile.TemporaryDirectory(prefix="plamo-optimize-") as directory:
        dump = Path(directory) / "logits.f32"
        for name, packed, context, text, limit in cases:
            samples = {"before": [], "after": []}
            expected_ids, expected_logits = None, None
            for label in ["before", "after", "after", "before"]:
                command = [str(before if label == "before" else after), "--to", "ja", "--repeat", "3",
                           "--max-tokens", str(limit), "--dump-logits", str(dump)]
                if label == "before" and args.before_model:
                    command += ["--model", str(args.before_model)]
                if args.mtp:
                    command += ["--mtp"]
                if packed:
                    command += ["--packed"]
                if context:
                    command += ["--context", str(context)]
                process = subprocess.run(command + [text], cwd=ROOT, text=True, capture_output=True, check=True, timeout=180)
                metrics = json.loads(process.stderr)
                runs = metrics["runs"]
                ids, logits = runs[0]["output_ids"], dump.read_bytes()
                assert all(r["stop_reason"] == "stop" and r["output_ids"] == ids for r in runs), name
                if expected_ids is None:
                    expected_ids, expected_logits = ids, logits
                assert ids == expected_ids and logits == expected_logits, (name, label, "output changed")
                samples[label].append(metrics)
            result = dict(case=name, mtp=args.mtp, input_tokens=runs[0]["input_tokens"], output_tokens=len(expected_ids),
                          logits_bitwise_equal=True, generated_ids_equal=True, samples=samples)
            for label in ("before", "after"):
                warm = [r for metrics in samples[label] for r in metrics["runs"][1:]]
                times = [r["prefill_seconds"] + r["decode_seconds"] for r in warm]
                result[label] = dict(seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
                                     prefill_seconds=statistics.median(r["prefill_seconds"] for r in warm),
                                     decode_seconds=statistics.median(r["decode_seconds"] for r in warm),
                                     gpu_seconds=statistics.median(r["gpu_seconds"] for r in warm),
                                     metal_bytes=max(m["metal_allocated_bytes"] for m in samples[label]))
            result["speedup"] = result["before"]["seconds"] / result["after"]["seconds"]
            report.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
