"""Sequential, warm-run comparison; no claims about sustained thermal performance."""
import json
import statistics
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("ja-en", "en", "会議は来週の月曜日に変更されました。"),
    ("en-ja", "ja", "The meeting has been rescheduled for next Monday."),
    ("paragraph", "ja", "Check the configuration file before starting the system. "
     "If any errors are found, stop the process and display a message. "
     "Do not delete the original file until the backup has been verified."),
]


def main():
    rows = []
    for name, target, text in CASES:
        for backend in ["native", "mlx"]:
            cmd = ([str(ROOT / ".build/release/plamo")] if backend == "native"
                   else [sys.executable, str(ROOT / "scripts/reference.py")])
            cmd += ["--to", target, "--max-tokens", "128", "--repeat", "5", text]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
            metrics = json.loads(result.stderr)
            warm = metrics["runs"][1:]
            assert all(r["stop_reason"] == "stop" for r in warm)
            assert all(r["output_ids"] == warm[0]["output_ids"] for r in warm)
            row = dict(case=name, backend=backend, translation=result.stdout.strip(),
                       input_tokens=warm[0]["input_tokens"], output_tokens=warm[0]["output_tokens"],
                       median_prefill_seconds=statistics.median(r["prefill_seconds"] for r in warm),
                       median_decode_tokens_per_second=statistics.median(r["decode_tokens_per_second"] for r in warm),
                       metrics=metrics)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}, ensure_ascii=False), flush=True)
    path = ROOT / "results/benchmark.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
