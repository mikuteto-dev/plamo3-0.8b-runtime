"""Measure complete translation, not just the last prefill token."""
import argparse
import json
import statistics
import subprocess
import time

from sacrebleu.metrics import CHRF

from reference import ROOT, load_tokenizer, prompt
from wmt import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--qsa", action="store_true")
    parser.add_argument("--qsa-budget", type=int, default=2048)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--direction", choices=["en-ja", "ja-en"], default="en-ja")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    tokenizer = load_tokenizer()
    data = [r for r in rows() if r["domain"] == "news"]
    if not 1 <= args.documents <= 16:
        parser.error("Use 1...16 documents")
    documents = list(dict.fromkeys(r["document_id"] for r in data))[:args.documents]
    document = documents[0]
    data = [r for d in documents for r in sorted([r for r in data if r["document_id"] == d], key=lambda r: r["segment_id"])]
    source_key, target_key, target = ("source", "target", "ja") if args.direction == "en-ja" else ("target", "source", "en")
    source, reference, ids, used_documents = [], [], [], []
    for row in data:
        candidate = "\n".join(source + [row[source_key]])
        if len(tokenizer.encode(prompt(candidate, target))) > args.input_tokens:
            break
        source.append(row[source_key]); reference.append(row[target_key]); ids.append(row["segment_id"])
        if row["document_id"] not in used_documents:
            used_documents.append(row["document_id"])
    if not source:
        raise ValueError("Token budget is too small")
    started = time.perf_counter()
    process = subprocess.run([str(ROOT / ".build/release/plamo"), *(["--packed"] if args.packed else []),
                              *(["--qsa", "--qsa-budget", str(args.qsa_budget)] if args.qsa else []), *(["--mtp"] if args.mtp else []), "--to", target,
                              "--max-tokens", str(args.max_tokens), "--repeat", str(args.repeat), "\n".join(source)],
                             cwd=ROOT, text=True, capture_output=True, check=True, timeout=480)
    wall = time.perf_counter()-started
    metrics = json.loads(process.stderr)
    runs = metrics["runs"][1:] or metrics["runs"]
    if any(r["stop_reason"] != "stop" for r in runs):
        raise AssertionError("Translation truncated")
    if any(r["output_ids"] != runs[0]["output_ids"] for r in runs):
        raise AssertionError("Nondeterministic translation")
    result = dict(label=args.label, direction=args.direction, document_id=document, document_ids=used_documents, segment_ids=ids,
                  process_wall_seconds=wall, warmup_runs=1 if args.repeat > 1 else 0,
                  warm_translation_seconds=statistics.median(r["prefill_seconds"]+r["decode_seconds"] for r in runs),
                  warm_gpu_seconds=statistics.median(r["gpu_seconds"] for r in runs),
                  chrf=CHRF().corpus_score([process.stdout.strip()], [["\n".join(reference)]]).score,
                  translation=process.stdout.strip(), metrics=metrics)
    output = ROOT / f"results/document-{args.label}-{args.direction}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("translation", "metrics")}))
    print(json.dumps({k: v for k, v in runs[0].items() if k != "output_ids"}))


if __name__ == "__main__":
    main()
