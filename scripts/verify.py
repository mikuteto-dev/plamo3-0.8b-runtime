"""Compare native tokenization and BF16 logits against PFN's published implementation."""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from reference import ROOT, load, prefill, prompt
from bench import CASES

BINARY = ROOT / ".build/release/plamo"


def native(*args, packed=False):
    return subprocess.run([str(BINARY), *(["--packed"] if packed else []), *args], cwd=ROOT, check=True, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    def run_native(*values):
        return native(*values, packed=args.packed)
    model, tokenizer = load()
    cases = ["", "Hello, world!", "会議は来週の月曜日です。", " a  b    c",
             "aaaaaa\n\n\n\n", "🧪𠮷\U0010ffff", "é e\u0301\r\n\t",
             "<|plamo:tag|>input<|plamo:msg|>hello", "Привет мир مرحبا بالعالم",
             " " * 20, "..." * 20, "0" * 100,
             "<|plamo:tag|>\u0301x", "a    \u0301b", "\u0301" * 4, "👩‍💻" * 4,
             "\r" * 4 + "a", "\u2028" * 4 + "b", "<|plamo:\u001c|>x", "\uee00\u0301x"]
    rng = random.Random(42)
    alphabet = list("a b日本語é\n\t!?") + ["🧪", "𠮷", "\u0301", "\u00a0"]
    cases += ["".join(rng.choices(alphabet, k=rng.randrange(1, 100))) for _ in range(24)]
    for text in cases:
        actual = json.loads(run_native("--tokenize", text).stdout)
        expected = tokenizer.encode(text, add_special_tokens=False)
        assert actual == expected, (repr(text), actual, expected)
    print(json.dumps({"tokenizer_cases": len(cases), "matched": True}), flush=True)

    texts = [
        ("会議は来週の月曜日に変更されました。", "en"),
        ("The meeting has been rescheduled for next Monday.", "ja"),
        ("This system is convenient.", "ja"),
        ("起動前に設定ファイルを確認してください。エラーがある場合は処理を中止します。", "en"),
        ("Do not delete the original file until the backup has been verified.", "ja"),
    ]
    with tempfile.TemporaryDirectory(prefix="plamo-check-") as tmp:
        for i, (text, target) in enumerate(texts):
            tokens = tokenizer.encode(prompt(text, target))
            cache = model.make_cache()
            expected = np.array(prefill(model, tokens, cache).tolist(), dtype=np.float32)
            path = Path(tmp) / f"{i}.f32"
            run = run_native("--tokens", ",".join(map(str, tokens)), "--max-tokens", "1", "--dump-logits", str(path))
            actual = np.fromfile(path, dtype="<f4")
            assert actual.shape == expected.shape and np.isfinite(actual).all()
            rmse = float(np.sqrt(np.mean((actual-expected)**2)))
            cosine = float(np.dot(actual, expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))
            assert actual.argmax() == expected.argmax(), (text, actual.argmax(), expected.argmax())
            assert rmse < 0.2 and cosine > 0.999, (text, rmse, cosine)
            print(json.dumps(dict(case=i, next_token=int(actual.argmax()), rmse=rmse, cosine=cosine,
                                  max_error=float(np.max(np.abs(actual-expected))))), flush=True)
    for name, target, text in CASES:
        tokens = tokenizer.encode(prompt(text, target))
        cache = model.make_cache()
        logits = prefill(model, tokens, cache)
        next_id = int(mx.argmax(logits).item())
        expected = []
        for _ in range(128):
            if next_id in (1, 16):
                break
            expected.append(next_id)
            logits = model(mx.array([[next_id]]), cache=cache)
            next_id = int(mx.argmax(logits.reshape(-1)).item())
        else:
            raise AssertionError("Reference generation did not stop")
        run = run_native("--to", target, "--max-tokens", "128", text)
        metrics = json.loads(run.stderr)["runs"][0]
        assert metrics["stop_reason"] == "stop"
        assert metrics["output_ids"] == expected, (name, metrics["output_ids"], expected)
        print(json.dumps(dict(generation_case=name, output_tokens=len(expected), matched=True)), flush=True)
    print("PASS")


if __name__ == "__main__":
    main()
