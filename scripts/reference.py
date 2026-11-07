"""Validation-only BF16 baseline using PFN's published MLX implementation."""
import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/plamo"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def load():
    source = module("plamo_reference", MODEL / "modeling_plamo_mlx.py")
    config = json.loads((MODEL / "config-mlx.json").read_text())
    model = source.Model(source.ModelArgs.from_dict(config))
    model.load_weights(list(mx.load(str(MODEL / "model.safetensors")).items()))
    model.eval()
    mx.eval(model.parameters())
    return model, load_tokenizer()


def load_tokenizer():
    tok_source = module("plamo_tokenizer", MODEL / "tokenization_plamo.py")
    tokenizer = tok_source.Plamo3Tokenizer(
        str(MODEL / "tokenizer.jsonl"),
        **{k: v for k, v in json.loads((MODEL / "tokenizer_config.json").read_text()).items()
           if k in ("add_bos_token", "add_eos_token", "break_around_consecutive_spaces_threshold",
                    "break_around_repeated_chars_threshold")},
    )
    return tokenizer


def prompt(text, target):
    return ("<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:msg|>" + text
            + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + target + "<|plamo:msg|>")


def prefill(model, tokens, cache):
    hidden = None
    for start in range(0, len(tokens), 16):
        hidden = model.model(mx.array([tokens[start:start+16]]), cache=cache)
        mx.eval(hidden, [c.state for c in cache])
        mx.clear_cache()
    logits = model.lm_head(hidden[:, -1:, :]).astype(mx.float32)
    mx.eval(logits)
    return logits.reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--to", default="ja")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    start = time.perf_counter()
    model, tok = load()
    load_seconds = time.perf_counter() - start
    tokens = tok.encode(prompt(args.text, args.to))
    runs = []
    for _ in range(args.repeat):
        cache = model.make_cache()
        start = time.perf_counter()
        logits = prefill(model, tokens, cache)
        next_id = int(mx.argmax(logits).item())
        first = time.perf_counter()
        output = []
        stopped = False
        for i in range(args.max_tokens):
            if next_id in (1, 16):
                stopped = True
                break
            output.append(next_id)
            if i + 1 < args.max_tokens:
                logits = model(mx.array([[next_id]]), cache=cache)
                next_id = int(mx.argmax(logits.reshape(-1)).item())
        end = time.perf_counter()
        count = max(0, len(output) - (0 if stopped else 1))
        runs.append(dict(input_tokens=len(tokens), output_tokens=len(output), output_ids=output,
                         prefill_seconds=first-start, decode_seconds=end-first,
                         decode_tokens_per_second=count/max(end-first, 1e-9),
                         stop_reason="stop" if stopped else "length"))
    print(tok.decode(output))
    print(json.dumps(dict(load_seconds=load_seconds, runs=runs,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          mlx_peak_bytes=mx.get_peak_memory())), file=sys.stderr)


if __name__ == "__main__":
    main()
