"""Small, held-out WMT24++ pilot for ShortGPT-style pruning and draft agreement.

This changes only the in-memory research model, never the original weight file.
Teacher-forced agreement is a screening proxy, not speculative acceptance rate.
"""
import argparse
import json
import random
import struct
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from sacrebleu.metrics import CHRF

from reference import ROOT, MODEL, load, prefill, prompt
from wmt import rows


def cases(tokenizer, pairs):
    values = []
    for row in pairs:
        for source, reference, target in [(row["source"], row["target"], "ja"),
                                           (row["target"], row["source"], "en")]:
            prefix = tokenizer.encode(prompt(source, target))
            suffix = tokenizer.encode(reference, add_special_tokens=False) + [16]
            values.append(dict(prefix=prefix, suffix=suffix, source=source, reference=reference,
                               target=target, id=row["segment_id"], domain=row["domain"]))
    return values


def influence(model, examples):
    total = [0.0] * 16
    count = 0
    for example in examples:
        ids = example["prefix"] + example["suffix"][:-1]
        x = model.model.embed_tokens(mx.array([ids]))
        full_mask = create_attention_mask(x, None)
        local_mask = create_attention_mask(x, None, window_size=2048)
        metrics = []
        for layer in model.layers:
            y = layer(x, mask=full_mask if layer.full_attn else local_mask)
            a, b = x.astype(mx.float32), y.astype(mx.float32)
            cosine = mx.sum(a*b, axis=-1) / mx.maximum(mx.sqrt(mx.sum(a*a, axis=-1)*mx.sum(b*b, axis=-1)), 1e-12)
            metrics.append(mx.mean(1-cosine))
            x = y
        mx.eval(metrics, x)
        for i, value in enumerate(metrics):
            total[i] += float(value.item()) * len(ids)
        count += len(ids)
        mx.clear_cache()
    return [value/count for value in total]


def configure(model, original, skipped):
    decoder = model.model.layers
    decoder.layers = [layer for i, layer in enumerate(original) if i not in skipped]
    decoder.full_idx = next(i for i, layer in enumerate(decoder.layers) if layer.full_attn)
    decoder.swa_idx = next(i for i, layer in enumerate(decoder.layers) if not layer.full_attn)


def score(model, example):
    prefix, suffix = example["prefix"], example["suffix"]
    ids = prefix + suffix[:-1]
    hidden = model.model(mx.array([ids]))[:, len(prefix)-1:, :]
    logits = model.lm_head(hidden).astype(mx.float32)
    labels = mx.array([suffix])
    nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
    top = mx.argmax(logits, axis=-1)
    mx.eval(nll, top)
    result = (float(mx.sum(nll).item()), top.reshape(-1).tolist())
    mx.clear_cache()
    return result


def generate(model, tokenizer, example):
    cache = model.make_cache()
    start = time.perf_counter()
    next_id = int(mx.argmax(prefill(model, example["prefix"], cache)).item())
    output = []
    for _ in range(128):
        if next_id in (1, 16):
            return dict(text=tokenizer.decode(output), ids=output, seconds=time.perf_counter()-start, stopped=True)
        output.append(next_id)
        next_id = int(mx.argmax(model(mx.array([[next_id]]), cache=cache).reshape(-1)).item())
    return dict(text=tokenizer.decode(output), ids=output, seconds=time.perf_counter()-start, stopped=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=int, default=4)
    parser.add_argument("--evaluate", type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.calibration <= 32 and 1 <= args.evaluate <= 32):
        parser.error("Use 1...32 pairs per split")
    model, tokenizer = load()
    data = []
    for row in rows():
        examples = cases(tokenizer, [row])
        if all(16 <= len(e["prefix"]) <= 128 and 8 <= len(e["suffix"]) <= 72 for e in examples):
            data.append(row)
    random.Random(42).shuffle(data)
    calibration = cases(tokenizer, data[:args.calibration])
    evaluation = cases(tokenizer, data[args.calibration:args.calibration+args.evaluate])
    assert {e["id"] for e in calibration}.isdisjoint({e["id"] for e in evaluation})
    original = list(model.layers)
    bi = influence(model, calibration)
    eligible = sorted([i for i in range(1, 15) if i != 7], key=lambda i: bi[i])
    print(json.dumps(dict(block_influence=bi, skip_order=eligible)), flush=True)
    with (MODEL / "model.safetensors").open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    block_bytes = [sum(v["data_offsets"][1]-v["data_offsets"][0] for k, v in header.items()
                       if k.startswith(f"model.layers.layers.{i}.")) for i in range(16)]
    token_count = sum(len(e["suffix"]) for e in evaluation)
    baseline_top = None
    summaries = []
    for count in [0, 1, 2, 4, 8]:
        skipped = set(eligible[:count])
        configure(model, original, skipped)
        scored = [score(model, example) for example in evaluation]
        top = [t for _, values in scored for t in values]
        if baseline_top is None:
            baseline_top = top
        summary = dict(skipped=sorted(skipped), nll=sum(x[0] for x in scored)/token_count,
                       teacher_forced_top1_agreement=sum(a == b for a, b in zip(top, baseline_top))/token_count,
                       removed_weight_bytes=sum(block_bytes[i] for i in skipped))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    generations = []
    for count in [0, 1]:
        configure(model, original, set(eligible[:count]))
        outputs = [generate(model, tokenizer, example) for example in evaluation]
        by_direction = {}
        for target in ["ja", "en"]:
            selected = [(e, out) for e, out in zip(evaluation, outputs) if e["target"] == target]
            by_direction[target] = CHRF().corpus_score([o["text"] for _, o in selected], [[e["reference"] for e, _ in selected]]).score
        report = dict(skipped=eligible[:count], chrf=by_direction,
                      all_stopped=all(o["stopped"] for o in outputs),
                      mean_seconds=sum(o["seconds"] for o in outputs)/len(outputs), outputs=outputs)
        generations.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "outputs"}), flush=True)
    configure(model, original, set())
    result = dict(calibration_ids=[e["id"] for e in calibration], evaluation_ids=[e["id"] for e in evaluation],
                  evaluation_pairs=args.evaluate, block_influence=bi, skip_order=eligible,
                  teacher_forced=summaries, generations=generations)
    output = ROOT / "results/architecture-probe.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
