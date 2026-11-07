"""Check cropped-cache positions against the HF algorithm using PFN's MLX layers."""
import argparse
import hashlib
import inspect
import json
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from reference import ROOT, MODEL, load, load_tokenizer


def source_prefill(model, tokens):
    caches = [None] * 16
    for start in range(0, len(tokens), 16):
        ids = tokens[start:start+16]
        x = model.model.embed_tokens(mx.array([ids]))
        length = len(ids)
        for i, layer in enumerate(model.layers):
            attn = layer.mixer
            h = layer.pre_mixer_norm(x)
            qkv = attn.qkv_proj(h).astype(h.dtype)
            q, k, v = mx.split(qkv, [2048, 2176], axis=-1)
            q = attn.q_norm(q.reshape(1, length, 16, 128).transpose(0, 2, 1, 3))
            k = attn.k_norm(k.reshape(1, length, 1, 128).transpose(0, 2, 1, 3))
            v = v.reshape(1, length, 1, 128).transpose(0, 2, 1, 3)
            if caches[i] is not None:
                old_k, old_v = caches[i]
                k = mx.concatenate([old_k, k], axis=2)
                v = mx.concatenate([old_v, v], axis=2)
            size = k.shape[2]
            # HF caches normalized, unrotated keys, crops to window_size-1,
            # then assigns local coordinates to the fetched sequence.
            caches[i] = (k, v) if layer.full_attn else (k[:, :, -2047:], v[:, :, -2047:])
            q = attn.rope(q, offset=size-length)
            rotated = attn.rope(k)
            qp = mx.arange(size-length, size)[:, None]
            kp = mx.arange(size)[None, :]
            mask = kp <= qp
            if not layer.full_attn:
                mask = mask & (kp > qp-2048)
            result = mx.fast.scaled_dot_product_attention(q, rotated, v, scale=128**-0.5, mask=mask)
            result = result.transpose(0, 2, 1, 3).reshape(1, length, 2048)
            projected = attn.o_proj(result).astype(h.dtype)
            x = x + layer.post_mixer_norm(projected)
            x = x + layer.post_mlp_norm(layer.mlp(layer.pre_mlp_norm(x)))
        mx.eval(x, caches)
        mx.clear_cache()
        if (start+length)%1024 == 0 or start+length == len(tokens):
            print(json.dumps(dict(stage="reference", processed_tokens=start+length, total_tokens=len(tokens))), file=sys.stderr, flush=True)
    logits = model.lm_head(model.model.norm(x[:, -1:, :])).astype(mx.float32).reshape(-1)
    mx.eval(logits)
    return np.array(logits.tolist(), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--lengths", nargs="+", type=int, default=[2049, 4097])
    parser.add_argument("--reuse-reference", action="store_true")
    args = parser.parse_args()
    if not all(1 <= n <= 32768 for n in args.lengths):
        parser.error("Input length must be 1...32768")
    model, tokenizer = (None, load_tokenizer()) if args.reuse_reference else load()
    sentence = tokenizer.encode("Check the configuration before starting the system. ", add_special_tokens=False)
    with (MODEL / "model.safetensors").open("rb") as source:
        teacher_sha = hashlib.file_digest(source,"sha256").hexdigest()
    implementation_sha = hashlib.sha256(inspect.getsource(source_prefill).encode()+(MODEL / "modeling_plamo_mlx.py").read_bytes()
                                        +(MODEL / "config-mlx.json").read_bytes()).hexdigest()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plamo-long-") as tmp:
        for length in args.lengths:
            tokens = ([1] + sentence * (length//len(sentence)+1))[:length]
            prefix = output / f"reference-long-{length}"
            ref_path = prefix.with_suffix(".f32")
            meta_path = prefix.with_suffix(".json")
            identity = dict(tokens_sha256=hashlib.sha256(np.array(tokens,dtype="<i4").tobytes()).hexdigest(),
                            teacher_sha256=teacher_sha, implementation_sha256=implementation_sha)
            if args.reuse_reference:
                metadata = json.loads(meta_path.read_text())
                assert metadata["identity"] == identity, "Reference cache is stale"
                assert metadata["sha256"] == hashlib.sha256(ref_path.read_bytes()).hexdigest(), "Reference cache is corrupt"
                expected = np.fromfile(ref_path,dtype="<f4")
            else:
                started = time.perf_counter()
                expected = source_prefill(model, tokens)
                expected.astype("<f4").tofile(ref_path)
                metadata = dict(identity=identity, sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                                reference_seconds=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory())
                meta_path.write_text(json.dumps(metadata,indent=2)+"\n")
            assert expected.shape == (107520,) and np.isfinite(expected).all()
            (output / f"reference-long-{length}.tokens.json").write_text(json.dumps(tokens)+"\n")
            path = Path(tmp) / f"{length}.f32"
            run = subprocess.run([str(ROOT / ".build/release/plamo"), *(["--packed"] if args.packed else []), "--tokens", ",".join(map(str, tokens)),
                                  "--max-tokens", "0", "--dump-logits", str(path)],
                                 cwd=ROOT, text=True, capture_output=True, check=True)
            actual = np.fromfile(path, dtype="<f4")
            assert actual.shape == expected.shape and np.isfinite(actual).all()
            actual.tofile(output / f"native-long-{'packed' if args.packed else 'raw'}-{length}.f32")
            rmse = float(np.sqrt(np.mean((actual-expected)**2)))
            cosine = float(np.dot(actual, expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))
            metrics = json.loads(run.stderr)
            assert metrics["runs"][0]["input_tokens"] == length and metrics["runs"][0]["stop_reason"] == "prefill"
            matched = actual.argmax() == expected.argmax()
            result = dict(input_tokens=length, rmse=rmse, cosine=cosine, matched=bool(matched),
                          next_token=int(actual.argmax()), reference_next_token=int(expected.argmax()),
                          max_error=float(np.max(np.abs(actual-expected))),
                          native_prefill_seconds=metrics["runs"][0]["prefill_seconds"],
                          reference=metadata, native=metrics,
                          passed=bool(matched and rmse < 0.2 and cosine > 0.999))
            (output / f"verify-long-{'packed' if args.packed else 'raw'}-{length}.json").write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps({k:v for k,v in result.items() if k not in ("native","reference")}), flush=True)
            assert result["passed"], (length, result)
    print("PASS")


if __name__ == "__main__":
    main()
