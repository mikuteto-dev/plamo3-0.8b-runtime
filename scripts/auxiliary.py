"""Trainable Qwen-style block indexers and a lightweight two-step MTP adapter."""
import math
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

TEACHER_SHA = "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5"


def save_weights(path, weights, metadata):
    # MLX does not pad its JSON header; Metal buffer offsets must be aligned.
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp.safetensors")
    aligned = path.with_name(path.stem+".aligned.safetensors")
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in weights.values()):
        raise ValueError("Adapter weights are not finite")
    try:
        mx.save_safetensors(str(temporary), weights, metadata=metadata)
        with temporary.open("rb") as source:
            count = struct.unpack("<Q", source.read(8))[0]
            header = source.read(count)
            size = (count+8+63)//64*64-8
            with aligned.open("wb") as output:
                output.write(struct.pack("<Q", size)); output.write(header); output.write(b" "*(size-count))
                while chunk := source.read(4*1024*1024):
                    output.write(chunk)
        aligned.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        aligned.unlink(missing_ok=True)


def norm(x, weight):
    value = x.astype(mx.float32)
    value *= mx.rsqrt(mx.mean(value*value, axis=-1, keepdims=True)+1e-6)
    return value.astype(mx.bfloat16) * (weight+1).astype(mx.bfloat16)


def rope(x, positions, frequencies):
    phase = positions.astype(mx.float32)[..., None]*frequencies
    phase = mx.concatenate([phase, phase], axis=-1)
    c = mx.cos(phase).astype(mx.bfloat16)*1.5545177444479562
    s = mx.sin(phase).astype(mx.bfloat16)*1.5545177444479562
    other = mx.concatenate([-x[..., 64:], x[..., :64]], axis=-1)
    return x*c + other*s


class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1536, 640, bias=False)
        self.q_norm = mx.zeros(128)
        self.k_norm = mx.zeros(128)

    def __call__(self, hidden, positions, frequencies):
        projected = self.proj(hidden).astype(mx.bfloat16)
        query = norm(projected[positions, :512].reshape(-1, 4, 128), self.q_norm)
        query = rope(query, positions[:, None], frequencies)
        blocks = hidden.shape[0]//4
        keys = projected[:blocks*4, 512:].astype(mx.float32).reshape(blocks, 4, 128).mean(axis=1).astype(mx.bfloat16)
        keys = rope(norm(keys, self.k_norm), mx.arange(blocks)*4, frequencies)
        scores = mx.einsum("qhd,bd->qhb", query.astype(mx.float32), keys.astype(mx.float32))
        scores = mx.maximum(scores, 0).sum(axis=1)/math.sqrt(128)
        visible = (mx.arange(blocks)[None, :]+1)*4 <= positions[:, None]+1
        return mx.where(visible, scores, -1e9)


class Future(nn.Module):
    def __init__(self, rank=256):
        super().__init__()
        self.up = nn.Linear(3072, rank, bias=False)
        self.down = nn.Linear(rank, 1536, bias=False)
        self.down.weight = mx.zeros_like(self.down.weight)

    def __call__(self, x):
        projected = self.up(x).astype(mx.bfloat16)
        activated = nn.silu(projected.astype(mx.float32)).astype(mx.bfloat16)
        return (x[..., :1536]+self.down(activated).astype(mx.bfloat16)).astype(mx.bfloat16)


class MTP(nn.Module):
    def __init__(self, rank=256):
        super().__init__()
        self.heads = [Future(rank), Future(rank)]

    def __call__(self, x):
        return mx.stack([head(x) for head in self.heads], axis=1)


def acceptance(drafts, verified):
    """verified[i] predicts the token after candidate[i]; candidate[0] is exact."""
    if len(verified) < len(drafts):
        raise ValueError("Missing verification predictions")
    accepted = 1
    for i, draft in enumerate(drafts):
        if draft != verified[i]:
            break
        accepted += 1
    return accepted


def select_blocks(scores, visible, budget):
    """Deterministic reference: score descending, block index ascending for ties."""
    if budget < 4 or budget % 4:
        raise ValueError("QSA budget must be a positive multiple of four")
    if visible < 0 or len(scores) < visible//4:
        raise ValueError("Incomplete visible scores")
    complete, tail = divmod(visible, 4)
    if any(not math.isfinite(s) for s in scores[:complete]):
        raise ValueError("Non-finite visible scores")
    chosen = sorted(range(complete), key=lambda i: (-scores[i], i))[:budget//4]
    tokens = [i*4+j for i in sorted(chosen) for j in range(4)]
    return tokens+list(range(complete*4, complete*4+tail))
