"""Fetch pinned public model artifacts; the runtime itself has no Python dependency."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from urllib.request import urlopen

BF16 = "cc8e88169807d5a272e8ff06db12eb0129b67f81"
MLX = "e5fe5b7f26ebaeae48db0ce644b932d1d9b91aaa"
ROOT = Path(__file__).resolve().parents[1]
VOCAB_SHA = "bdce9e430e65931b28c56de2054f52b4cf8ed59b322db125e016d724d06dbb0e"
TRIE_SHA = "325c68005301172dffa3124a5ef813537d0e666a9b2c6411c4f406fd16511e53"
FILES = [
    ("model.safetensors", "bf16", "model.safetensors", "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5"),
    ("config.json", "bf16", "config.json", "b825929ff52a9b94f911df21622cde1a2edf0a71c0710ab06b0dc60c83cacbd0"),
    ("tokenizer.jsonl", "bf16", "tokenizer.jsonl", VOCAB_SHA),
    ("tokenizer_config.json", "bf16", "tokenizer_config.json", "5468ed5c68b5cf4cf971bcb34be6bfbfd749fe30cc5bc65ace5835ae3bff65b3"),
    ("config-mlx.json", "mlx", "config.json", "5175a9cd6ba7bab7eecafc508aa3278e58ece44d8edafa4a175ca067c4ed8c69"),
    ("LICENSE/en", "bf16", "LICENSE/en", "d9979039b8f1a96fa81e1be61c4550d06f8542b25119e3f765c7c011a68597fb"),
    ("LICENSE/ja", "bf16", "LICENSE/ja", "988b37b0b21ca130ad84a092429e4e5bceec5f86dfe85a8ca3e06ea703b960ca"),
]
REFERENCE = [
    ("modeling_plamo.py", "bf16", "modeling_plamo.py", "498ff5f52b565315325ffff29b6c5e628a55156a4df5cd2bacb48a9c8697f32b"),
    ("tokenization_plamo.py", "bf16", "tokenization_plamo.py", "68b769d516541603857ce9d2813e48193efe41abb64c1091311b973d491724d9"),
    ("chat_template.jinja", "bf16", "chat_template.jinja", "cd49e7e0dfa8ec847c163db88dfca89b280dddbb73ddc60ba38c62a420eaced2"),
    ("modeling_plamo_mlx.py", "mlx", "modeling_plamo_mlx.py", "eb6a576df1c20e9d40653671c8e63ded584661aa3f858ff5bee620357b479367"),
    ("LICENSE.mlx-lm", "mlx", "LICENSE.mlx-lm", "ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01"),
]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(directory, entry):
    local, kind, remote, expected = entry
    path = directory / local
    if path.exists() and sha(path) == expected:
        print(f"verified {local}", flush=True)
        return
    repo = "plamo-3-translate-0.8b" + ("-MLX-4bit" if kind == "mlx" else "")
    revision = MLX if kind == "mlx" else BF16
    url = f"https://huggingface.co/pfnet/{repo}/resolve/{revision}/{remote}"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=120) as response, part.open("wb") as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
        if sha(part) != expected:
            raise ValueError(f"SHA-256 mismatch: {local}")
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    print(f"downloaded {local}", flush=True)


def cache_tokenizer(directory):
    source = directory / "tokenizer.jsonl"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != VOCAB_SHA:
        raise ValueError("Unexpected tokenizer source")
    path = source.with_name(source.name + ".trie")
    if path.exists() and sha(path) == TRIE_SHA:
        print("verified tokenizer.jsonl.trie", flush=True)
        return
    records = [json.loads(line) for line in data.split(b"\n") if line]
    nodes, scores, roots, children = [[-1, -1, 0, -1]], [], {}, {}
    byte_ids, offsets, text = [-1] * 256, [0], bytearray()
    for token, (piece, score, kind, *_) in enumerate(records):
        text.extend(piece.encode())
        offsets.append(len(text))
        if kind == "BYTE":
            byte_ids[int(piece[3:-1], 16)] = token
            scores.append(0)
            continue
        scores.append(round(score * 1e4))
        parent = 0
        for char in piece:
            scalar = ord(char)
            key = parent, scalar
            if key not in children:
                child = len(nodes)
                nodes.append([-1, -1, scalar, nodes[parent][0] if parent else -1])
                children[key] = child
                if parent:
                    nodes[parent][0] = child
                else:
                    roots[scalar] = child
            parent = children[key]
        nodes[parent][1] = token
    if len(records) != 107520 or len(nodes) != 195624 or len(roots) != 6873 or min(byte_ids) < 0:
        raise ValueError("Unexpected tokenizer structure")
    cache = (b"PLAMO\x01\x00\x00" + b"".join(struct.pack("<iiIi", *node) for node in nodes)
             + struct.pack("<107520q", *scores)
             + b"".join(struct.pack("<II", key, value) for key, value in sorted(roots.items()))
             + struct.pack("<256I", *byte_ids) + struct.pack("<107521I", *offsets) + bytes(text))
    if hashlib.sha256(cache).hexdigest() != TRIE_SHA:
        raise ValueError("Unexpected tokenizer cache")
    part = path.with_name(path.name + ".part")
    try:
        part.write_bytes(cache)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    print("prepared tokenizer.jsonl.trie", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    for entry in FILES + (REFERENCE if args.reference else []):
        fetch(ROOT / "models/plamo", entry)
    cache_tokenizer(ROOT / "models/plamo")


if __name__ == "__main__":
    main()
