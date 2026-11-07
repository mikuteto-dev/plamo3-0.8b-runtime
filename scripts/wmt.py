"""Pinned public WMT24++ evaluation data; canary and bad sources are excluded."""
import hashlib
import json
from urllib.request import urlopen

from reference import ROOT

REVISION = "fd7405c06494bc66a57b25f55d217a72f96e60dc"
SHA256 = "58bbee69aed537d21e25f04c5eaf49c2c7aef1b1dddb918f6eac424ab93c7ec8"


def rows():
    path = ROOT / "results/data/wmt24pp-en-ja.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/google/wmt24pp/resolve/{REVISION}/en-ja_JP.jsonl"
        with urlopen(url, timeout=60) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("WMT24++ checksum mismatch")
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("WMT24++ checksum mismatch")
    values = [json.loads(line) for line in data.splitlines()]
    return [r for r in values if not r.get("is_bad_source") and r["domain"] != "canary"
            and r["source"].strip() and r["target"].strip()]
