"""Lossless BF16 packing, aligned to M2's 32 lanes x 4 weights.

A 128-value block keeps all sign/mantissa bits and four-bit exponent deltas.
The aligned offset's low six bits hold minimum-107; zero marks raw BF16.
Blocks outside that exponent range remain raw BF16, never rounded.
"""
import hashlib
import json
import struct

import numpy as np

from reference import MODEL

SOURCE_SHA = "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5"
HEADER = 65536


def pack(values):
    blocks = values.reshape(-1, 128)
    exponent = (blocks >> 7) & 255
    nonzero = (blocks & 0x7fff) != 0
    low = np.where(nonzero, exponent, 255).min(axis=1)
    high = np.where(nonzero, exponent, 0).max(axis=1)
    # This model has minima 108...123; six offset bits cover 108...170 and reserve zero for raw.
    eligible = (low >= 108) & (low <= 170) & (high < 255) & (high.astype(np.int32)-low.astype(np.int32) < 15)
    minima = np.where(eligible, low, 0).astype(np.uint8)
    sizes = np.where(eligible, 192, 256).astype(np.uint32)
    offsets = np.empty(len(blocks), dtype="<u4")
    offsets[0] = 0
    np.cumsum(sizes[:-1], out=offsets[1:])
    data = np.empty(int(sizes.sum()), dtype=np.uint8)
    for start in range(0, len(blocks), 4096):
        end = min(start+4096, len(blocks))
        src = blocks[start:end]
        flag = eligible[start:end]
        ids = np.arange(start, end)
        if flag.any():
            v = src[flag]
            codes = ((v >> 7) & 255) - minima[ids[flag], None]
            codes = np.where((v & 0x7fff) == 0, 15, codes).astype(np.uint8)
            sign_mantissa = ((v & 127) | ((v >> 8) & 128)).astype(np.uint8)
            encoded = np.concatenate([sign_mantissa, codes[:, 0::2] | (codes[:, 1::2] << 4)], axis=1)
            data[offsets[ids[flag], None] + np.arange(192)] = encoded
        if (~flag).any():
            data[offsets[ids[~flag], None] + np.arange(256)] = src[~flag].copy().view(np.uint8).reshape(-1, 256)
        restored = unpack(data, offsets[start:end], minima[start:end])
        if not np.array_equal(restored, src):
            raise AssertionError("Packing changed BF16 bits")
    return data, offsets, minima


def unpack(data, offsets, minima):
    output = np.empty((len(offsets), 128), dtype="<u2")
    packed = minima != 0
    if packed.any():
        encoded = data[offsets[packed, None] + np.arange(192)]
        sm = encoded[:, :128].astype(np.uint16)
        nibbles = encoded[:, 128:]
        codes = np.empty((len(encoded), 128), dtype=np.uint16)
        codes[:, 0::2] = nibbles & 15
        codes[:, 1::2] = nibbles >> 4
        exp = np.where(codes == 15, 0, codes + minima[packed, None])
        output[packed] = (sm & 127) | ((sm & 128) << 8) | (exp << 7)
    if (~packed).any():
        raw = data[offsets[~packed, None] + np.arange(256)].copy()
        output[~packed] = raw.view("<u2").reshape(-1, 128)
    return output


def tag(offsets, minima):
    assert np.all((offsets & 63) == 0)
    assert np.all((minima == 0) | ((minima >= 108) & (minima <= 170)))
    return offsets | np.where(minima != 0, minima.astype(np.uint32)-107, 0)


def self_test():
    # Every possible BF16 bit pattern, including signed zero, subnormals, NaN and infinity.
    for values in [np.arange(65536, dtype="<u2"), np.zeros(256, dtype="<u2"),
                   np.full(128, 0x8000, dtype="<u2"),
                   np.tile(np.array([0, 0x8000, 0x3f80, 0xbf80], dtype="<u2"), 64)]:
        data, offsets, minima = pack(values)
        entries = tag(offsets, minima)
        codes = entries & 63
        restored = unpack(data, entries & ~np.uint32(63), np.where(codes != 0, codes+107, 0).astype(np.uint8))
        assert np.array_equal(restored.reshape(-1), values)


def main():
    self_test()
    source = MODEL / "model.safetensors"
    with source.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SOURCE_SHA:
            raise ValueError("Original weight checksum mismatch")
        f.seek(0)
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    output = MODEL / "model.p12"
    part = output.with_suffix(".p12.part")
    manifest = dict(format="plamo-bf16-pack-v2", block=128, minimum_base=107, source_sha256=SOURCE_SHA, tensors={})
    total_raw = total_packed = compressed_blocks = blocks = 0
    try:
        with part.open("w+b") as f:
            f.write(bytes(HEADER))
            def append(array):
                padding = (-f.tell()) % 64
                f.write(bytes(padding))
                begin = f.tell()-HEADER
                f.write(array.tobytes())
                return [begin, f.tell()-HEADER]
            for name, tensor in header.items():
                if name == "__metadata__" or name == "model.embed_tokens.weight":
                    continue
                assert tensor["dtype"] == "BF16"
                shape = tensor["shape"]
                begin, end = tensor["data_offsets"]
                values = np.memmap(source, dtype="<u2", mode="r", offset=8+length+begin, shape=tuple(shape))
                record = dict(shape=shape)
                if len(shape) == 1:
                    record["raw"] = append(values)
                else:
                    data, offsets, minima = pack(values)
                    record["offsets"] = append(tag(offsets, minima))
                    record["data"] = append(data)
                    compressed_blocks += int((minima != 0).sum()); blocks += len(minima)
                    total_packed += data.nbytes + offsets.nbytes
                    total_raw += values.nbytes
                    del data, offsets, minima
                manifest["tensors"][name] = record
            manifest["payload_bytes"] = f.tell()-HEADER
            metadata = json.dumps(manifest, separators=(",", ":")).encode()
            if len(metadata) > HEADER-8:
                raise ValueError("Packed header overflow")
            f.seek(0)
            f.write(struct.pack("<Q", HEADER-8)); f.write(metadata); f.write(b" "*(HEADER-8-len(metadata)))
        part.replace(output)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(original_linear_bytes=total_raw, packed_linear_bytes=total_packed,
                          archive_bytes=output.stat().st_size, compressed_blocks=compressed_blocks,
                          blocks=blocks, all_weights_bit_exact=True)))


if __name__ == "__main__":
    main()
