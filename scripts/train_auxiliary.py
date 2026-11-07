"""Local pilot training: frozen PLaMo teacher, Tatoeba train/validation partitions.

Only the block indexers and two MTP heads are updated. WMT24++ is never trained on.
"""
import argparse
import hashlib
import json
import random
import time
import zipfile
from urllib.request import urlopen

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from auxiliary import Indexer, MTP, TEACHER_SHA, rope, save_weights
from reference import ROOT, MODEL, load

DATA_SHA = "cb8d6b684865296037787da80bcdb5f314327e4e4c08dabaf315734f9ef747c4"
FEATURES = ROOT / "results/auxiliary-features.safetensors"


def pairs(tokenizer):
    path = ROOT / "results/data/tatoeba-en-ja.zip"
    if not path.exists():
        url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2021-07-22/moses/en-ja.txt.zip"
        with urlopen(url, timeout=60) as response:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.read(20_000_001))
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("Tatoeba checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        english = archive.read("Tatoeba.en-ja.en").decode().splitlines()
        japanese = archive.read("Tatoeba.en-ja.ja").decode().splitlines()
        (path.parent / "Tatoeba-LICENSE.txt").write_bytes(archive.read("LICENSE"))
    order = list(range(len(english))); random.Random(42).shuffle(order)
    train, valid, seen_en, seen_ja = [], [], set(), set()
    for i in order:
        en, ja = english[i].strip(), japanese[i].strip()
        if not en or not ja or len(en) > 240 or len(ja) > 240 or en in seen_en or ja in seen_ja:
            continue
        if "<|plamo:" in en+ja or "\uee00" in en+ja:
            continue
        if not (12 <= len(tokenizer.encode(en, add_special_tokens=False)) <= 64
                and 12 <= len(tokenizer.encode(ja, add_special_tokens=False)) <= 64):
            continue
        split = int.from_bytes(hashlib.sha256(en.encode()).digest()[:4], "little") % 10
        target = valid if split == 0 else train
        if len(target) >= (512 if split == 0 else 1024):
            continue
        target.append((i, en, ja)); seen_en.add(en); seen_ja.add(ja)
        if len(train) >= 1024 and len(valid) >= 512:
            break
    assert {r[1] for r in train}.isdisjoint({r[1] for r in valid})
    assert {r[2] for r in train}.isdisjoint({r[2] for r in valid})
    return train, valid


def translation_prompt(text, target):
    from reference import prompt
    return prompt(text, target)


def mtp_features(model, tokenizer, examples):
    inputs, labels, targets = [], [], []
    for number, (_, en, ja) in enumerate(examples):
        text, target = (en, "ja") if number % 2 == 0 else (ja, "en")
        ids = tokenizer.encode(translation_prompt(text, target))
        cache = model.make_cache()
        h = None
        for start in range(0, len(ids), 16):
            h = model.model(mx.array([ids[start:start+16]]), cache=cache)[:, -1:, :]
            mx.eval(h, [c.state for c in cache])
        states, tokens = [], []
        for _ in range(64):
            token = int(mx.argmax(model.lm_head(h).reshape(-1)).item())
            states.append(mx.stop_gradient(h.reshape(1536)))
            tokens.append(token)
            if token in (1, 16):
                break
            h = model.model(mx.array([[token]]), cache=cache)
            mx.eval(h, [c.state for c in cache])
        if len(states) >= 3:
            embeddings = model.model.embed_tokens(mx.array(tokens[:-2]))
            states = mx.stack(states)
            x = mx.concatenate([states[:-2], embeddings], axis=-1)
            y = mx.array(list(zip(tokens[1:-1], tokens[2:])), dtype=mx.int32)
            future = mx.stack([states[1:-1], states[2:]], axis=1)
            mx.eval(x, y, future)
            inputs.append(x); labels.append(y); targets.append(future)
        mx.clear_cache()
        if (number+1) % 16 == 0:
            print(json.dumps(dict(stage="mtp_features", sentences=number+1)), flush=True)
    if not inputs:
        raise ValueError("No MTP training features")
    return dict(x=mx.concatenate(inputs), y=mx.concatenate(labels), hidden=mx.concatenate(targets))


def history(tokenizer, examples, length, target):
    ids = [1] + tokenizer.encode("<|plamo:tag|>dataset<|plamo:msg|>translation", add_special_tokens=False)
    positions, used = [], 0
    for _, en, ja in examples:
        source, reference = (en, ja) if target == "ja" else (ja, en)
        prefix = ("<|plamo:tag|>input<|plamo:msg|>"+source
                  +"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>"+target+"<|plamo:msg|>")
        ids += tokenizer.encode(prefix, add_special_tokens=False)
        begin = len(ids)-1
        ids += tokenizer.encode(reference, add_special_tokens=False)
        positions += list(range(begin, len(ids)-1))
        used += 1
        if len(ids) >= length:
            break
    if len(ids) < length:
        raise ValueError("Not enough disjoint training pairs for a long context")
    positions = [p for p in positions if 31 <= p < length]
    random.Random(42+used).shuffle(positions)
    return ids[:length], sorted(positions[:96]), used


def teacher_distribution(layer, hidden, positions, frequencies):
    attn = layer.mixer
    q = (hidden[positions] @ attn.qkv_proj.weight[:2048].T).astype(mx.bfloat16)
    k = (hidden @ attn.qkv_proj.weight[2048:2176].T).astype(mx.bfloat16)
    q = rope(attn.q_norm(q.reshape(-1, 16, 128)), positions[:, None], frequencies)
    k = rope(attn.k_norm(k), mx.arange(hidden.shape[0]), frequencies)
    scores = mx.einsum("qhd,td->qht", q.astype(mx.float32), k.astype(mx.float32))*(128**-0.5)
    visible = mx.arange(hidden.shape[0])[None, :] <= positions[:, None]
    probabilities = mx.softmax(mx.where(visible[:, None, :], scores, -1e9), axis=-1).mean(axis=1)
    blocks = probabilities.reshape(len(positions), -1, 4).max(axis=-1)
    complete = (mx.arange(blocks.shape[-1])[None, :]+1)*4 <= positions[:, None]+1
    blocks = mx.where(complete, blocks, 0)
    return mx.stop_gradient(blocks / mx.maximum(blocks.sum(axis=-1, keepdims=True), 1e-12))


def qsa_features(model, tokenizer, examples, count, length, frequencies):
    features = {}
    consumed = 0
    for number in range(count):
        ids, query_positions, used = history(tokenizer, examples[consumed:], length, "ja" if number % 2 == 0 else "en")
        consumed += used
        x = model.model.embed_tokens(mx.array([ids]))
        qp = mx.arange(length)[:, None]; kp = mx.arange(length)[None, :]
        local_mask = (kp <= qp) & (kp > qp-2048)
        for i, layer in enumerate(model.layers):
            if i in (7, 15):
                hidden = mx.stop_gradient(layer.pre_mixer_norm(x).reshape(length, 1536))
                positions = mx.array(query_positions, dtype=mx.int32)
                teacher = teacher_distribution(layer, hidden, positions, frequencies)
                mx.eval(hidden, positions, teacher)
                for key, value in dict(hidden=hidden, positions=positions, teacher=teacher).items():
                    features[f"{number}.{i}.{key}"] = value
            x = layer(x, mask="causal" if layer.full_attn else local_mask)
            mx.eval(x)
            mx.clear_cache()
        print(json.dumps(dict(stage="qsa_features", context=number+1, tokens=length)), flush=True)
    return features


def qsa_loss(module, hidden, positions, teacher, frequencies):
    scores = module(hidden, positions, frequencies)
    log_prediction = scores-mx.logsumexp(scores, axis=-1, keepdims=True)
    return mx.mean(mx.sum(teacher*(mx.log(mx.maximum(teacher, 1e-12))-log_prediction), axis=-1))


def qsa_metrics(module, features, layer, frequencies):
    losses, recalls, active_mass, active_counts = [], [], [], []
    contexts = len(features)//6
    for i in range(contexts):
        h, p, t = (features[f"{i}.{layer}.{key}"] for key in ("hidden", "positions", "teacher"))
        scores = module(h, p, frequencies)
        k = min(512, scores.shape[-1])
        chosen = mx.argpartition(-scores, k-1, axis=-1)[:, :k]
        row_mass = mx.take_along_axis(t, chosen, axis=-1).sum(axis=-1)
        mass = row_mass.mean()
        active = (p+1)//4 > k
        active_sum = mx.sum(mx.where(active, row_mass, 0))
        active_count = mx.sum(active)
        loss = qsa_loss(module, h, p, t, frequencies)
        mx.eval(mass, loss, active_sum, active_count)
        losses.append(float(loss.item())); recalls.append(float(mass.item()))
        active_mass.append(float(active_sum.item())); active_counts.append(int(active_count.item()))
    count = sum(active_counts)
    return dict(kl=sum(losses)/len(losses), retained_teacher_mass=sum(recalls)/len(recalls),
                sparse_queries=count, sparse_retained_teacher_mass=sum(active_mass)/count if count else None)


def mtp_loss(module, x, y, targets, output_weight):
    hidden = module(x)
    logits = (hidden @ output_weight.T).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, y, reduction="mean")
    mse = mx.mean((hidden.astype(mx.float32)-targets.astype(mx.float32))**2)
    return ce+0.05*mse


def mtp_metrics(module, features, output_weight):
    matches = []
    for start in range(0, features["x"].shape[0], 64):
        prediction = mx.argmax(module(features["x"][start:start+64]) @ output_weight.T, axis=-1)
        matches.append(prediction == features["y"][start:start+64])
    match = mx.concatenate(matches)
    first = match[:, 0].astype(mx.float32).mean()
    both = mx.all(match, axis=-1).astype(mx.float32).mean()
    mx.eval(first, both)
    return dict(first_token_agreement=float(first.item()), two_token_prefix_agreement=float(both.item()),
                mean_extra_accepted_tokens=float((first+both).item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train-sentences", type=int, default=64)
    parser.add_argument("--valid-sentences", type=int, default=16)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=3072)
    parser.add_argument("--qsa-steps", type=int, default=128)
    parser.add_argument("--mtp-steps", type=int, default=256)
    args = parser.parse_args()
    if not (1 <= args.train_sentences <= 256 and 1 <= args.valid_sentences <= 64 and 1 <= args.contexts <= 8
            and 512 <= args.context_length <= 8192 and args.context_length % 4 == 0
            and 1 <= args.qsa_steps <= 4096 and 1 <= args.mtp_steps <= 4096):
        parser.error("Training size out of bounded pilot range")
    mx.random.seed(42)
    started = time.perf_counter()
    model, tokenizer = load()
    model.freeze()
    frequencies = mx.array(model.config.plamo_rope_inv_freq["full"], dtype=mx.float32)
    settings = dict(feature_version=2, min_sentence_tokens=12,
                    train_sentences=args.train_sentences, valid_sentences=args.valid_sentences,
                    contexts=args.contexts, context_length=args.context_length, teacher_sha256=TEACHER_SHA, data_sha256=DATA_SHA)
    manifest_path = FEATURES.with_suffix(".json")
    if args.collect:
        train_pairs, valid_pairs = pairs(tokenizer)
        arrays = {}
        for split, examples, count in [("train", train_pairs, args.train_sentences), ("valid", valid_pairs, args.valid_sentences)]:
            for key, value in mtp_features(model, tokenizer, examples[:count]).items():
                arrays[f"mtp.{split}.{key}"] = value
            contexts = args.contexts if split == "train" else 2
            for key, value in qsa_features(model, tokenizer, examples[count:], contexts, args.context_length, frequencies).items():
                arrays[f"qsa.{split}.{key}"] = value
        mx.eval(arrays)
        mx.save_safetensors(str(FEATURES), arrays)
        manifest_path.write_text(json.dumps(settings, indent=2)+"\n")
    elif not FEATURES.exists() or json.loads(manifest_path.read_text()) != settings:
        raise ValueError("Collect matching training features first with --collect")
    arrays = mx.load(str(FEATURES))
    qsa_data = {split: {k.removeprefix(f"qsa.{split}."): v for k, v in arrays.items() if k.startswith(f"qsa.{split}.")}
                for split in ("train", "valid")}
    mtp_data = {split: {key: arrays[f"mtp.{split}.{key}"] for key in ("x", "y", "hidden")} for split in ("train", "valid")}
    weights, metrics = {}, {}
    for layer_id in (7, 15):
        indexer = Indexer()
        attention = model.layers[layer_id].mixer
        indexer.proj.weight = mx.concatenate([attention.qkv_proj.weight[i*128:(i+1)*128] for i in (0,4,8,12)]
                                             + [attention.qkv_proj.weight[2048:2176]], axis=0)
        indexer.q_norm = attention.q_norm.weight
        indexer.k_norm = attention.k_norm.weight
        indexer.set_dtype(mx.bfloat16)
        before = qsa_metrics(indexer, qsa_data["valid"], layer_id, frequencies)
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0)
        loss_and_grad = nn.value_and_grad(indexer, qsa_loss)
        for step in range(args.qsa_steps):
            i = step % args.contexts
            h, p, t = (qsa_data["train"][f"{i}.{layer_id}.{key}"] for key in ("hidden", "positions", "teacher"))
            loss, grads = loss_and_grad(indexer, h, p, t, frequencies)
            optimizer.update(indexer, grads)
            mx.eval(indexer.parameters(), optimizer.state, loss)
            if step % 32 == 0:
                print(json.dumps(dict(stage="qsa_train", layer=layer_id, step=step, loss=float(loss.item()))), flush=True)
        after = qsa_metrics(indexer, qsa_data["valid"], layer_id, frequencies)
        metrics[f"qsa.{layer_id}"] = dict(before=before, after=after)
        weights[f"qsa.{layer_id}.proj.weight"] = indexer.proj.weight
        weights[f"qsa.{layer_id}.q_norm.weight"] = indexer.q_norm
        weights[f"qsa.{layer_id}.k_norm.weight"] = indexer.k_norm
        print(json.dumps(dict(stage="qsa_valid", layer=layer_id, before=before, after=after)), flush=True)
    predictor = MTP(); predictor.set_dtype(mx.bfloat16)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0)
    loss_and_grad = nn.value_and_grad(predictor, mtp_loss)
    before = mtp_metrics(predictor, mtp_data["valid"], model.lm_head.weight)
    count = mtp_data["train"]["x"].shape[0]
    for step in range(args.mtp_steps):
        indices = mx.random.randint(0, count, shape=(32,))
        data = mtp_data["train"]
        loss, grads = loss_and_grad(predictor, data["x"][indices], data["y"][indices], data["hidden"][indices], model.lm_head.weight)
        optimizer.update(predictor, grads)
        mx.eval(predictor.parameters(), optimizer.state, loss)
        if step % 32 == 0:
            print(json.dumps(dict(stage="mtp_train", step=step, loss=float(loss.item()))), flush=True)
    after = mtp_metrics(predictor, mtp_data["valid"], model.lm_head.weight)
    metrics["mtp"] = dict(before=before, after=after, train_positions=count, valid_positions=mtp_data["valid"]["x"].shape[0])
    for name, value in tree_flatten(predictor.parameters()):
        weights["mtp."+name] = value.astype(mx.bfloat16)
    mx.eval(weights)
    save_weights(MODEL / "auxiliary.safetensors", weights,
                        metadata=dict(format="plamo-aux-v1", teacher_sha256=TEACHER_SHA, rank="256", qsa_steps=str(args.qsa_steps),
                                      mtp_steps=str(args.mtp_steps), data_sha256=DATA_SHA,
                                      qsa_backbone_adaptation="false", mtp_architecture="parallel-residual-mlp-2",
                                      data_license="CC-BY-2.0-FR"))
    report = dict(settings=settings, qsa_steps=args.qsa_steps, mtp_steps=args.mtp_steps, metrics=metrics,
                  parameter_bytes=sum(a.nbytes for a in weights.values()),
                  adapter_sha256=hashlib.sha256((MODEL / "auxiliary.safetensors").read_bytes()).hexdigest(),
                  training_seconds=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory(),
                  qsa_backbone_adaptation=False, dataset_license="Tatoeba CC BY 2.0 FR")
    (MODEL / "auxiliary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
