import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from scfuse.data import load_packed
from scfuse.prep.common import Records
from scfuse.schema import CHUNK, MAX_CHUNKS
from scfuse.utils.solidity import match_close

_CMP = {"<": ">=", ">": "<=", "<=": ">", ">=": "<"}


def _body(text: str) -> int:
    i = text.find("{")
    return i + 1 if i >= 0 else len(text)


def _sub_at(text: str, m: re.Match, new: str) -> str:
    return text[:m.start()] + new + text[m.end():]


def _statements(text: str, start: int) -> list[tuple[int, int]]:
    out, s, depth = [], start, 0
    for i in range(start, len(text)):
        c = text[i]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth < 0:
                break
        elif c == ";" and depth == 0:
            out.append((s, i + 1))
            s = i + 1
        elif c == "{" and depth == 0:
            s = i + 1
    return out


def candidates(text: str) -> list[tuple[str, callable]]:
    b, out = _body(text), []
    for m in re.finditer(r"(?<![<>=!])(<=|>=|<|>)(?![<>=])", text[b:]):
        mm = re.compile(re.escape(m.group())).match(text, b + m.start())
        out.append(("cmp_flip", lambda t, mm=mm: _sub_at(t, mm, _CMP[mm.group()])))
    for m in re.finditer(r"==|!=", text[b:]):
        mm = re.compile(re.escape(m.group())).match(text, b + m.start())
        out.append(("eq_flip", lambda t, mm=mm: _sub_at(t, mm, "!=" if mm.group() == "==" else "==")))
    for m in re.finditer(r"\b(require|assert)\s*\(", text[b:]):
        s = b + m.start()
        close = match_close(text, b + m.end() - 1)
        end = text.find(";", close)
        if end > 0:
            out.append(("drop_check", lambda t, s=s, e=end: t[:s] + t[e + 1:]))
    for m in re.finditer(r"\bmsg\.sender\b", text[b:]):
        mm = re.compile(r"msg\.sender").match(text, b + m.start())
        out.append(("sender_origin", lambda t, mm=mm: _sub_at(t, mm, "tx.origin")))
    for m in re.finditer(r"([+\-])=", text[b:]):
        mm = re.compile(r"[+\-]=").match(text, b + m.start())
        out.append(("arith_flip", lambda t, mm=mm: _sub_at(t, mm, "-=" if mm.group() == "+=" else "+=")))
    for m in re.finditer(r"\b(true|false)\b", text[b:]):
        mm = re.compile(r"true|false").match(text, b + m.start())
        out.append(("bool_flip", lambda t, mm=mm: _sub_at(t, mm, "false" if mm.group() == "true" else "true")))
    st = _statements(text, b)
    for k in range(len(st) - 1):
        (s0, e0), (s1, e1) = st[k], st[k + 1]
        if re.search(r"\.(call|send|transfer)\b", text[s0:e0]) and "=" in text[s1:e1]:
            out.append(("call_order", lambda t, s0=s0, e0=e0, s1=s1, e1=e1: t[:s0] + t[s1:e1] + t[e0:s1] + t[s0:e0]
                        + t[e1:]))
    return out


@torch.no_grad()
def encode_texts(texts: list[str], backbone: str, device: str, batch: int = 32) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device).eval()
    prefix = [tok.cls_token_id, tok.convert_tokens_to_ids("<encoder-only>"), tok.sep_token_id]
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    rows = []
    for u, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        for c in [ids[i:i + CHUNK] for i in range(0, max(len(ids), 1), CHUNK)][:MAX_CHUNKS]:
            rows.append((u, prefix + c + [tok.sep_token_id]))
    counts = np.zeros(len(texts))
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        L = max(len(x) for _, x in part)
        ids = torch.full((len(part), L), tok.pad_token_id, dtype=torch.long)
        for r, (_, x) in enumerate(part):
            ids[r, :len(x)] = torch.tensor(x)
        ids = ids.to(device)
        mask = ids != tok.pad_token_id
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            h = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state.float()
        pooled = ((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).cpu().numpy()
        for (u, _), v in zip(part, pooled):
            out[u] += v
            counts[u] += 1
    return out / np.maximum(counts[:, None], 1)


def main():
    ap = argparse.ArgumentParser(description="P5': selector-preserving semantic mutations of the lifted view (GPU)")
    ap.add_argument("--cache", default="data/scfuse")
    ap.add_argument("--feats", default="unixcoder-base")
    ap.add_argument("--backbone", default="microsoft/unixcoder-base")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cache = Path(args.cache)
    p = load_packed(cache)
    units = Records(cache / "units_D.jsonl")
    meta = p.meta
    manifest, texts = [], []
    for row in np.flatnonzero(meta.split.isin(["gold_val", "gold_test"]).to_numpy() & (meta.m_D.to_numpy() > 0)):
        rec = units.get(meta.id.iloc[row]) or {"units": []}
        us = rec["units"][:int(meta.D_unit_n.iloc[row])]
        for d in range(args.draws):
            rng = np.random.default_rng([args.seed, int(row), d])
            cands = [(u, op, fn) for u, x in enumerate(us) if x["sel"] is not None for op, fn in candidates(x["text"])]
            entry = {"id": meta.id.iloc[row], "row": int(row), "draw": d, "valid": bool(cands)}
            if cands:
                u, op, fn = cands[int(rng.integers(len(cands)))]
                new = fn(us[u]["text"])
                entry.update(unit=u, op=op, feat_row=len(texts), delta_chars=len(new) - len(us[u]["text"]))
                texts.append(new)
            manifest.append(entry)
    out_dir = cache / "feats" / args.feats
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = encode_texts(texts, args.backbone, args.device) if texts else np.zeros((0, 768), np.float32)
    arr = open_memmap(out_dir / "D_mut.npy", mode="w+", dtype=np.float16, shape=feats.shape)
    arr[:] = feats.astype(np.float16)
    arr.flush()
    json.dump(manifest, open(cache / "mutations.json", "w"))
    ops = {}
    for e in manifest:
        ops[e.get("op", "none")] = ops.get(e.get("op", "none"), 0) + 1
    print(f"{len(manifest)} records, {sum(e['valid'] for e in manifest)} valid; operators {ops}")


if __name__ == "__main__":
    main()
