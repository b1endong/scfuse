import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

from scfuse.data import load_packed
from scfuse.schema import CHUNK, MAX_CHUNKS


def global_chunks(p, view: str):
    meta = p.meta
    chunk_c = np.repeat(np.arange(len(meta)), meta[f"{view}_chunk_n"].to_numpy())
    chunks = p.arr[f"{view}_chunk"]
    tok_start = chunks[:, 0] + meta[f"{view}_tok_0"].to_numpy()[chunk_c]
    unit_c = np.repeat(np.arange(len(meta)), meta[f"{view}_unit_n"].to_numpy())
    units = p.arr[f"{view}_unit"]
    unit_start = units[:, 0] + meta[f"{view}_chunk_0"].to_numpy()[unit_c]
    return tok_start, chunks[:, 1].copy(), unit_start, units[:, 1].copy()


def span_pieces(p):
    meta = p.meta
    if "SP_span" not in p.arr or not len(p.arr["SP_span"]):
        return None
    sp = np.asarray(p.arr["SP_span"], np.int64)
    c = np.repeat(np.arange(len(meta)), meta["SP_span_n"].to_numpy())
    unit = sp[:, 0] + meta["S_unit_0"].to_numpy()[c]
    units = np.asarray(p.arr["S_unit"])
    first = units[unit, 0] + meta["S_chunk_0"].to_numpy()[c]
    lens = np.asarray(p.arr["S_chunk"][:, 1])
    out = []
    for k in range(MAX_CHUNKS):
        ok = k < units[unit, 1]
        g = np.where(ok, first + k, 0)
        a = np.maximum(sp[:, 1], k * CHUNK)
        b = np.minimum(sp[:, 2], k * CHUNK + lens[g])
        ok &= a < b
        out.append(np.stack([g[ok], a[ok] - k * CHUNK, b[ok] - k * CHUNK, np.flatnonzero(ok)], 1))
    pieces = np.concatenate(out)
    return pieces[np.argsort(pieces[:, 0], kind="stable")], len(sp)


def _bucket(n: int, step: int = 64) -> int:
    return -(-n // step) * step


@torch.no_grad()
def encode_view(p, view: str, out_dir: Path, backbone: str, batch_tokens: int, device: str,
                source_tokenizer: str = "microsoft/unixcoder-base"):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone)
    enc_only = tok.convert_tokens_to_ids("<encoder-only>")
    native = enc_only != tok.unk_token_id
    prefix = [tok.cls_token_id, enc_only, tok.sep_token_id] if native else [tok.cls_token_id]
    src = None if native else AutoTokenizer.from_pretrained(source_tokenizer)
    limit = tok.model_max_length - len(prefix) - 1
    model = AutoModel.from_pretrained(backbone).to(device).eval()
    tok_start, tok_len, unit_start, unit_n = global_chunks(p, view)
    tokens, hidden = p.arr[f"{view}_tok"], model.config.hidden_size
    sp = span_pieces(p) if view == "S" and native else None
    state_path = out_dir / f"{view}_state.json"
    done = json.load(open(state_path))["done"] if state_path.exists() else 0
    mode = "r+" if done else "w+"
    emb = open_memmap(out_dir / f"{view}_chunks.npy", mode=mode, dtype=np.float16, shape=(len(tok_len), hidden))
    if sp is not None:
        pieces, n_spans = sp
        sums = open_memmap(out_dir / "SP_sums.npy", mode=mode, dtype=np.float32, shape=(n_spans, hidden))
        cnt = open_memmap(out_dir / "SP_counts.npy", mode=mode, dtype=np.float32, shape=(n_spans,))
    order = np.argsort(tok_len, kind="stable")
    pbar = tqdm(total=len(order), initial=done, desc=f"encode {view}")
    i, extra = done, len(prefix) + 1
    while i < len(order):
        j = i + 1
        while j < len(order) and (j + 1 - i) * _bucket(int(tok_len[order[j]]) + extra) <= batch_tokens:
            j += 1
        seqs = []
        for c in order[i:j]:
            body = tokens[tok_start[c]:tok_start[c] + tok_len[c]].astype(np.int64).tolist()
            if src is not None:
                text = src.decode(body, clean_up_tokenization_spaces=False)
                body = tok(text, add_special_tokens=False, truncation=True, max_length=limit)["input_ids"]
            seqs.append(prefix + body + [tok.sep_token_id])
        ids = np.full((j - i, _bucket(max(map(len, seqs)))), tok.pad_token_id, np.int64)
        for r, seq in enumerate(seqs):
            ids[r, :len(seq)] = seq
        x = torch.from_numpy(ids).to(device)
        mask = x != tok.pad_token_id
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            h = model(input_ids=x, attention_mask=mask.long()).last_hidden_state.float()
        emb[order[i:j]] = ((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).cpu().numpy().astype(np.float16)
        if sp is not None:
            batch = order[i:j]
            lo, hi = np.searchsorted(pieces[:, 0], batch), np.searchsorted(pieces[:, 0], batch, side="right")
            sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(batch) else np.zeros(0, int)
            if len(sel):
                row = {int(c): r for r, c in enumerate(batch)}
                pc = pieces[sel]
                rows = torch.as_tensor([row[int(c)] for c in pc[:, 0]], device=device)
                cs = h.cumsum(1)
                a = torch.as_tensor(pc[:, 1] + len(prefix) - 1, device=device)
                b = torch.as_tensor(pc[:, 2] + len(prefix) - 1, device=device)
                part = (cs[rows, b] - cs[rows, a]).cpu().numpy()
                np.add.at(sums, pc[:, 3], part)
                np.add.at(cnt, pc[:, 3], (pc[:, 2] - pc[:, 1]).astype(np.float32))
        del h, x, mask
        pbar.update(j - i)
        if j == len(order) or j // 50_000 != i // 50_000:
            emb.flush()
            if sp is not None:
                sums.flush()
                cnt.flush()
            json.dump({"done": j}, open(state_path, "w"))
        i = j
    out = open_memmap(out_dir / f"{view}.npy", mode="w+", dtype=np.float16, shape=(len(unit_n), hidden))
    for u0 in range(0, len(unit_n), 100_000):
        u1 = min(len(unit_n), u0 + 100_000)
        idx = np.repeat(np.arange(u0, u1), unit_n[u0:u1])
        rows = np.concatenate([np.arange(s, s + n) for s, n in zip(unit_start[u0:u1], unit_n[u0:u1])])
        acc = np.zeros((u1 - u0, hidden), np.float32)
        np.add.at(acc, idx - u0, emb[rows].astype(np.float32))
        out[u0:u1] = (acc / np.maximum(unit_n[u0:u1, None], 1)).astype(np.float16)
    out.flush()
    if sp is not None:
        sp_out = open_memmap(out_dir / "SP.npy", mode="w+", dtype=np.float16, shape=sums.shape)
        sp_out[:] = (np.asarray(sums) / np.maximum(np.asarray(cnt)[:, None], 1)).astype(np.float16)
        sp_out.flush()
        (out_dir / "SP_sums.npy").unlink()
        (out_dir / "SP_counts.npy").unlink()
    (out_dir / f"{view}_chunks.npy").unlink()
    state_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Frozen backbone features of the S and D function units (GPU)")
    ap.add_argument("--cache", default="data/scfuse")
    ap.add_argument("--backbone", default="microsoft/unixcoder-base")
    ap.add_argument("--views", nargs="+", default=["S", "D"])
    ap.add_argument("--batch-tokens", type=int, default=65_536)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    p = load_packed(args.cache)
    out_dir = Path(args.cache) / "feats" / args.backbone.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in args.views:
        encode_view(p, v, out_dir, args.backbone, args.batch_tokens, args.device)
    print(f"features -> {out_dir}")


if __name__ == "__main__":
    main()
