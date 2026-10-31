import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasketch import MinHash

from scfuse.data import load_packed
from scfuse.labels import load_label_map
from scfuse.metrics import f1_per_class

_TOK = re.compile(r"\w+|[^\w\s]")


def unit_texts(p, tok, i: int) -> list[str]:
    m = p.meta
    units = p.arr["S_unit"][m.S_unit_0.iat[i]:m.S_unit_0.iat[i] + m.S_unit_n.iat[i]]
    chunks = p.arr["S_chunk"][m.S_chunk_0.iat[i]:m.S_chunk_0.iat[i] + m.S_chunk_n.iat[i]]
    toks = p.arr["S_tok"][m.S_tok_0.iat[i]:m.S_tok_0.iat[i] + m.S_tok_n.iat[i]]
    out = []
    for c0, n, _ in units:
        ids = np.concatenate([toks[s:s + k] for s, k in chunks[c0:c0 + n]]) if n else np.zeros(0, int)
        out.append(" ".join(tok.decode(ids.astype(int).tolist()).split()))
    return out


def minhash(text: str) -> MinHash:
    toks = _TOK.findall(text)
    mh = MinHash(num_perm=128)
    for k in range(max(1, len(toks) - 4)):
        mh.update(" ".join(toks[k:k + 5]).encode())
    return mh


def macro(pred, y, kind, ev) -> dict:
    out = {}
    for sub, rows in (("all", np.ones(len(kind), bool)), ("solidifi", kind == "solidifi"),
                      ("smartbugs", kind == "smartbugs")):
        with np.errstate(all="ignore"):
            out[sub] = float(np.nanmean(f1_per_class(pred[rows], y[rows])[list(ev)]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Memorization checks: nearest-anchor contract and shared-function rule")
    ap.add_argument("--cache", default="data/scfuse")
    ap.add_argument("--out", default="results/sanity/memorization.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    lm = load_label_map()
    p = load_packed(args.cache, ["solidifi_unchecked_send_acc"])
    m = p.meta
    y = np.stack(m.y_gold.to_numpy())
    split, kind = m.split.to_numpy(), m.kind.to_numpy()
    test, anchor = np.flatnonzero(split == "gold_test"), np.flatnonzero(split == "gold_anchor")
    texts = {i: unit_texts(p, tok, i) for i in np.concatenate([test, anchor])}
    sig = {i: minhash(" ".join(t)) for i, t in texts.items()}
    res = {}
    for name, pool in (("anchors_all", anchor), ("anchors_smartbugs", anchor[kind[anchor] == "smartbugs"])):
        nn = np.asarray([pool[np.argmax([sig[i].jaccard(sig[a]) for a in pool])] for i in test])
        res[f"nearest_anchor.{name}"] = macro(y[nn], y[test], kind[test], lm.eval_idx)
        funcs = {}
        for a in pool:
            for t in texts[a]:
                funcs.setdefault(t, np.zeros(lm.K, int))
                funcs[t] |= y[a].astype(int)
        pred = np.asarray([np.bitwise_or.reduce([funcs.get(t, np.zeros(lm.K, int)) for t in texts[i]] or
                                                [np.zeros(lm.K, int)]) for i in test])
        res[f"shared_function.{name}"] = macro(pred, y[test], kind[test], lm.eval_idx)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
