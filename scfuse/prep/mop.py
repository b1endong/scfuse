import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scfuse.data import load_packed
from scfuse.model.encoders import N_OPCODES, OpcodeEncoder


def main():
    ap = argparse.ArgumentParser(description="Masked-opcode pretraining of the block encoder (GPU)")
    ap.add_argument("--cache", default="data/scfuse")
    ap.add_argument("--out", default="opcode_mop.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mask", type=float, default=0.15)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    p = load_packed(args.cache)
    tr = p.meta[p.meta.split.eq("train")]
    rows = np.concatenate([np.arange(a, a + n) for a, n in zip(tr.G_x_0, tr.G_x_n)])
    X, dev = p.arr["G_x"], args.device
    enc = OpcodeEncoder(args.d, args.layers, args.heads).to(dev)
    head = nn.Linear(args.d, N_OPCODES + 2).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = args.epochs * math.ceil(len(rows) / args.batch)
    warm = max(1, total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist, t0 = [], time.time()
    pbar = tqdm(total=total, desc="MOP")
    for _ in range(args.epochs):
        perm = rng.permutation(rows)
        for i in range(0, len(perm), args.batch):
            x = torch.from_numpy(X[np.sort(perm[i:i + args.batch])].astype(np.int64)).to(dev)
            msk = (torch.rand(x.shape, device=dev) < args.mask) & (x != 0)
            if not msk.any():
                continue
            with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                logits = head(enc.tokens(x.masked_fill(msk, enc.mask_id)))[msk].float()
            loss = F.cross_entropy(logits, x[msk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            hist.append((loss.item(), (logits.argmax(-1) == x[msk]).float().mean().item()))
            pbar.update(1)
    pbar.close()
    tail = hist[-max(1, len(hist) // 20):]
    log = {"blocks": int(len(rows)), "steps": len(hist), "seconds": round(time.time() - t0),
           "loss_last5pct": float(np.mean([h[0] for h in tail])), "acc_last5pct": float(np.mean([h[1] for h in tail]))}
    out = Path(args.cache) / "feats" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": enc.state_dict(), "cfg": {"d": args.d, "layers": args.layers, "heads": args.heads},
                "log": log}, out)
    print(json.dumps(log), "->", out)


if __name__ == "__main__":
    main()
