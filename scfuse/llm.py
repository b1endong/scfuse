import argparse
import json
import re
from pathlib import Path

import numpy as np

from scfuse.config import load_config
from scfuse.data import load_packed
from scfuse.eval import host_of
from scfuse.labels import load_label_map
from scfuse.sanity import unit_texts

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DESCRIPTIONS = {
    "REE": "Reentrancy (SWC-107): an external call is made before the contract's own state is updated.",
    "ARI": "Integer overflow or underflow (SWC-101).",
    "ULC": "Unchecked return value of a low-level call such as call, send or delegatecall (SWC-104).",
    "ACC": "Access control: tx.origin used for authorization, or an unprotected Ether withdrawal or selfdestruct "
           "(SWC-115, SWC-105, SWC-106).",
    "TSD": "Timestamp dependence: block.timestamp drives a critical decision (SWC-116).",
    "TOD": "Transaction order dependence, i.e. front-running (SWC-114).",
}
VIEW_TEXT = {"S": "the source code (comments removed, some identifiers renamed)",
             "D": "pseudo-Solidity decompiled from the runtime bytecode"}


def prompt(code: str, view: str, truncated: bool, codes: list) -> list[dict]:
    lines = "\n".join(f"{c}: {DESCRIPTIONS[c]}" for c in codes)
    schema = ", ".join(f'"{c}": "<yes|no>"' for c in codes)
    return [{"role": "system", "content": "You are an expert smart contract security auditor."},
            {"role": "user", "content":
                f"Below is {VIEW_TEXT[view]} of an Ethereum smart contract{' (truncated)' if truncated else ''}.\n\n"
                f"```solidity\n{code}\n```\n\nFor each vulnerability type below, decide whether the contract contains "
                f"it.\n{lines}\n\nAnswer with a JSON object only, mapping every code to \"yes\" or \"no\": {{{schema}}}"}]


def parse(text: str, codes: list) -> tuple[dict, bool]:
    m = re.search(r"\{[^{}]*\}", text, re.S)
    try:
        obj = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        obj = {}
    return {c: str(obj.get(c, "no")).strip().lower().startswith("y") for c in codes}, bool(obj)


def main():
    ap = argparse.ArgumentParser(description="Zero-shot LLM baseline on the gold test set (GPU)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--view", choices=("S", "D"), required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-code-tokens", type=int, default=12_000)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config(args.config)
    lm = load_label_map()
    codes = [lm.codes[k] for k in lm.eval_idx]
    p = load_packed(cfg["data"]["cache"], cfg["data"]["label_fix"])
    test = np.flatnonzero(p.meta.split.to_numpy() == "gold_test")
    unix = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    out_dir = Path(cfg["run"]["eval"]) / f"llm_{args.model.split('/')[-1].lower()}_{args.view}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, readable, records = np.zeros((len(test), lm.K), bool), 0, []
    for r, i in enumerate(test):
        if args.view == "S":
            code = "\n".join(unit_texts(p, unix, i))
        else:
            m = p.meta
            sl = slice(m.D_unit_0.iat[i], m.D_unit_0.iat[i] + m.D_unit_n.iat[i])
            chunks = p.arr["D_chunk"][m.D_chunk_0.iat[i]:m.D_chunk_0.iat[i] + m.D_chunk_n.iat[i]]
            toks = p.arr["D_tok"][m.D_tok_0.iat[i]:m.D_tok_0.iat[i] + m.D_tok_n.iat[i]]
            code = "\n".join(unix.decode(np.concatenate([toks[s:s + k] for s, k in chunks[c0:c0 + n]]).astype(int).tolist())
                             for c0, n, _ in p.arr["D_unit"][sl] if n)
        ids = tok(code, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > args.max_code_tokens
        if truncated:
            code = tok.decode(ids[:args.max_code_tokens])
        x = tok.apply_chat_template(prompt(code, args.view, truncated, codes), add_generation_prompt=True,
                                    return_tensors="pt").to("cuda")
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        answer = tok.decode(y[0, x.shape[1]:], skip_special_tokens=True)
        ans, ok = parse(answer, codes)
        readable += ok
        for c, v in ans.items():
            pred[r, lm.codes.index(c)] = v
        records.append({"id": p.meta.id.iat[i], "answer": answer, "truncated": truncated})
    y = np.stack(p.meta.y_gold.to_numpy())[test]
    kind = p.meta.kind.to_numpy()[test]
    from scfuse.metrics import f1_per_class
    with np.errstate(all="ignore"):
        summ = {sub: float(np.nanmean(f1_per_class(pred[rows], y[rows])[list(lm.eval_idx)]))
                for sub, rows in (("all", np.ones(len(kind), bool)), ("solidifi", kind == "solidifi"),
                                  ("smartbugs", kind == "smartbugs"))}
    json.dump({"macro_f1": summ, "readable": readable, "n": len(test), "model": args.model, "revision": args.revision},
              open(out_dir / "summary.json", "w"), indent=1)
    with open(out_dir / "answers.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    np.savez_compressed(out_dir / "pred_s0.npz", P1_pred=pred, y=y, kind=kind,
                        host=np.asarray([host_of(c) for c in p.meta.id.to_numpy()[test]]))
    print(summ)


if __name__ == "__main__":
    main()
