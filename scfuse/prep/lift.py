import json
import re
import time
from pathlib import Path

from scfuse.prep.common import Bytecodes, Raw, load_index, parser
from scfuse.utils.evm import disassembly, runtime

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
_FENCE = re.compile(r"```(?:solidity)?\s*\n(.*?)```", re.S)


def prompt(asm: str, solc: str) -> list[dict]:
    return [{"role": "system", "content": "You are an expert in EVM bytecode and Solidity."},
            {"role": "user", "content":
                f"Below is the disassembly of the runtime bytecode of an Ethereum contract compiled with solc {solc} "
                f"(metadata removed; each JUMPDEST carries its byte offset in hex).\n\n```\n{asm}\n```\n\n"
                f"Decompile it into Solidity source code that compiles with solc {solc} and behaves like this bytecode. "
                f"Name every function reached through the selector dispatcher func_<selector>, for example "
                f"func_3ccfd60b for selector 0x3ccfd60b. Answer with a single ```solidity code block."}]


def code_block(text: str) -> str:
    blocks = _FENCE.findall(text)
    return max(blocks, key=len) if blocks else text


def main():
    ap = parser("Lift runtime bytecode to pseudo-Solidity with a code foundation model (GPU)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", required=True, help="pinned Hugging Face commit")
    ap.add_argument("--splits", nargs="*", default=None, help="restrict to these splits")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-input-tokens", type=int, default=28_000)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = Path(args.cache)
    out_dir = cache / "lifted"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = cache / f"lift_{args.shard}of{args.num_shards}.jsonl"
    done = {json.loads(line)["id"] for line in open(log)} if log.exists() else set()
    rows = [r for r in load_index(cache) if args.splits is None or r["split"] in args.splits]
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r["id"])) if i % args.num_shards == args.shard]
    bc = Bytecodes(Raw(args.raw))
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    with open(log, "a") as f:
        for r in rows:
            if r["id"] in done:
                continue
            code = bc.get(r["id"])
            rec = {"id": r["id"], "model": args.model, "revision": args.revision}
            if not code:
                rec["status"] = "no_bytecode"
            else:
                x = tok.apply_chat_template(prompt(disassembly(runtime(code)), r.get("solc") or "0.4.24"),
                                            add_generation_prompt=True, return_tensors="pt").to("cuda")
                rec["prompt_tokens"] = int(x.shape[1])
                if x.shape[1] > args.max_input_tokens:
                    rec["status"] = "context"
                else:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            y = model.generate(x, max_new_tokens=args.max_new_tokens, do_sample=False,
                                               pad_token_id=tok.eos_token_id)
                        torch.cuda.synchronize()
                        text = tok.decode(y[0, x.shape[1]:], skip_special_tokens=True)
                        (out_dir / f"{r['id']}.sol").write_text(code_block(text))
                        rec.update(status="ok", new_tokens=int(y.shape[1] - x.shape[1]),
                                   seconds=round(time.perf_counter() - t0, 2))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rec["status"] = "oom"
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
