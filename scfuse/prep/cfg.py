from pathlib import Path

from scfuse.prep.common import Bytecodes, Raw, load_index, parser, pool_map, write_jsonl
from scfuse.utils.evm import build_cfg, covariates

_G = {}


def _init(raw_root: str):
    _G["bc"] = Bytecodes(Raw(raw_root))


def process(cid: str) -> dict:
    code = _G["bc"].get(cid)
    if not code or len(code) <= 4:
        return {"id": cid, "ok": False}
    g = build_cfg(code)
    return {"id": cid, "ok": g.n_blocks > 0, "sels": g.sels, "q_G": [round(float(x), 5) for x in g.q],
            "n_blocks": g.n_blocks, "z": covariates(code).tolist()}


def main():
    args = parser("View G: CFG statistics, dispatcher selectors, q_G and bytecode covariates z").parse_args()
    ids = [r["id"] for r in load_index(args.cache)]
    n = write_jsonl(Path(args.cache) / "cfg.jsonl", pool_map(process, ids, args.workers, "cfg", 16, _init, (args.raw,)))
    print(f"cfg: {n} contracts")


if __name__ == "__main__":
    main()
