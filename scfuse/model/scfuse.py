import torch
import torch.nn as nn

from scfuse.model.baselines import HEADS, build_head
from scfuse.model.certify import selector_coverage, selector_share
from scfuse.model.encoders import (ASTEncoder, AttnPool, CrossView, GraphEncoder, LabelAttention, LoraText,
                                    OpcodeEncoder, ViewAdapter)
from scfuse.model.fusion import RULES, deferral, fuse, opinion
from scfuse.schema import SOURCES, TEXT_VIEWS, VI, VIEWS
from scfuse.utils.ast import LABELS


class SCFuse(nn.Module):
    def __init__(self, cfg: dict, K: int, base_rate: torch.Tensor, q_dims: dict, ast_vocab: int = len(LABELS),
                 d_feat: int = 768):
        super().__init__()
        mc, fc = cfg["model"], cfg["fusion"]
        d, p = mc["d"], mc["dropout"]
        assert fc["rule"] in RULES and fc["reliability"] in ("learned", "one"), fc
        self.head = mc["head"]
        assert self.head == "evidential" or self.head in HEADS, self.head
        self.use = [v in cfg["views"]["use"] for v in VIEWS]
        self.register_buffer("view_on", torch.tensor([float(u) for u in self.use]))
        self.register_buffer("a", base_rate.float())
        self.regime, self.use_J, self.d, self.K = mc["regime"], mc["use_J"], d, K
        self.rule, self.learned_rel = fc["rule"], fc["reliability"] == "learned"
        self.Emax, self.W, self.lam_dis = fc["Emax"], fc["W"], fc.get("lam_dis", 0.0)
        self.robust = bool(mc.get("reliability_robust", False))
        if self.regime == "cached":
            self.text = nn.ModuleDict({v: ViewAdapter(d_feat, d, p) for v in TEXT_VIEWS})
        elif self.regime == "lora":
            lc = mc["lora"]
            self.text = LoraText(lc["backbone"], len(TEXT_VIEWS), d, lc["r"], lc["alpha"], lc["dropout"],
                                 lc["grad_ckpt"], lc["max_rows"])
        else:
            raise ValueError(self.regime)
        oc, gc, ac = mc["opcode"], mc["graph"], mc["ast"]
        self.freeze_opcode = oc["freeze"]
        self.opcode = OpcodeEncoder(oc["d"], oc["layers"], oc["heads"], p).requires_grad_(not oc["freeze"])
        self.graph = GraphEncoder(oc["d"], d, gc["layers"], gc["heads"], gc["rw_steps"], p)
        self.gpool = AttnPool(d)
        self.ast = ASTEncoder(ast_vocab, ac["d"], d, ac["layers"], ac["heads"], p) if self.use[VI["A"]] else None
        self.att = LabelAttention(K, d)
        if self.head != "evidential":
            self.bhead = build_head(self.head, K, d, len(VIEWS), mc["baselines"], p, base_rate)
            return
        cc = mc["cross"]
        self.cross = CrossView(d, cc["layers"], cc["heads"], p, len(VIEWS)) if self.use_J else None
        self.U = nn.Parameter(torch.randn(len(SOURCES), K, 2, d) * 0.02)
        self.bias = nn.Parameter(torch.full((len(SOURCES), K, 2), -3.0))
        qd = [q_dims[v] + (2 if v == "D" and self.robust else 0) for v in VIEWS]
        qd.append(len(VIEWS) + (sum(q_dims[v] for v in VIEWS) if self.robust else 0))
        self.rel = nn.ModuleList(nn.Sequential(nn.Linear(q + d, 64), nn.GELU(), nn.Linear(64, 1)) for q in qd)
        self.corrupt = (nn.Sequential(nn.Linear(qd[VI["D"]] + d, 64), nn.GELU(), nn.Linear(64, 1))
                        if self.robust else None)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_opcode:
            self.opcode.eval()
        return self

    def blocks(self, x, mask):
        out = torch.zeros(*x.shape[:2], self.opcode.emb.embedding_dim, device=x.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_opcode):
            out[mask] = self.opcode(x[mask]).float()
        return out

    def encode(self, batch: dict, m: torch.Tensor):
        hs, masks, sels = [None] * len(VIEWS), [None] * len(VIEWS), [None] * len(VIEWS)
        sp = batch.get("SP")
        span_h = span_ok = None
        if self.regime == "cached":
            for v in TEXT_VIEWS:
                hs[VI[v]] = self.text[v](batch[v]["feat"])
            if sp is not None:
                span_h, span_ok = self.text["S"](sp["feat"]), sp["valid"]
        else:
            keep = [m[batch[v]["chunk_unit"] // batch[v]["mask"].shape[1], VI[v]] > 0 for v in TEXT_VIEWS]
            (hs[VI["S"]], hs[VI["D"]]), span_h, span_ok = self.text([batch[v] for v in TEXT_VIEWS], keep, sp)
        for v in TEXT_VIEWS:
            i = VI[v]
            masks[i], sels[i] = batch[v]["mask"] & (m[:, i:i + 1] > 0), batch[v]["sel"]
        G, i = batch["G"], VI["G"]
        nodes = self.graph(self.blocks(G["x"], G["mask"]), G["adj"], G["mask"])
        hs[i] = self.gpool(nodes, G["fn"])
        masks[i], sels[i] = G["fnmask"] & (m[:, i:i + 1] > 0), G["fnsel"]
        A, i = batch["A"], VI["A"]
        B, Ua = A["sel"].shape
        hA = nodes.new_zeros(B * Ua, self.d, dtype=torch.float32)
        if self.ast is not None and len(A["row"]):
            hA[A["row"]] = self.ast(A["node"]).float()
        hs[i] = hA.view(B, Ua, self.d)
        masks[i], sels[i] = A["mask"] & (m[:, i:i + 1] > 0), A["sel"]
        hs = [h.float() for h in hs]
        anchors = None
        if sp is not None:
            anchors = {"block": nodes.flatten(0, 1)[sp["block"]].float(), "span": span_h.float(), "valid": span_ok}
        return hs, masks, sels, anchors

    def reliability_inputs(self, batch: dict, m: torch.Tensor) -> list:
        q = batch["q"]
        qs = [q[v].float() for v in VIEWS]
        if not self.robust:
            return qs + [m]
        D, G = batch["D"], batch["G"]
        share, _ = selector_share(D["sel"], D["mask"], G["fnsel"], G["fnmask"])
        cover = selector_coverage(D["sel"], D["mask"], G["fnsel"], G["fnmask"])
        qs[VI["D"]] = torch.cat([qs[VI["D"]], share[:, None], cover[:, None]], 1)
        return qs + [torch.cat([m] + [m[:, i:i + 1] * q[v].float() for i, v in enumerate(VIEWS)], 1)]

    def forward(self, batch: dict, m: torch.Tensor) -> dict:
        m = m * self.view_on
        hs, masks, sels, anchors = self.encode(batch, m)
        c = [self.att(hs[i], masks[i])[0] for i in range(len(VIEWS))]
        base = {"h": hs, "mask": masks, "sel": sels, "anchors": anchors}
        if self.head != "evidential":
            cs = torch.stack(c, 1).float() * m[:, :, None, None]
            extra = self.bhead(cs, m, batch["G"]["x"]) if self.head == "vulnsense" else self.bhead(cs, m)
            logit = extra.pop("logit").float()
            P = torch.sigmoid(logit)
            m_src = torch.cat([m, torch.zeros_like(m[:, :1])], 1)
            u = 1 - (2 * P - 1).abs()
            return {**base, **extra, "P": P, "u": u, "upsilon": u, "logit": logit, "omega": torch.ones_like(m_src),
                    "m_src": m_src}
        if self.use_J:
            tok, keep = self.cross(hs, masks)
            c.append(self.att(tok.float(), keep)[0])
        else:
            c.append(torch.zeros_like(c[0]))
        c = torch.stack(c, 1).float()
        ev = self.Emax * torch.sigmoid(torch.einsum("bjkd,jkcd->bjkc", c, self.U.float()) + self.bias.float())
        r, s = ev[..., 0], ev[..., 1]
        m_src = torch.cat([m, (m.sum(1, keepdim=True) > 0).float() * float(self.use_J)], 1)
        qs = self.reliability_inputs(batch, m)
        cbar = c.mean(2)
        logit = torch.cat([self.rel[j](torch.cat([qs[j], cbar[:, j].detach()], -1)) for j in range(len(SOURCES))],
                          1).float()
        omega = torch.sigmoid(logit) if self.learned_rel else torch.ones_like(logit)
        P, u, R, S = fuse(r, s, omega, m_src, self.a, self.W, self.rule)
        Pj, _ = opinion(r, s, self.a, self.W)
        out = {**base, "P": P, "u": u, "R": R, "S": S, "r": r, "s": s, "omega": omega, "omega_logit": logit,
               "Pj": Pj, "m_src": m_src, "upsilon": deferral(r, s, omega, m_src, u, self.W, self.lam_dis)}
        if self.corrupt is not None:
            out["corrupt_logit"] = self.corrupt(torch.cat([qs[VI["D"]], c[:, VI["D"]].mean(1)], -1)).squeeze(-1)
        return out
