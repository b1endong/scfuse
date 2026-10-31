from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

LABEL_MAP = Path(__file__).resolve().parents[1] / "configs" / "label_map.yaml"
ANALYZERS = ("slither", "mythril")


@dataclass(frozen=True)
class LabelMap:
    codes: tuple[str, ...]
    eval_idx: tuple[int, ...]
    external_idx: tuple[int, ...]
    slither: dict
    mythril: dict
    covered: dict
    views: dict

    @property
    def K(self) -> int:
        return len(self.codes)


@lru_cache
def load_label_map(path: str | Path = LABEL_MAP) -> LabelMap:
    cfg = yaml.safe_load(open(path))
    codes = tuple(c["code"] for c in cfg["classes"])
    slither, mythril = {}, {}
    covered = {a: np.zeros(len(codes), bool) for a in ANALYZERS}
    for k, c in enumerate(cfg["classes"]):
        for det in c["slither"]:
            slither[det] = k
        for swc in c["mythril"]:
            mythril.setdefault(int(swc), []).append(k)
        covered["slither"][k] = bool(c["slither"])
        covered["mythril"][k] = bool(c["mythril"])
    return LabelMap(codes, tuple(codes.index(c) for c in cfg["eval_classes"]),
                    tuple(codes.index(c) for c in cfg["external_classes"]), slither, mythril, covered,
                    cfg["analyzer_views"])


def slither_label(detectors: list[str], lm: LabelMap) -> np.ndarray:
    y = np.zeros(lm.K, np.int8)
    for det in detectors:
        if det in lm.slither:
            y[lm.slither[det]] = 1
    return y


def mythril_label(swc_ids: list, lm: LabelMap) -> np.ndarray:
    y = np.zeros(lm.K, np.int8)
    for swc in swc_ids:
        for k in lm.mythril.get(int(swc), []):
            y[k] = 1
    return y
