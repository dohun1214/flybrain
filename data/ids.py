"""논문 뉴런 ID(data/paper_ids.json, v630 root_id) ↔ 커넥톰 인덱스 변환 유틸.

    from data.ids import Connectome
    c = Connectome("630")           # data/processed/v630.npz 로드 (lazy)
    c.index_of(720575940660219265)  # MN9 → int 인덱스
    c.lists["neu_sugar"]            # 논문 목록 이름 → np.ndarray 인덱스 (해당 버전에 없는 ID는 제외)
    c.missing["neu_sugar"]          # 해당 버전에 없는 root_id 목록
    c.annot                         # 뉴런 인덱스 순서의 주석 DataFrame (parquet)

논문 명칭(aDN1, aBN1, GF, MN9)은 FlyWire cell_type에 없고, 당·물 GRN은 둘 다 cell_type LB3라
타입으로 구분할 수 없다 → 반드시 이 ID 목록으로 찾는다.
"""
import json
from functools import cached_property
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_IDS = ROOT / "data" / "paper_ids.json"
PROCESSED = ROOT / "data" / "processed"

# 논문 핵심 단일 뉴런 (v630 root_id). 명칭은 논문(Shiu et al. 2024) 기준
SINGLE = {
    "MN9": 720575940660219265,
    "aBN1": 720575940630907434,
    "aDN1": 720575940616185531,
    "aDN2": 720575940629806974,
}


class Connectome:
    def __init__(self, version: str = "630"):
        self.version = str(version)
        self.path = PROCESSED / f"v{self.version}.npz"
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} 없음 — 먼저 `python data/build_csr.py {self.version}` 실행")
        z = np.load(self.path)
        self.root_ids = z["root_ids"]
        self.indptr = z["indptr"]
        self.indices = z["indices"]
        self.weights_mv = z["weights_mv"]
        self.syn_count = z["syn_count"]
        self.excitatory = z["excitatory"]
        self.n = len(self.root_ids)
        self._id2i = {int(r): i for i, r in enumerate(self.root_ids)}
        self.lists, self.missing = self._load_paper_lists()

    # --- ID 변환
    def index_of(self, root_id: int) -> int:
        return self._id2i[int(root_id)]

    def indices_of(self, root_ids, strict: bool = True):
        out, miss = [], []
        for r in root_ids:
            i = self._id2i.get(int(r))
            (out if i is not None else miss).append(i if i is not None else int(r))
        if strict and miss:
            raise KeyError(f"v{self.version}에 없는 root_id {len(miss)}개: {miss[:5]} …")
        return np.asarray(out, dtype=np.int64), miss

    def _load_paper_lists(self):
        raw = json.loads(PAPER_IDS.read_text(encoding="utf-8"))
        lists, missing = {}, {}
        for name, ids in raw["lists"].items():
            lists[name], missing[name] = self.indices_of(ids, strict=False)
        for name, rid in SINGLE.items():
            idx, miss = self.indices_of([rid], strict=False)
            lists[name], missing[name] = idx, miss
        return lists, missing

    # --- 주석
    @cached_property
    def annot(self):
        import pandas as pd

        return pd.read_parquet(PROCESSED / f"v{self.version}.annot.parquet")

    @cached_property
    def meta(self):
        return json.loads((ROOT / "data" / f"v{self.version}.meta.json").read_text(encoding="utf-8"))

    def out_edges(self, i: int):
        """뉴런 i의 출력 시냅스 (post 인덱스, 가중치 mV)."""
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.weights_mv[a:b]

    def summary(self) -> str:
        lines = [f"v{self.version}: {self.n:,} neurons, {len(self.indices):,} edges"]
        for name, idx in self.lists.items():
            miss = self.missing[name]
            lines.append(f"  {name:12s} {len(idx):3d} found" + (f", {len(miss)} missing" if miss else ""))
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    for v in (sys.argv[1:] or ["630", "783"]):
        c = Connectome(v)
        print(c.summary())
        for name in SINGLE:
            if len(c.lists[name]):
                i = int(c.lists[name][0])
                row = c.annot.iloc[i]
                print(f"  {name}: idx {i}, cell_type={row['cell_type']}, super_class={row['super_class']}, side={row['side']}, out_edges={c.indptr[i+1]-c.indptr[i]}")
