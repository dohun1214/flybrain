"""Shiu 리포의 커넥톰 parquet/CSV → PyTorch 커널용 CSR(npz) + 메타데이터 + 주석 조인 (이슈 #3).

실행 (.venv):
    python data/build_csr.py            # v630, v783 둘 다
    python data/build_csr.py 630        # 하나만

입력 (gitignore, external/):
    external/Drosophila_brain_model/2023_03_23_completeness_630_final.csv / ..._connectivity_630_final.parquet
    external/Drosophila_brain_model/Completeness_783.csv / Connectivity_783.parquet
    external/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv  (v783 root_id 기준)

출력 (gitignore, data/processed/):
    v{ver}.npz         root_ids[N] int64 · indptr[N+1] int64 · indices[nnz] int32 (post) · weights_mv[nnz] float32
                       · syn_count[nnz] int32 · excitatory[nnz] int8   — CSR 행 = presynaptic 뉴런 (스파이크 전파용)
    v{ver}.annot.parquet  뉴런 인덱스 순서의 주석 (super_class, cell_class, cell_type, side, 좌표, top_nt …)

출력 (git 추적, data/):
    v{ver}.meta.json   뉴런 수, edge 수, 시냅스 합, ≥5 연결 수, 임계값(없음), 주석 매칭률 …

뉴런 순서 = Completeness CSV 행 순서 (= parquet의 *_Index). 가중치 = `Excitatory x Connectivity` × 0.275 mV.
임계값 없음: 모든 뉴런쌍 edge를 그대로 쓴다 (계획서 §3.2).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / "external" / "Drosophila_brain_model"
ANNOT = ROOT / "external" / "flywire_annotations" / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
OUT = ROOT / "data" / "processed"

W_SYN_MV = 0.275  # 시냅스 1개당 mV (원논문 free parameter)

FILES = {
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}
EXPECTED = {  # 계획서 §3.2 (2026-09-13 리포 파일에서 계산한 값) — 빌드 시 재검증
    "630": dict(n_neurons=127_400, n_edges=14_687_178, syn_sum=52_793_639, n_edges_ge5=2_614_028),
    "783": dict(n_neurons=138_639, n_edges=15_091_983, syn_sum=54_492_922, n_edges_ge5=2_700_513),
}
ANNOT_COLS = ["super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side",
              "flow", "nerve", "top_nt", "top_nt_conf", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z"]


def build(ver: str):
    t0 = time.time()
    comp_path, con_path = (SHIU / f for f in FILES[ver])
    df_comp = pd.read_csv(comp_path, index_col=0)
    root_ids = df_comp.index.values.astype(np.int64)
    n = len(root_ids)

    df = pd.read_parquet(con_path)
    pre = df["Presynaptic_Index"].values.astype(np.int64)
    post = df["Postsynaptic_Index"].values.astype(np.int64)
    # 인덱스 ↔ root_id 일관성 검증
    assert (root_ids[pre] == df["Presynaptic_ID"].values).all(), "Presynaptic_Index/ID 불일치"
    assert (root_ids[post] == df["Postsynaptic_ID"].values).all(), "Postsynaptic_Index/ID 불일치"
    assert not df.duplicated(["Presynaptic_Index", "Postsynaptic_Index"]).any(), "중복 edge"

    # CSR (행 = pre). 안정 정렬로 pre → post 순
    order = np.lexsort((post, pre))
    pre_s, post_s = pre[order], post[order]
    syn = df["Connectivity"].values[order].astype(np.int32)
    exc = df["Excitatory"].values[order].astype(np.int8)
    w = (df["Excitatory x Connectivity"].values[order] * W_SYN_MV).astype(np.float32)
    assert set(np.unique(exc)) <= {-1, 1}
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre_s + 1, 1)
    indptr = np.cumsum(indptr)

    meta = dict(
        version=ver,
        n_neurons=int(n),
        n_edges=int(len(df)),
        syn_sum=int(df["Connectivity"].sum()),
        n_edges_ge5=int((df["Connectivity"] >= 5).sum()),
        threshold="none (all edges)",
        w_syn_mv=W_SYN_MV,
        n_excitatory_edges=int((exc == 1).sum()),
        n_inhibitory_edges=int((exc == -1).sum()),
        weight_abs_sum_mv=float(np.abs(w).sum()),
        max_out_degree=int(np.diff(indptr).max()),
        n_neurons_no_output=int((np.diff(indptr) == 0).sum()),
        n_neurons_no_input=int(n - len(np.unique(post))),
        source_files=[str(comp_path.name), str(con_path.name)],
    )
    for k, v in EXPECTED[ver].items():
        assert meta[k] == v, f"{ver} {k}: got {meta[k]}, expected {v}"
    meta["expected_values_match_plan"] = True

    # 주석 조인 (v783 root_id 기준 TSV)
    annot = pd.read_csv(ANNOT, sep="\t", low_memory=False)
    annot = annot.drop_duplicates("root_id").set_index("root_id")
    joined = annot.reindex(root_ids)[ANNOT_COLS]
    joined.insert(0, "root_id", root_ids)
    joined.index = np.arange(n)
    joined.index.name = "index"
    matched = joined["super_class"].notna()
    meta["annotation"] = dict(
        source="flyconnectome/flywire_annotations Supplemental_file1_neuron_annotations.tsv",
        n_annotation_rows=int(len(annot)),
        n_matched=int(matched.sum()),
        match_rate=float(matched.mean()),
        super_class_counts={k: int(v) for k, v in joined["super_class"].value_counts().items()},
    )

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"v{ver}.npz", root_ids=root_ids, indptr=indptr, indices=post_s.astype(np.int32),
             weights_mv=w, syn_count=syn, excitatory=exc)
    joined.to_parquet(OUT / f"v{ver}.annot.parquet")
    meta["build_seconds"] = round(time.time() - t0, 1)
    (ROOT / "data" / f"v{ver}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "annotation"}, indent=1))
    print("annotation match: %d / %d = %.2f%%" % (meta["annotation"]["n_matched"], n, 100 * meta["annotation"]["match_rate"]))
    return meta


if __name__ == "__main__":
    for ver in (sys.argv[1:] or ["630", "783"]):
        build(ver)
