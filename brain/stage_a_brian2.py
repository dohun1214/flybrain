"""Stage A: Shiu et al. 2024 원본 Brian2 모델로 v630 당 GRN → MN9 벤치마크 재현 (정답 생성).

원본 리포(external/Drosophila_brain_model)의 model.py를 그대로 import해 쓴다.
원본과 다른 점은 (1) trial마다 brian2.seed()를 명시적으로 고정하는 것,
(2) 결과를 data/ground_truth/ 에 저장하는 것뿐이다.

실행 (.venv-brian2):
    python brain/stage_a_brian2.py run      # 시뮬레이션 (수 분 ~ 수십 분)
    python brain/stage_a_brian2.py analyze  # 요약 지표 → data/ground_truth/summary.json + stdout 마크다운

실험 목록
- sugarR_150Hz : 당 GRN 우측 21개 포아송 150Hz (코드 기본값), 시드 0..4 × 30 trial
- sugarR_100Hz : 동일, 100Hz (example.ipynb 두 번째 조건), 시드 0..4 × 30 trial
- sugarR_100Hz_slnc_<id> : 100Hz 조건에서 가장 활발한 뉴런 3개를 각각 silence (example.ipynb 절차), 시드 0 × 30 trial
- sugarR_200Hz : 200Hz, 시드 0 × 30 trial — 저자 동봉 결과 results/example/sugarR.parquet 는 당 GRN이 약 197Hz로
  발화하므로 200Hz로 생성된 것(노트북 본문 "200 Hz"와 일치, 코드 기본값 150Hz와 불일치). 저자 결과와 직접 비교용.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / "external" / "Drosophila_brain_model"
sys.path.insert(0, str(SHIU))

PATH_COMP = SHIU / "2023_03_23_completeness_630_final.csv"
PATH_CON = SHIU / "2023_03_23_connectivity_630_final.parquet"
PATH_AUTHOR = SHIU / "results" / "example"
OUT = ROOT / "data" / "ground_truth"

ID_MN9 = 720575940660219265
SEEDS = [0, 1, 2, 3, 4]
N_JOBS = 12


# ---------------------------------------------------------------- run
def _trial(seed, exc, exc2, slnc, params):
    """원본 model.run_trial을 시드 고정 후 실행 (loky 워커에서 호출)."""
    if str(SHIU) not in sys.path:
        sys.path.insert(0, str(SHIU))
    import brian2
    from brian2 import prefs

    prefs.codegen.target = "numpy"  # 컴파일러 없음 → 경고 대신 명시
    import model

    brian2.seed(seed)
    return model.run_trial(exc, exc2, slnc, PATH_COMP, PATH_CON, params)


def run_exp(exp_name, neu_exc, params, seed, neu_slnc=(), force=False):
    from joblib import Parallel, delayed, parallel_backend

    import model

    path_save = OUT / f"{exp_name}_seed{seed}.parquet"
    if path_save.is_file() and not force:
        print(f">>> skip {path_save.name} (exists)")
        return

    df_comp = pd.read_csv(PATH_COMP, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    exc = [flyid2i[n] for n in neu_exc]
    slnc = [flyid2i[n] for n in neu_slnc]

    n_run = params["n_run"]
    print(f">>> {exp_name} seed={seed} n_run={n_run} exc={len(exc)} slnc={len(slnc)}")
    t0 = time.time()
    with parallel_backend("loky", n_jobs=N_JOBS):
        res = Parallel()(
            delayed(_trial)(seed * 1000 + k, exc, [], slnc, params) for k in range(n_run)
        )
    wall = time.time() - t0
    df = model.construct_dataframe(res, exp_name, i2flyid)
    df["seed"] = seed
    df.to_parquet(path_save, compression="brotli")
    print(f"    {len(df)} spikes, {df.flywire_id.nunique()} active neurons, {wall:.0f}s -> {path_save.name}")


def cmd_run():
    from brian2 import Hz

    import model

    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads((ROOT / "data" / "paper_ids.json").read_text(encoding="utf-8"))
    neu_sugar = ids["lists"]["neu_sugar"]
    assert len(neu_sugar) == 21

    p150 = dict(model.default_params)  # r_poi = 150 Hz
    p100 = dict(model.default_params)
    p100["r_poi"] = 100 * Hz
    p200 = dict(model.default_params)
    p200["r_poi"] = 200 * Hz

    for seed in SEEDS:
        run_exp("sugarR_150Hz", neu_sugar, p150, seed)
    for seed in SEEDS:
        run_exp("sugarR_100Hz", neu_sugar, p100, seed)

    # silencing: 100Hz seed0 결과에서 가장 활발한 뉴런 3개 (example.ipynb 절차)
    df = pd.read_parquet(OUT / "sugarR_100Hz_seed0.parquet")
    top3 = df.groupby("flywire_id").size().sort_values(ascending=False).index[:3]
    for i in top3:
        run_exp(f"sugarR_100Hz_slnc_{i}", neu_sugar, p100, 0, neu_slnc=[int(i)])

    run_exp("sugarR_200Hz", neu_sugar, p200, 0)

    # 저자 동봉 silencing 결과(results/example/sugarR-<id>.parquet)와 같은 뉴런으로도 실행 → 직접 비교
    for f in PATH_AUTHOR.glob("sugarR-*.parquet"):
        i = int(f.stem.split("-")[1])
        run_exp(f"sugarR_100Hz_slnc_{i}", neu_sugar, p100, 0, neu_slnc=[i])


# ---------------------------------------------------------------- analyze
def rates(df, t_run=1.0, n_run=30):
    """뉴런별 평균 발화율(Hz) Series (trial 평균, 안 쏜 trial은 0)."""
    cnt = df.groupby("flywire_id").size()
    return cnt / (t_run * n_run)


def mn9_stats(df, t_run=1.0, n_run=30):
    m = df[df.flywire_id == ID_MN9]
    per_trial = np.zeros(n_run)
    for t, g in m.groupby("trial"):
        per_trial[int(t)] = len(g) / t_run
    return {
        "mn9_rate_mean": float(per_trial.mean()),
        "mn9_rate_std": float(per_trial.std()),
        "mn9_trials_active": int((per_trial > 0).sum()),
    }


def summarize(df, neu_sugar=None):
    r = rates(df)
    s = {
        "n_spikes": int(len(df)),
        "n_active_neurons": int(df.flywire_id.nunique()),
        "n_active_per_trial_mean": float(df.groupby("trial").flywire_id.nunique().mean()),
    }
    s.update(mn9_stats(df))
    s["top10"] = [(int(k), float(v)) for k, v in r.sort_values(ascending=False).head(10).items()]
    if neu_sugar is not None:
        s["sugar_grn_mean_rate"] = float(r.reindex(neu_sugar).fillna(0).mean())  # ≈ 자극 주파수
    return s, r


def compare(r_a, r_b, min_rate=0.0):
    """두 발화율 벡터 비교: 활성 집합 Jaccard, 발화율 Pearson 상관(합집합 기준)."""
    a = set(r_a[r_a > min_rate].index)
    b = set(r_b[r_b > min_rate].index)
    jac = len(a & b) / len(a | b) if (a | b) else float("nan")
    idx = sorted(a | b)
    va = r_a.reindex(idx).fillna(0).values
    vb = r_b.reindex(idx).fillna(0).values
    corr = float(np.corrcoef(va, vb)[0, 1]) if len(idx) > 1 else float("nan")
    return {"jaccard": float(jac), "rate_corr": corr, "n_union": len(idx), "n_inter": len(a & b)}


def cmd_analyze():
    ids = json.loads((ROOT / "data" / "paper_ids.json").read_text(encoding="utf-8"))
    neu_sugar = ids["lists"]["neu_sugar"]
    out = {"experiments": {}, "seed_variation": {}, "vs_author": {}}
    files = sorted(OUT.glob("*.parquet"))
    R = {}
    for f in files:
        df = pd.read_parquet(f)
        s, r = summarize(df, neu_sugar)
        out["experiments"][f.stem] = s
        R[f.stem] = r

    # 시드 간 편차 (같은 실험, seed i vs seed 0 및 모든 쌍)
    for exp in ["sugarR_150Hz", "sugarR_100Hz"]:
        keys = [k for k in R if k.startswith(exp + "_seed")]
        pairs = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                c = compare(R[a], R[b])
                c["pair"] = [a, b]
                pairs.append(c)
        if pairs:
            out["seed_variation"][exp] = {
                "jaccard_min": min(p["jaccard"] for p in pairs),
                "jaccard_mean": float(np.mean([p["jaccard"] for p in pairs])),
                "rate_corr_min": min(p["rate_corr"] for p in pairs),
                "rate_corr_mean": float(np.mean([p["rate_corr"] for p in pairs])),
                "mn9_rate_by_seed": [out["experiments"][k]["mn9_rate_mean"] for k in keys],
                "pairs": pairs,
            }

    # 저자 결과(리포 동봉 results/example)와 비교
    author = {
        "sugarR_200Hz": "sugarR.parquet",  # 저자 파일은 당 GRN 발화율로 볼 때 200Hz 자극
        "sugarR_100Hz": "sugarR_100Hz.parquet",
    }
    for exp, fname in author.items():
        p = PATH_AUTHOR / fname
        if not p.is_file():
            continue
        dfa = pd.read_parquet(p)
        sa, ra = summarize(dfa, neu_sugar)
        out["vs_author"][exp] = {"author": sa, "compare_seed0": compare(R[f"{exp}_seed0"], ra) if f"{exp}_seed0" in R else None}
    for f in PATH_AUTHOR.glob("sugarR-*.parquet"):
        nid = f.stem.split("-")[1]
        ours = f"sugarR_100Hz_slnc_{nid}"
        if ours + "_seed0" in R:
            dfa = pd.read_parquet(f)
            sa, ra = summarize(dfa, neu_sugar)
            out["vs_author"][ours] = {"author": sa, "compare_seed0": compare(R[ours + "_seed0"], ra)}

    (OUT / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print_markdown(out)


def print_markdown(out):
    print("| 실험 | 당 GRN 평균 Hz | 총 스파이크 | 활성 뉴런(30 trial 합) | trial당 활성 | MN9 Hz (mean±std) | MN9 발화 trial |")
    print("|---|---|---|---|---|---|---|")
    for k, s in out["experiments"].items():
        print(f"| {k} | {s['sugar_grn_mean_rate']:.0f} | {s['n_spikes']:,} | {s['n_active_neurons']} | {s['n_active_per_trial_mean']:.1f} | "
              f"{s['mn9_rate_mean']:.1f} ± {s['mn9_rate_std']:.1f} | {s['mn9_trials_active']}/30 |")
    print()
    for exp, v in out["seed_variation"].items():
        print(f"- {exp} 시드 5개 쌍별: Jaccard min {v['jaccard_min']:.3f} / mean {v['jaccard_mean']:.3f}, "
              f"rate corr min {v['rate_corr_min']:.4f} / mean {v['rate_corr_mean']:.4f}, "
              f"MN9 Hz by seed {['%.1f' % x for x in v['mn9_rate_by_seed']]}")
    print()
    for exp, v in out["vs_author"].items():
        a = v["author"]
        c = v["compare_seed0"]
        line = (f"- {exp} vs 저자: 저자 당 GRN {a['sugar_grn_mean_rate']:.0f} Hz, MN9 {a['mn9_rate_mean']:.1f}±{a['mn9_rate_std']:.1f} Hz, "
                f"활성 {a['n_active_neurons']}, trial당 활성 {a['n_active_per_trial_mean']:.1f}, 스파이크 {a['n_spikes']:,}")
        if c:
            line += f" | seed0 비교 Jaccard {c['jaccard']:.3f}, rate corr {c['rate_corr']:.4f}"
        print(line)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": cmd_run, "analyze": cmd_analyze}[cmd]()
