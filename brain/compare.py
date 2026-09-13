"""Stage B 검증: PyTorch LIF(brain/lif.py) vs Brian2 정답(data/ground_truth/, 이슈 #2) — 이슈 #4.

실행 (.venv):
    python brain/compare.py quick       # 1 trial 1초 150Hz: 요약 + 속도
    python brain/compare.py snapshot    # snapshot/restore 결정성 + 소요 시간
    python brain/compare.py full        # 30 trial × {100,150,200}Hz + silencing 3종 → 정답과 비교, 결과 json/markdown

정답과 같은 지표: 총 스파이크, 활성 뉴런(30 trial 합), trial당 활성, MN9 Hz(mean±std), 활성 집합 Jaccard, 발화율 상관.
합격 기준(docs/benchmark-week1.md Stage A 시드 편차 기준): MN9 30/30, MN9 Hz 시드 범위 ±2, Jaccard ≥ 0.93, 상관 ≥ 0.999.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.lif import LIFBrain  # noqa: E402
from brain.stage_a_brian2 import ID_MN9, compare, rates, summarize  # noqa: E402 (지표 함수 재사용)
from data.ids import Connectome  # noqa: E402

GT = ROOT / "data" / "ground_truth"
OUT = ROOT / "data" / "stage_b"
N_RUN = 30
AUTHOR_SLNC = [720575940617937543, 720575940621754367, 720575940622695448]


def run_trials(brain: LIFBrain, targets, rate, exp_name, seed=0, n_run=N_RUN, slnc=(), log=print):
    frames, walls = [], []
    for k in range(n_run):
        brain.reset_state()
        brain.seed(seed * 1000 + k)
        brain.unsilence()
        brain.set_poisson(targets, rate)
        if len(slnc):
            brain.silence(slnc)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spk = brain.run(1.0)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        frames.append(brain.spikes_to_frame(spk, k, exp_name))
        if k == 0:
            log(f"  {exp_name} trial0: {len(spk)} spikes, {spk[:, 1].unique().numel()} active, "
                f"MN9 {(spk[:, 1] == MN9_IDX).sum().item()}, {walls[-1]:.1f}s")
    return pd.concat(frames, ignore_index=True), float(np.mean(walls))


def cmd_quick():
    c = Connectome("630")
    brain = LIFBrain("630", seed=0)
    brain.set_poisson(c.lists["neu_sugar"], 150)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spk = brain.run(1.0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_mn9 = int((spk[:, 1] == MN9_IDX).sum())
    print(f"150Hz 1s: {len(spk)} spikes, {spk[:, 1].unique().numel()} active, MN9 {n_mn9}, "
          f"wall {wall:.2f}s ({1 / wall:.2f}x realtime), VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    print("  (Brian2 seed0 trial0 참고: 13,698 spikes / 375 active / MN9 80)")
    sugar_rate = np.mean([(spk[:, 1] == i).sum().item() for i in c.lists["neu_sugar"]])
    print(f"  sugar GRN mean rate {sugar_rate:.0f} Hz")


def cmd_snapshot():
    c = Connectome("630")
    brain = LIFBrain("630", seed=0)
    brain.set_poisson(c.lists["neu_sugar"], 150)
    brain.run(0.3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    snap = brain.snapshot()
    torch.cuda.synchronize()
    t_snap = time.perf_counter() - t0
    a = brain.run(0.2)
    t0 = time.perf_counter()
    brain.restore(snap)
    torch.cuda.synchronize()
    t_rest = time.perf_counter() - t0
    b = brain.run(0.2)
    same = a.shape == b.shape and bool(torch.equal(a, b))
    # restore 후 silence 하면 달라져야 함
    brain.restore(snap)
    brain.silence(c.lists["neu_sugar"][:5])
    d = brain.run(0.2)
    diff = not (d.shape == a.shape and bool(torch.equal(a, d)))
    res = dict(snapshot_ms=t_snap * 1e3, restore_ms=t_rest * 1e3, restore_reproduces_identical_spikes=same,
               silence_after_restore_changes_result=diff, n_spikes_normal=int(len(a)), n_spikes_silenced=int(len(d)))
    print(json.dumps(res, indent=1))
    return res


def cmd_full():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Connectome("630")
    brain = LIFBrain("630", seed=0)
    sugar = c.lists["neu_sugar"]
    results = {}
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    plan = [("sugarR_100Hz", 100, ()), ("sugarR_150Hz", 150, ()), ("sugarR_200Hz", 200, ())]
    plan += [(f"sugarR_100Hz_slnc_{i}", 100, (c.index_of(i),)) for i in AUTHOR_SLNC]
    for exp, rate, slnc in plan:
        df, wall = run_trials(brain, sugar, rate, exp, slnc=slnc, log=log)
        df.to_parquet(OUT / f"{exp}_torch.parquet")
        s, r = summarize(df, [int(x) for x in c.root_ids[sugar]])
        s["wall_per_trial_s"] = wall
        entry = {"torch": s}
        # 정답(Brian2) 파일들과 비교: 같은 실험 이름의 모든 시드
        gts = sorted(GT.glob(f"{exp}_seed*.parquet"))
        entry["vs_brian2"] = {}
        for gpath in gts:
            gdf = pd.read_parquet(gpath)
            gs, gr = summarize(gdf, [int(x) for x in c.root_ids[sugar]])
            cmp = compare(r, gr)
            cmp["brian2_mn9_rate"] = gs["mn9_rate_mean"]
            cmp["brian2_n_spikes"] = gs["n_spikes"]
            entry["vs_brian2"][gpath.stem] = cmp
        if slnc:
            entry["silenced_neuron_rate_hz"] = float((df.flywire_id == c.root_ids[slnc[0]]).sum() / N_RUN)
        results[exp] = entry
        mn9_gt = [v["brian2_mn9_rate"] for v in entry["vs_brian2"].values()]
        jac = [v["jaccard"] for v in entry["vs_brian2"].values()]
        cor = [v["rate_corr"] for v in entry["vs_brian2"].values()]
        log(f"{exp}: torch {s['n_spikes']:,} spikes, {s['n_active_neurons']} active, MN9 {s['mn9_rate_mean']:.1f}±{s['mn9_rate_std']:.1f} Hz "
            f"({s['mn9_trials_active']}/30) | Brian2 MN9 {min(mn9_gt):.1f}~{max(mn9_gt):.1f} Hz | "
            f"Jaccard {min(jac):.3f}~{max(jac):.3f} | corr {min(cor):.4f}~{max(cor):.4f} | {wall:.1f}s/trial")

    res = {"snapshot": cmd_snapshot(), "experiments": results,
           "vram_mb": torch.cuda.max_memory_allocated() / 2**20}
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "compare.log").write_text("\n".join(log_lines), encoding="utf-8")
    print_markdown(res)


def print_markdown(res):
    print("\n| 실험 | PyTorch 스파이크 | 활성 | MN9 Hz | Brian2 MN9 Hz (시드 범위) | Jaccard (시드별 min~max) | 발화율 상관 |")
    print("|---|---|---|---|---|---|---|")
    for exp, e in res["experiments"].items():
        s = e["torch"]
        v = list(e["vs_brian2"].values())
        mn9 = [x["brian2_mn9_rate"] for x in v]
        jac = [x["jaccard"] for x in v]
        cor = [x["rate_corr"] for x in v]
        print(f"| {exp} | {s['n_spikes']:,} | {s['n_active_neurons']} | {s['mn9_rate_mean']:.1f} ± {s['mn9_rate_std']:.1f} ({s['mn9_trials_active']}/30) | "
              f"{min(mn9):.1f} ~ {max(mn9):.1f} | {min(jac):.3f} ~ {max(jac):.3f} | {min(cor):.4f} ~ {max(cor):.4f} |")


if __name__ == "__main__":
    MN9_IDX = Connectome("630").index_of(ID_MN9)
    {"quick": cmd_quick, "snapshot": cmd_snapshot, "full": cmd_full}[sys.argv[1] if len(sys.argv) > 1 else "quick"]()
