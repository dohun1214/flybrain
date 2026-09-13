"""lif_graph.py (고정 형상 + CUDA graph) 검증·측정: 결정성, snapshot/restore, 활성 수별 속도, 통계가 eager/Brian2 와 맞는지.

실행 (.venv): python scripts/bench_brain_graph.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.lif_graph import LIFBrainGraph  # noqa: E402
from brain.stage_a_brian2 import ID_MN9  # noqa: E402
from data.ids import Connectome  # noqa: E402

OUT = ROOT / "data" / "bench"


def timed(brain, dur):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spk = brain.run(dur)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return spk, wall


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Connectome("630")
    mn9 = c.index_of(ID_MN9)
    sugar = c.lists["neu_sugar"]
    res = {}

    # 1. 결정성: 같은 시드 두 번 → 동일, 다른 시드 → 다름
    b = LIFBrainGraph("630", seed=0)
    b.set_poisson(sugar, 150)
    s1, w_first = timed(b, 1.0)            # 첫 호출은 graph 캡처 포함
    b.reset_state(); b.seed(0)
    s2, w1 = timed(b, 1.0)
    b.reset_state(); b.seed(1)
    s3, _ = timed(b, 1.0)
    res["determinism"] = dict(same_seed_identical=bool(s1.shape == s2.shape and torch.equal(s1, s2)),
                              diff_seed_differs=not (s1.shape == s3.shape and torch.equal(s1, s3)),
                              n_spikes=[int(len(s1)), int(len(s2)), int(len(s3))],
                              mn9=[int((s[:, 1] == mn9).sum()) for s in (s1, s2, s3)],
                              wall_first_incl_capture_s=w_first, wall_s=w1, realtime_x=1.0 / w1)
    print("determinism:", res["determinism"], flush=True)

    # 2. snapshot / restore
    b.reset_state(); b.seed(0)
    b.run(0.3)
    torch.cuda.synchronize(); t0 = time.perf_counter(); snap = b.snapshot(); torch.cuda.synchronize(); t_s = time.perf_counter() - t0
    a = b.run(0.2)
    t0 = time.perf_counter(); b.restore(snap); torch.cuda.synchronize(); t_r = time.perf_counter() - t0
    a2 = b.run(0.2)
    b.restore(snap); b.silence(sugar[:5]); d = b.run(0.2); b.unsilence()
    res["snapshot"] = dict(snapshot_ms=t_s * 1e3, restore_ms=t_r * 1e3,
                           restore_identical=bool(a.shape == a2.shape and torch.equal(a, a2)),
                           silence_changes=not (a.shape == d.shape and torch.equal(a, d)))
    print("snapshot:", res["snapshot"], flush=True)

    # 3. 활성 수별 속도 (bench_brain.py 와 같은 조건)
    rng = np.random.default_rng(0)
    res["scale"] = []
    for extra in [0, 1_000, 10_000, 50_000]:
        bb = LIFBrainGraph("630", seed=0)
        bb.set_poisson(sugar, 150)
        if extra:
            bb.set_poisson(rng.choice(bb.n, extra, replace=False), 50, clear=False)
        bb.run(0.1); bb.reset_state(); bb.seed(0)
        spk, wall = timed(bb, 1.0)
        r = dict(extra=extra, n_spikes=int(len(spk)), n_active=int(spk[:, 1].unique().numel()),
                 spikes_per_step=len(spk) / 10000, wall_s=wall, realtime_x=1.0 / wall,
                 ms_per_step=wall / 10, vram_mb=torch.cuda.max_memory_allocated() / 2**20)
        res["scale"].append(r)
        print(f"extra={extra:>6}: active {r['n_active']:>6}, {r['spikes_per_step']:.1f} spk/step, "
              f"{r['ms_per_step']:.3f} ms/step, {r['realtime_x']:.2f}x realtime, VRAM {r['vram_mb']:.0f} MB", flush=True)
        del bb; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # 4. 통계: 30 trial 150Hz → Brian2 정답과 비교 (compare.py 의 지표)
    import pandas as pd
    from brain.stage_a_brian2 import compare, summarize
    bb = LIFBrainGraph("630", seed=0)
    bb.set_poisson(sugar, 150)
    frames = []
    t0 = time.perf_counter()
    for k in range(30):
        bb.reset_state(); bb.seed(k)
        frames.append(bb.spikes_to_frame(bb.run(1.0), k, "sugarR_150Hz"))
    wall30 = time.perf_counter() - t0
    df = pd.concat(frames, ignore_index=True)
    sug_ids = [int(x) for x in c.root_ids[sugar]]
    s, r = summarize(df, sug_ids)
    cmp = {}
    for gp in sorted((ROOT / "data" / "ground_truth").glob("sugarR_150Hz_seed*.parquet")):
        gs, gr = summarize(pd.read_parquet(gp), sug_ids)
        cmp[gp.stem] = compare(r, gr) | {"brian2_mn9": gs["mn9_rate_mean"]}
    res["stats_150Hz"] = dict(torch=s, vs_brian2=cmp, wall_30_trials_s=wall30)
    print(f"150Hz 30 trial: {s['n_spikes']:,} spikes, {s['n_active_neurons']} active, MN9 {s['mn9_rate_mean']:.1f}±{s['mn9_rate_std']:.1f} "
          f"({s['mn9_trials_active']}/30) | Jaccard {min(v['jaccard'] for v in cmp.values()):.3f}~{max(v['jaccard'] for v in cmp.values()):.3f} "
          f"| corr {min(v['rate_corr'] for v in cmp.values()):.4f}~ | {wall30:.0f}s / 30 trial", flush=True)

    (OUT / "brain_graph.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
