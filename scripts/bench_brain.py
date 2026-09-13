"""뇌 커널 성능 측정 (이슈 #6, 계획서 §7): 생물 1초당 실제 소요, 활성 뉴런 수별, VRAM, 구간별 오버헤드.

실행 (.venv):  python scripts/bench_brain.py  [--steps 2000]
출력: data/bench/brain.json + stdout 표
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.lif import LIFBrain  # noqa: E402
from data.ids import Connectome  # noqa: E402

OUT = ROOT / "data" / "bench"


def timed_run(brain, steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_spk = 0
    active = set()
    for _ in range(steps):
        rows = brain.step_once()
        if rows.numel():
            n_spk += rows.numel()
            active.update(rows.tolist())
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    bio = steps * brain.dt * 1e-3
    return dict(steps=steps, wall_s=wall, bio_s=bio, realtime_x=bio / wall, ms_per_step=1e3 * wall / steps,
                n_spikes=n_spk, spikes_per_step=n_spk / steps, n_active=len(active))


def bench_scale(steps):
    """활성 뉴런 수를 늘려가며 측정: 당 GRN(≈400 활성) + 무작위 뉴런 M개에 50 Hz 포아송."""
    c = Connectome("630")
    rng = np.random.default_rng(0)
    res = []
    for extra in [0, 1_000, 10_000, 50_000]:
        brain = LIFBrain("630", seed=0)
        brain.set_poisson(c.lists["neu_sugar"], 150)
        if extra:
            brain.set_poisson(rng.choice(brain.n, extra, replace=False), 50, clear=False)
        timed_run(brain, 200)  # warmup
        brain.reset_state()
        r = timed_run(brain, steps)
        r["extra_poisson_neurons"] = extra
        r["vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res.append(r)
        print(f"extra={extra:>6}: active {r['n_active']:>6}, {r['spikes_per_step']:.1f} spk/step, "
              f"{r['ms_per_step']:.3f} ms/step, {r['realtime_x']:.3f}x realtime, VRAM {r['vram_mb']:.0f} MB", flush=True)
        del brain
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return res


def bench_parts(steps):
    """구간별: (a) 자극 없음(elementwise + 런치 오버헤드만) (b) 당 GRN 자극(gather 포함) (c) elementwise 를 CUDA graph 로 캡처."""
    c = Connectome("630")
    out = {}
    brain = LIFBrain("630", seed=0)
    out["no_stimulus_eager"] = timed_run(brain, steps)

    brain.set_poisson(c.lists["neu_sugar"], 150)
    timed_run(brain, 200)
    brain.reset_state()
    out["sugar_eager"] = timed_run(brain, steps)

    # elementwise 부분만 CUDA graph 캡처 (스파이크 없이) → 런치 오버헤드 하한
    brain2 = LIFBrain("630", seed=0)
    v, g = brain2.v, brain2.g
    last = brain2.last_spike
    k_t = torch.zeros((), dtype=torch.int64, device="cuda")
    static_spiked = torch.zeros(brain2.n, dtype=torch.bool, device="cuda")

    def elementwise():
        not_ref = (k_t - last) >= brain2.rfc_steps
        v_new = brain2.v_0 + (v - brain2.v_0) * brain2.a_v + g * brain2.c_vg
        g_new = g * brain2.a_g
        v.copy_(torch.where(not_ref, v_new, v))
        g.copy_(torch.where(not_ref, g_new, g))
        spiked = (v > brain2.v_th) & not_ref
        last.copy_(torch.where(spiked, k_t.expand_as(last), last))
        static_spiked.copy_(spiked)
        v.copy_(torch.where(spiked, torch.full_like(v, brain2.v_rst), v))
        g.copy_(torch.where(spiked, torch.zeros_like(g), g))
        k_t.add_(1)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            elementwise()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        elementwise()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        graph.replay()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    out["elementwise_cuda_graph"] = dict(steps=steps, wall_s=wall, ms_per_step=1e3 * wall / steps,
                                         realtime_x=steps * 1e-4 / wall)
    for k, r in out.items():
        print(f"{k:24s}: {r['ms_per_step']:.3f} ms/step, {r['realtime_x']:.3f}x realtime", flush=True)
    return out


def bench_vram():
    out = {}
    for ver in ["630", "783"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        b = LIFBrain(ver, seed=0)
        b.set_poisson(Connectome(ver).lists["neu_sugar"], 150)
        timed_run(b, 500)
        out[ver] = dict(n=b.n, n_edges=int(b.indices.numel()), vram_alloc_mb=torch.cuda.memory_allocated() / 2**20,
                        vram_peak_mb=torch.cuda.max_memory_allocated() / 2**20)
        print(f"v{ver}: {out[ver]}", flush=True)
        del b
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    res["parts"] = bench_parts(a.steps)
    res["scale"] = bench_scale(a.steps)
    res["vram"] = bench_vram()
    (OUT / "brain.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
