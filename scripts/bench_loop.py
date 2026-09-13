"""결합 폐루프 처리량 (이슈 #6): 뇌(GPU, CUDA graph 10 ms 청크) + FlyGym(CPU, 100 물리 스텝) + 임시 디코더, 10 ms 마다 교환.

뇌→몸: 하강뉴런 대신 임시로 무작위 뉴런 2집단의 스파이크 수를 좌/우 신호로 (실제 디코더는 2주차).
몸→뇌: 접촉/시각 인코더가 아직 없으므로 당 GRN 포아송 자극을 그대로 (인코더 비용은 겹눈 호출 8 ms 로 대체 측정).

실행 (.venv): python scripts/bench_loop.py [--sim-s 2.0] [--vision]
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
from brain.lif_graph import LIFBrainGraph  # noqa: E402
from data.ids import Connectome  # noqa: E402
from scripts.bench_flygym import make  # noqa: E402

OUT = ROOT / "data" / "bench"


def main(sim_s, vision, render):
    from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

    c = Connectome("630")
    brain = LIFBrainGraph("630", seed=0)                     # chunk 100 steps = 10 ms
    brain.set_poisson(c.lists["neu_sugar"], 150)
    brain.run(0.1)                                           # graph 캡처 + 워밍업
    brain.reset_state(); brain.seed(0)
    rng = np.random.default_rng(0)
    grpL, grpR = rng.choice(brain.n, 500, replace=False), rng.choice(brain.n, 500, replace=False)
    grpL_t, grpR_t = torch.as_tensor(grpL, device="cuda"), torch.as_tensor(grpR, device="cuda")

    fly, sim, ctrl = make("flat", vision=vision, render=render)
    if vision:
        sim.get_ommatidia_readouts(fly.name)
    n_chunks = int(sim_s / 0.01)
    steps_per_chunk = int(0.01 / sim.timestep)
    t = dict(brain=0.0, decode=0.0, body=0.0, vision=0.0)
    signal = np.array([1.0, 1.0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        a = time.perf_counter()
        spk = brain.run(0.01)                                # 뇌 10 ms
        torch.cuda.synchronize()
        b = time.perf_counter(); t["brain"] += b - a
        # 임시 디코더: 집단 스파이크 수 → [0.4, 1.2] 범위 신호
        ids = spk[:, 1]
        nL = int(torch.isin(ids, grpL_t.cpu()).sum()); nR = int(torch.isin(ids, grpR_t.cpu()).sum())
        signal = 0.4 + 0.8 * np.clip(np.array([nL, nR]) / 5.0, 0, 1)
        cc = time.perf_counter(); t["decode"] += cc - b
        for _k in range(steps_per_chunk):                    # 몸 10 ms
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, ctrl.step(signal, obs))
            sim.step()
            if render:
                sim.render_as_needed()
        d = time.perf_counter(); t["body"] += d - cc
        if vision:
            sim.get_ommatidia_readouts(fly.name)
            t["vision"] += time.perf_counter() - d
    wall = time.perf_counter() - t0
    res = dict(sim_s=sim_s, vision=vision, render=render, wall_s=wall, realtime_x=sim_s / wall,
               ms_per_10ms_chunk={k: 1e3 * v / n_chunks for k, v in t.items()},
               brain_alone_realtime_x=sim_s / t["brain"], body_alone_realtime_x=sim_s / t["body"])
    print(json.dumps(res, indent=1), flush=True)
    sim.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-s", type=float, default=2.0)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = main(a.sim_s, a.vision, a.render)
    tag = ("_vision" if a.vision else "") + ("_render" if a.render else "")
    (OUT / f"loop{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
