"""FlyGym 2.1 CPU 성능 측정 (이슈 #6): 구간별 스텝 시간, 평지/험지, 렌더·겹눈 on/off.

실행 (.venv): python scripts/bench_flygym.py  [--sim-s 1.0]
출력: data/bench/flygym.json + stdout
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "bench"


def make(terrain="flat", vision=False, render=False):
    from flygym import Simulation
    from flygym.anatomy import ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (HybridTurningController, LocomotionAction, PreprogrammedSteps,
                                             apply_locomotion_action, make_locomotion_fly)

    fly = make_locomotion_fly(name="bench", add_adhesion=True, colorize=render)
    if vision:
        fly.add_vision()
    cam = fly.add_tracking_camera(name="cam", pos_offset=(-0.5, -7.5, 0.0),
                                  rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=35.0)
    if terrain == "flat":
        world = FlatGroundWorld()
    else:
        from flygym.compose.world.complex_terrain import BlocksTerrainWorld
        world = BlocksTerrainWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
    sim = Simulation(world)
    if render:
        sim.set_renderer([cam], camera_res=(480, 640), playback_speed=1.0, output_fps=30)
    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dof_order)
    sim.reset(); ctrl.reset(seed=0)
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order), adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    return fly, sim, ctrl


def bench(terrain, vision, render, sim_s, vision_every_ms=10.0):
    from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

    fly, sim, ctrl = make(terrain, vision, render)
    n = int(sim_s / sim.timestep)
    every = int(vision_every_ms * 1e-3 / sim.timestep)
    t = dict(obs=0.0, ctrl=0.0, act=0.0, phys=0.0, render=0.0, vision=0.0)
    if vision:
        sim.get_ommatidia_readouts(fly.name)  # 초기화 제외
    t0 = time.perf_counter()
    for k in range(n):
        a = time.perf_counter()
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        b = time.perf_counter(); t["obs"] += b - a
        action = ctrl.step(np.array([1.0, 1.0]), obs)
        c = time.perf_counter(); t["ctrl"] += c - b
        apply_locomotion_action(sim, fly.name, action)
        d = time.perf_counter(); t["act"] += d - c
        sim.step()
        e = time.perf_counter(); t["phys"] += e - d
        if render:
            sim.render_as_needed()
            f = time.perf_counter(); t["render"] += f - e; e = f
        if vision and k % every == 0:
            sim.get_ommatidia_readouts(fly.name)
            t["vision"] += time.perf_counter() - e
    wall = time.perf_counter() - t0
    res = dict(terrain=terrain, vision=vision, render=render, sim_s=sim_s, n_steps=n, wall_s=wall,
               realtime_x=sim_s / wall, us_per_step={k: 1e6 * v / n for k, v in t.items()},
               phys_only_realtime_x=sim_s / t["phys"] if t["phys"] else None)
    if render:
        res["n_frames"] = len(next(iter(sim.renderer.frames.values())))
        res["ms_per_frame"] = 1e3 * t["render"] / max(1, res["n_frames"])
    if vision:
        res["ms_per_vision_call"] = 1e3 * t["vision"] / (n // every)
    sim.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-s", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [("flat", False, False), ("flat", False, True), ("flat", True, False), ("flat", True, True),
               ("blocks", False, False), ("blocks", True, True)]
    results = []
    for terrain, vision, render in configs:
        r = bench(terrain, vision, render, a.sim_s)
        results.append(r)
        u = r["us_per_step"]
        print(f"{terrain:6s} vision={vision!s:5s} render={render!s:5s}: {r['realtime_x']:.3f}x realtime "
              f"(phys only {r['phys_only_realtime_x']:.2f}x) | us/step obs {u['obs']:.0f} ctrl {u['ctrl']:.0f} act {u['act']:.0f} "
              f"phys {u['phys']:.0f} render {u['render']:.0f} vision {u['vision']:.0f}"
              + (f" | {r['ms_per_frame']:.1f} ms/frame" if render else "")
              + (f" | {r['ms_per_vision_call']:.1f} ms/vision" if vision else ""), flush=True)
    (OUT / "flygym.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
