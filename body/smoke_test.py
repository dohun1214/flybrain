"""FlyGym 2.1.0 스모크 테스트 (이슈 #5).

이 PC에서 우리가 쓸 API가 실제로 동작하는지 확인하고 결과를 demo_output/smoke/ 에 저장한다.

실행 (.venv):
    python body/smoke_test.py            # 전부
    python body/smoke_test.py turning    # 개별: turning | backward | vision | sites | joints

각 테스트는 독립적으로 실패할 수 있다. 결과 요약은 demo_output/smoke/summary.json.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path("demo_output/smoke")
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {}


def _fly_and_world(name, *, vision=False, sites=(), colorize=True, terrain="flat"):
    """튜토리얼 4d와 같은 구성의 파리 + 세계. vision/sites 옵션 추가."""
    from flygym import Simulation
    from flygym.anatomy import AnatomicalJoint, ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import make_locomotion_fly

    fly = make_locomotion_fly(name=name, add_adhesion=True, colorize=colorize)
    if vision:
        fly.add_vision()
    if sites:
        fly.add_joint_sites([AnatomicalJoint(p, c) for p, c in sites])
    body_cam = fly.add_tracking_camera(
        name="body_cam",
        pos_offset=(-0.5, -7.5, 0.0),
        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
        fovy=35.0,
    )
    if terrain == "flat":
        world = FlatGroundWorld()
    else:
        from flygym.compose.world.complex_terrain import BlocksTerrainWorld

        world = BlocksTerrainWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    return fly, world, sim, body_cam


def _run_walk(tag, signal_fn, run_time=2.0, render=True):
    """HybridTurningController로 run_time초 보행. signal_fn(t) -> (2,) 하강 신호."""
    from flygym.anatomy import BodySegment
    from flygym_demo.complex_terrain import (
        HybridControllerObservation,
        HybridTurningController,
        LocomotionAction,
        PreprogrammedSteps,
        apply_locomotion_action,
    )

    fly, world, sim, body_cam = _fly_and_world(tag)
    renderer = None
    if render:
        renderer = sim.set_renderer([body_cam], camera_res=(240, 320), playback_speed=0.1, output_fps=25)

    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dof_order)

    sim.reset()
    ctrl.reset(seed=0)
    apply_locomotion_action(
        sim, fly.name,
        LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dof_order), adhesion_onoff=np.ones(6, dtype=bool)),
    )
    sim.warmup()

    n = int(run_time / sim.timestep)
    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    pos = np.zeros((n, 3), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    t_phys = 0.0
    t0 = time.perf_counter()
    for k in range(n):
        sig = np.asarray(signal_fn(k * sim.timestep), dtype=float)
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        action = ctrl.step(sig, obs)
        apply_locomotion_action(sim, fly.name, action)
        tp = time.perf_counter()
        sim.step()
        t_phys += time.perf_counter() - tp
        pos[k] = sim.get_body_positions(fly.name)[thorax_idx]
        quat[k] = sim.get_body_rotations(fly.name)[thorax_idx]
        if renderer is not None:
            sim.render_as_needed()
    wall = time.perf_counter() - t0

    res = {
        "timestep": sim.timestep,
        "n_steps": n,
        "n_actuated_dofs": len(dof_order),
        "wall_s": wall,
        "physics_s": t_phys,
        "realtime_x_total": run_time / wall,
        "realtime_x_physics_only": run_time / t_phys,
        "thorax_displacement_mm": (pos[-1] - pos[0]).round(3).tolist(),
        "thorax_z_min_max": [float(pos[:, 2].min()), float(pos[:, 2].max())],
    }
    # 몸통 heading 기준 전진 성분 (초기 heading = +x)
    q0 = quat[0]
    w, x, y, z = q0
    heading0 = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    res["forward_along_initial_heading_mm"] = float(np.dot(pos[-1] - pos[0], heading0))
    if renderer is not None:
        renderer.save_video(OUT / f"{tag}.mp4")
        res["video"] = str(OUT / f"{tag}.mp4")
        res["n_frames"] = len(renderer.frames[body_cam.name])
    np.save(OUT / f"{tag}_thorax_pos.npy", pos)
    return res


def test_turning():
    """튜토리얼 4d: 1초 [1.2,0.4] → 1초 [0.4,1.2]."""
    return _run_walk("turning", lambda t: [1.2, 0.4] if t < 1.0 else [0.4, 1.2])


def test_backward():
    """음수 신호 [-1,-1] 2초. 비교용으로 [1,1] 전진도 같이 잰다."""
    fwd = _run_walk("forward_1_1", lambda t: [1.0, 1.0])
    bwd = _run_walk("backward_-1_-1", lambda t: [-1.0, -1.0])
    return {"forward": fwd, "backward": bwd,
            "backward_moves_backward": bwd["forward_along_initial_heading_mm"] < 0}


def test_vision():
    """add_vision → get_raw_vision / get_ommatidia_readouts. 이미지 저장 + 렌더 시간."""
    import imageio.v3 as iio

    fly, world, sim, _ = _fly_and_world("vision", vision=True)
    sim.reset()
    sim.warmup()
    t0 = time.perf_counter()
    raw = sim.get_raw_vision(fly.name)
    t_raw = time.perf_counter() - t0
    t0 = time.perf_counter()
    omm = sim.get_ommatidia_readouts(fly.name)
    t_omm = time.perf_counter() - t0
    # 두 번째 호출(초기화 제외)
    t0 = time.perf_counter()
    for _ in range(10):
        omm = sim.get_ommatidia_readouts(fly.name)
    t_omm10 = (time.perf_counter() - t0) / 10

    retina = sim.retina
    res = {
        "raw_vision_shape": list(raw.shape), "raw_dtype": str(raw.dtype),
        "raw_min_max": [float(raw.min()), float(raw.max())],
        "ommatidia_shape": list(omm.shape),
        "n_ommatidia_per_eye": int(retina.num_ommatidia_per_eye),
        "t_first_raw_s": t_raw, "t_first_ommatidia_s": t_omm, "t_ommatidia_avg_s": t_omm10,
    }
    for i, side in enumerate(["left", "right"]):
        img = raw[i]
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        iio.imwrite(OUT / f"eye_{side}_raw.png", img)
        # yellow/pale 중 0이 아닌 값을 합쳐 1채널로 → 육각 격자 사람용 이미지
        val = omm[i].sum(axis=1)
        hexa = retina.hex_pxls_to_human_readable(val, color_8bit=True)
        iio.imwrite(OUT / f"eye_{side}_hex.png", hexa)
    return res


def test_sites():
    """더듬이·주둥이·앞다리 사이트 위치 + 몸 부위 접촉력."""
    from flygym.anatomy import ALL_SEGMENT_NAMES

    sites = [
        ("c_head", "l_pedicel"), ("c_head", "r_pedicel"),
        ("l_pedicel", "l_funiculus"), ("r_pedicel", "r_funiculus"),
        ("c_head", "c_rostrum"), ("c_rostrum", "c_haustellum"),
        ("c_thorax", "lf_coxa"), ("c_thorax", "rf_coxa"),
    ]
    fly, world, sim, _ = _fly_and_world("sites", sites=sites)
    sim.reset()
    sim.warmup()
    positions = sim.get_site_positions(fly.name)
    order = [j.name for j in fly.get_sites_order()]
    res = {"site_positions_mm": {n: positions[i].round(3).tolist() for i, n in enumerate(order)}}
    segs = ["l_funiculus", "r_funiculus", "c_haustellum", "lf_tarsus5", "rf_tarsus5", "lf_tibia"]
    t0 = time.perf_counter()
    f = sim.get_bodysegment_contact_forces(fly.name, segs, ground_only=True)
    res["t_contact_forces_s"] = time.perf_counter() - t0
    res["contact_forces_ground"] = {s: f[i].round(4).tolist() for i, s in enumerate(segs)}
    res["n_all_segment_names"] = len(ALL_SEGMENT_NAMES)
    return res


def test_joints():
    """관절 구조 확인: 기본(LEGS_ONLY) 파리와 ALL_POSSIBLE 골격으로 만든 파리의 MJCF 관절 목록."""
    import mujoco as mj
    from flygym.anatomy import AxisOrder, JointPreset, Skeleton
    from flygym.compose import NeuroMechFly

    def joints_of(fly):
        model = fly.mjcf_root.compile()  # mjcf_root는 mujoco.MjSpec
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        bodies = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        return names, bodies

    res = {}
    from flygym_demo.complex_terrain import make_locomotion_fly

    fly = make_locomotion_fly(name="j_default")
    jn, bn = joints_of(fly)
    res["default_locomotion_fly"] = {"n_joints": len(jn), "n_bodies": len(bn)}

    fly2 = NeuroMechFly(name="j_all")
    sk = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_POSSIBLE)
    fly2.add_joints(sk)
    jn2, bn2 = joints_of(fly2)
    keys = ["rostrum", "haustellum", "wing", "haltere", "abdomen", "pedicel", "funiculus", "arista", "head", "eye"]
    res["all_possible_skeleton"] = {
        "n_joints": len(jn2),
        "joints_by_keyword": {k: [j for j in jn2 if k in j] for k in keys},
        "bodies_by_keyword": {k: [b for b in bn2 if b and k in b] for k in keys},
    }
    fly3 = NeuroMechFly(name="j_bio")
    fly3.add_joints(Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL))
    jn3, _ = joints_of(fly3)
    res["all_biological_skeleton"] = {"n_joints": len(jn3), "joints": jn3}
    return res


TESTS = {
    "turning": test_turning,
    "backward": test_backward,
    "vision": test_vision,
    "sites": test_sites,
    "joints": test_joints,
}


def main():
    names = sys.argv[1:] or list(TESTS)
    for n in names:
        print(f"=== {n}", flush=True)
        t0 = time.perf_counter()
        try:
            r = TESTS[n]()
            r = {"ok": True, "result": r}
        except Exception:
            r = {"ok": False, "error": traceback.format_exc()}
        r["elapsed_s"] = time.perf_counter() - t0
        SUMMARY[n] = r
        print(json.dumps(r, indent=1, ensure_ascii=False, default=str), flush=True)
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
