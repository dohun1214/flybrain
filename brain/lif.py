"""PyTorch CUDA LIF 뇌 (Shiu et al. 2024 모델, 원본 model.py 의미론 그대로) — 이슈 #4.

원본(Brian2, dt 0.1 ms, method='linear')과 같은 한 스텝의 순서:
  1. 불응기 판정  not_ref = (k - last_spike) >= rfc_steps        (Brian2: timestep(t-lastspike) >= timestep(rfc))
  2. 상태 갱신    v, g 를 정확해(exp)로 dt 만큼 진행 — not_ref 인 뉴런만 (unless refractory)
  3. 임계값      spiked = (v > v_th) & not_ref ; last_spike[spiked] = k
  4. 시냅스 슬롯  g += (18스텝 전 스파이크의 시냅스 입력)  /  포아송 대상: v += 68.75 mV × Bernoulli(rate·dt)
                 — 둘 다 not_ref & ~spiked 인 뉴런에만 (아래 참고). 이번 스텝 스파이크의 출력 시냅스를 링버퍼 (k+18) 슬롯에 적립
  5. 리셋        v[spiked] = v_rst ; g[spiked] = 0

**불응기 중인 뉴런(이번 스텝에 스파이크한 뉴런 포함)에 도착한 시냅스 입력·포아송 입력은 버려진다** —
Brian2 가 `(unless refractory)` 변수 v, g 에 conditional write 를 걸기 때문 (`g += w` 도 not_refractory 인 뉴런에만 적용).
Brian2 로 결정적 동치 테스트를 해서 확인함 (2026-09-13). 포아송 대상 뉴런은 불응기 0 이라 영향 없음. silencing 은 해당 뉴런의 **출력 시냅스만** 0 (원본 `syn.w['i == n'] = 0`).

    from brain.lif import LIFBrain
    brain = LIFBrain("630", device="cuda", seed=0)
    brain.set_poisson(indices, rate_hz=150)
    brain.silence([idx])            # / brain.unsilence()
    snap = brain.snapshot()         # brain.restore(snap)
    spikes = brain.run(1.0)         # -> (step, neuron_index) int64 텐서 (CPU)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# 원논문 파라미터 (model.py default_params) — 바꾸지 않는다
PARAMS = dict(
    dt_ms=0.1,
    v_0=-52.0,      # mV resting
    v_rst=-52.0,    # mV reset
    v_th=-45.0,     # mV threshold
    t_mbr=20.0,     # ms membrane time constant
    tau=5.0,        # ms synaptic time constant
    t_rfc=2.2,      # ms refractory
    t_dly=1.8,      # ms synaptic delay
    w_syn=0.275,    # mV per synapse
    f_poi=250,      # Poisson weight factor -> 68.75 mV per event
)


@dataclass
class Snapshot:
    step: int
    v: torch.Tensor
    g: torch.Tensor
    last_spike: torch.Tensor
    buf: torch.Tensor
    rng_state: torch.Tensor
    silenced: list = field(default_factory=list)


class LIFBrain:
    def __init__(self, version: str = "630", device: str = "cuda", seed: int = 0, dtype=torch.float64,
                 npz_path: Path | None = None):
        self.device = torch.device(device)
        self.dtype = dtype
        p = PARAMS
        z = np.load(npz_path or PROCESSED / f"v{version}.npz")
        self.root_ids = z["root_ids"]
        self.n = len(self.root_ids)
        # CSR (행 = presynaptic)
        self.indptr = torch.as_tensor(z["indptr"], device=self.device)
        self.indices = torch.as_tensor(z["indices"].astype(np.int64), device=self.device)
        self.weights0 = torch.as_tensor(z["weights_mv"], device=self.device, dtype=torch.float32)
        self.weights = self.weights0.clone()  # silencing 은 이 복사본만 수정

        # 스텝 상수
        self.dt = p["dt_ms"]
        self.rfc_steps_default = int(round(p["t_rfc"] / self.dt))   # 22
        self.delay_steps = int(round(p["t_dly"] / self.dt))         # 18
        self.v_0, self.v_rst, self.v_th = p["v_0"], p["v_rst"], p["v_th"]
        self.poisson_w = p["w_syn"] * p["f_poi"]                    # 68.75 mV
        # 정확해 계수:  g' = g·a_g ,  v' = v0 + (v-v0)·a_v + g·c
        tm, ts = p["t_mbr"], p["tau"]
        self.a_v = math.exp(-self.dt / tm)
        self.a_g = math.exp(-self.dt / ts)
        self.c_vg = ts / (ts - tm) * (self.a_g - self.a_v)

        # 상태
        self.v = torch.full((self.n,), self.v_0, device=self.device, dtype=dtype)
        self.g = torch.zeros(self.n, device=self.device, dtype=dtype)
        self.last_spike = torch.full((self.n,), -10**9, device=self.device, dtype=torch.int64)
        self.rfc_steps = torch.full((self.n,), self.rfc_steps_default, device=self.device, dtype=torch.int64)
        self.buf = torch.zeros((self.delay_steps, self.n), device=self.device, dtype=torch.float32)
        self.step = 0
        self.gen = torch.Generator(device=self.device)
        self.seed(seed)
        self.poisson: list[tuple[torch.Tensor, float]] = []  # (indices, p per step)
        self.inject = None  # 테스트용: callable(step) -> (indices, bool mask) — 포아송 대신 정해진 이벤트 주입
        self.silenced: list[int] = []

    # ------------------------------------------------------------ 설정
    def seed(self, seed: int):
        self.gen.manual_seed(int(seed))

    def reset_state(self):
        self.v.fill_(self.v_0)
        self.g.zero_()
        self.last_spike.fill_(-10**9)
        self.buf.zero_()
        self.step = 0

    def set_poisson(self, indices, rate_hz: float, clear: bool = True):
        """indices 뉴런에 rate_hz 포아송 입력 (이벤트당 68.75 mV, 대상 불응기 0)."""
        if clear:
            self.rfc_steps.fill_(self.rfc_steps_default)
            self.poisson = []
        idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        self.rfc_steps[idx] = 0
        self.poisson.append((idx, rate_hz * self.dt * 1e-3))

    def silence(self, indices):
        """출력 시냅스만 0 (원본과 동일). 뉴런 자신은 계속 발화할 수 있다."""
        for i in np.asarray(indices, dtype=np.int64).tolist():
            a, b = int(self.indptr[i]), int(self.indptr[i + 1])
            self.weights[a:b] = 0.0
            self.silenced.append(i)

    def unsilence(self):
        self.weights.copy_(self.weights0)
        self.silenced = []

    # ------------------------------------------------------------ 스냅샷
    def snapshot(self) -> Snapshot:
        return Snapshot(self.step, self.v.clone(), self.g.clone(), self.last_spike.clone(), self.buf.clone(),
                        self.gen.get_state(), list(self.silenced))

    def restore(self, s: Snapshot):
        self.step = s.step
        self.v.copy_(s.v)
        self.g.copy_(s.g)
        self.last_spike.copy_(s.last_spike)
        self.buf.copy_(s.buf)
        self.gen.set_state(s.rng_state)
        self.unsilence()
        if s.silenced:
            self.silence(s.silenced)

    # ------------------------------------------------------------ 실행
    @torch.no_grad()
    def step_once(self) -> torch.Tensor:
        k = self.step
        not_ref = (k - self.last_spike) >= self.rfc_steps
        # 2. 정확해 상태 갱신 (불응기 뉴런은 동결)
        v_new = self.v_0 + (self.v - self.v_0) * self.a_v + self.g * self.c_vg
        g_new = self.g * self.a_g
        self.v = torch.where(not_ref, v_new, self.v)
        self.g = torch.where(not_ref, g_new, self.g)
        # 3. 임계값
        spiked = (self.v > self.v_th) & not_ref
        self.last_spike = torch.where(spiked, torch.full_like(self.last_spike, k), self.last_spike)
        # 4. 시냅스 슬롯: 18스텝 전 스파이크 전달 + 포아송 + 이번 스파이크 적립
        #    Brian2 는 `(unless refractory)` 변수에 conditional write 를 건다: 불응기 중(이번 스텝에 스파이크한
        #    뉴런 포함)인 뉴런에 대한 `g += w`, `v += kick` 는 **버려진다** (원본 model.py 의 실제 동작).
        recv = not_ref & ~spiked
        slot = k % self.delay_steps
        self.g += torch.where(recv, self.buf[slot].to(self.dtype), 0.0)
        self.buf[slot].zero_()
        for idx, p in self.poisson:
            kick = torch.rand(idx.numel(), generator=self.gen, device=self.device) < p
            self.v[idx] += self.poisson_w * (kick & recv[idx]).to(self.dtype)
        if self.inject is not None:
            idx, kick = self.inject(k)
            self.v[idx] += self.poisson_w * (kick & recv[idx]).to(self.dtype)
        rows = spiked.nonzero(as_tuple=False).flatten()
        if rows.numel():
            starts = self.indptr[rows]
            lens = self.indptr[rows + 1] - starts
            total = int(lens.sum())
            if total:
                offs = torch.repeat_interleave(starts - torch.cumsum(lens, 0) + lens, lens)
                flat = torch.arange(total, device=self.device) + offs
                self.buf[(k + self.delay_steps) % self.delay_steps].index_add_(
                    0, self.indices[flat], self.weights[flat])
        # 5. 리셋
        self.v = torch.where(spiked, torch.full_like(self.v, self.v_rst), self.v)
        self.g = torch.where(spiked, torch.zeros_like(self.g), self.g)
        self.step += 1
        return rows

    @torch.no_grad()
    def run(self, duration_s: float, record: bool = True) -> torch.Tensor:
        """duration_s 동안 실행. 반환: (n_spikes, 2) int64 [step, neuron] (CPU)."""
        n_steps = int(round(duration_s * 1e3 / self.dt))
        out = []
        for _ in range(n_steps):
            rows = self.step_once()
            if record and rows.numel():
                out.append(torch.stack([torch.full_like(rows, self.step - 1), rows], 1))
        if not out:
            return torch.zeros((0, 2), dtype=torch.int64)
        return torch.cat(out).cpu()

    # ------------------------------------------------------------ 편의
    def spikes_to_frame(self, spikes: torch.Tensor, trial: int, exp_name: str):
        """Stage A parquet 과 같은 스키마: t(초), trial, flywire_id, exp_name."""
        import pandas as pd

        s = spikes.numpy()
        return pd.DataFrame({
            "t": s[:, 0] * self.dt * 1e-3,
            "trial": trial,
            "flywire_id": self.root_ids[s[:, 1]],
            "exp_name": exp_name,
        })
