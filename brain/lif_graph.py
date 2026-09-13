"""고정 형상(fixed-shape) LIF 스텝 + CUDA graph 캡처 — brain/lif.py 와 같은 의미론, 런치·동기화 오버헤드 제거.

lif.py 의 eager 스텝은 `nonzero`·`int(lens.sum())` 같은 데이터 의존 형상 때문에 스텝마다 CPU↔GPU 동기화가 생겨
활성 뉴런 수와 무관하게 1.5~2 ms/step (0.05~0.07× 실시간)이었다 (scripts/bench_brain.py).
여기서는 스파이크 압축·시냅스 전개를 고정 크기 버퍼(S_MAX 스파이크/스텝, E_MAX 시냅스/스텝)로 하고,
CUDA 에서는 K 스텝을 하나의 CUDA graph 로 캡처해 replay 한다. 예산 초과는 overflow 플래그로 감지한다.

    from brain.lif_graph import LIFBrainGraph
    brain = LIFBrainGraph("630", device="cuda", seed=0)   # CPU 도 가능 (graph 없이 fixed-shape 만)
    brain.set_poisson(idx, 150); spikes = brain.run(1.0)  # lif.py 와 같은 인터페이스·반환 형식
"""
from __future__ import annotations

import numpy as np
import torch

from brain.lif import LIFBrain, Snapshot


class LIFBrainGraph(LIFBrain):
    def __init__(self, version="630", device="cuda", seed=0, dtype=torch.float64, npz_path=None,
                 s_max: int = 2048, e_max: int = 1 << 17, chunk_steps: int = 100):
        super().__init__(version, device, seed, dtype, npz_path)
        self.s_max, self.e_max, self.K = s_max, e_max, chunk_steps
        n, dev = self.n, self.device
        # 더미 뉴런 N (출력 없음) 을 위한 indptr 확장
        nnz = int(self.indptr[-1])
        self.indptr_ext = torch.cat([self.indptr, torch.tensor([nnz], device=dev, dtype=self.indptr.dtype)])
        self.arange_n = torch.arange(n, device=dev)
        self.arange_e = torch.arange(e_max, device=dev)
        self.arange_e_mod_n = self.arange_e % n
        self.k_t = torch.zeros((), dtype=torch.int64, device=dev)
        self.k_local = torch.zeros((), dtype=torch.int64, device=dev)
        self.spk_ext = torch.full((s_max + 1,), n, dtype=torch.int64, device=dev)  # 마지막 칸 = 더미
        self.rec = torch.full((chunk_steps, s_max), -1, dtype=torch.int32, device=dev)
        self.overflow = torch.zeros((), dtype=torch.bool, device=dev)
        self.buf_flat = self.buf.view(-1)
        self.poisson_static: list[tuple[torch.Tensor, torch.Tensor]] = []  # (idx, p 텐서)
        self.inject_events: torch.Tensor | None = None  # (T, len(idx)) bool — 테스트용 고정 이벤트
        self.inject_idx: torch.Tensor | None = None
        self._graph = None
        self.use_graph = self.device.type == "cuda"
        # RNG: CUDA graph 안에서는 기본 CUDA 생성기를 쓴다
        self.seed(seed)

    # ---------------------------------------------------------------- 설정
    def seed(self, seed: int):
        super().seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))

    def reset_state(self):
        super().reset_state()
        self.k_t.zero_()
        self.k_local.zero_()
        self.overflow.zero_()

    def set_poisson(self, indices, rate_hz, clear=True):
        super().set_poisson(indices, rate_hz, clear)
        self.poisson_static = [(idx, torch.full((idx.numel(),), p, device=self.device, dtype=torch.float32))
                               for idx, p in self.poisson]
        self._graph = None  # 자극 구성이 바뀌면 재캡처

    def _rand(self, n):
        if self.device.type == "cuda":
            return torch.rand(n, device=self.device, dtype=torch.float32)
        return torch.rand(n, generator=self.gen, device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- 고정 형상 스텝
    @torch.no_grad()
    def _fixed_step(self):
        n, D = self.n, self.delay_steps
        k = self.k_t
        not_ref = (k - self.last_spike) >= self.rfc_steps
        v_new = self.v_0 + (self.v - self.v_0) * self.a_v + self.g * self.c_vg
        g_new = self.g * self.a_g
        self.v.copy_(torch.where(not_ref, v_new, self.v))
        self.g.copy_(torch.where(not_ref, g_new, self.g))
        spiked = (self.v > self.v_th) & not_ref
        self.last_spike.copy_(torch.where(spiked, k.expand_as(self.last_spike), self.last_spike))
        recv = not_ref & ~spiked
        slot = torch.remainder(k, D).view(1)
        row = self.buf.index_select(0, slot).view(-1)
        self.g.add_(torch.where(recv, row.to(self.dtype), 0.0))
        self.buf.index_fill_(0, slot, 0.0)
        for idx, p in self.poisson_static:
            kick = self._rand(idx.numel()) < p
            self.v.index_add_(0, idx, self.poisson_w * (kick & recv[idx]).to(self.dtype))
        if self.inject_events is not None:
            kick = self.inject_events.index_select(0, k.view(1)).view(-1)
            self.v.index_add_(0, self.inject_idx, self.poisson_w * (kick & recv[self.inject_idx]).to(self.dtype))
        # 스파이크 압축 (고정 형상)
        pos = torch.cumsum(spiked.to(torch.int64), 0) - 1
        n_spk = pos[-1] + 1
        self.spk_ext.fill_(n)
        target = torch.where(spiked & (pos < self.s_max), pos, torch.full_like(pos, self.s_max))
        self.spk_ext.scatter_(0, target, self.arange_n)
        self.spk_ext[self.s_max:].fill_(n)
        spk = self.spk_ext[: self.s_max]
        # 시냅스 전개 (고정 예산 E_MAX)
        starts = self.indptr_ext[spk]
        lens = self.indptr_ext[spk + 1] - starts
        cum = torch.cumsum(lens, 0)
        total = cum[-1]
        r = torch.searchsorted(cum, self.arange_e, right=True).clamp_(max=self.s_max - 1)
        edge = starts[r] + (self.arange_e - (cum[r] - lens[r]))
        valid = self.arange_e < total
        edge = torch.where(valid, edge, torch.zeros_like(edge))
        # 무효 칸은 가중치 0 을 서로 다른 주소(e % n)에 더한다 — 한 주소에 몰면 원자 연산 경합으로 0.58 ms/step 걸림
        tgt = torch.where(valid, self.indices[edge], self.arange_e_mod_n)
        w = torch.where(valid, self.weights[edge], torch.zeros_like(self.weights[edge]))
        self.buf_flat.index_add_(0, slot * n + tgt, w)   # 도착 슬롯 (k+D)%D == k%D
        self.overflow.logical_or_((n_spk > self.s_max) | (total > self.e_max))
        # 리셋 · 기록
        self.v.copy_(torch.where(spiked, torch.full_like(self.v, self.v_rst), self.v))
        self.g.copy_(torch.where(spiked, torch.zeros_like(self.g), self.g))
        self.rec.index_copy_(0, self.k_local.view(1), spk.to(torch.int32).view(1, -1))
        self.k_t.add_(1)
        self.k_local.add_(1)

    def _chunk(self):
        for _ in range(self.K):
            self._fixed_step()

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        snap = self.snapshot()
        with torch.cuda.stream(s):
            self._chunk()  # warmup (상태를 바꾸므로 뒤에서 복원)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._chunk()
        self.restore(snap)
        self._graph = g

    # ---------------------------------------------------------------- 실행
    @torch.no_grad()
    def run(self, duration_s: float, record: bool = True) -> torch.Tensor:
        n_steps = int(round(duration_s * 1e3 / self.dt))
        assert n_steps % self.K == 0, f"duration 은 chunk({self.K} 스텝 = {self.K * self.dt} ms) 의 배수여야 함"
        if self.use_graph and self._graph is None:
            self._capture()
        out = []
        for _ in range(n_steps // self.K):
            self.k_local.zero_()
            if self.use_graph:
                self._graph.replay()
            else:
                self._chunk()
            if record:
                out.append(self.rec.clone())
        if bool(self.overflow):
            raise RuntimeError(f"스파이크/시냅스 예산 초과 (s_max={self.s_max}, e_max={self.e_max}) — 값을 키워서 다시 실행")
        if not record:
            self.step += n_steps
            return torch.zeros((0, 2), dtype=torch.int64)
        rec = torch.cat(out).cpu()                       # (n_steps, s_max)
        steps = torch.arange(rec.shape[0]).view(-1, 1).expand_as(rec) + self.step
        m = (rec >= 0) & (rec < self.n)   # 더미(n) 제외
        self.step += n_steps
        return torch.stack([steps[m].to(torch.int64), rec[m].to(torch.int64)], 1)

    # ---------------------------------------------------------------- 스냅샷 (RNG 는 CUDA 기본 생성기)
    def snapshot(self) -> Snapshot:
        rng = torch.cuda.get_rng_state() if self.device.type == "cuda" else self.gen.get_state()
        return Snapshot(self.step, self.v.clone(), self.g.clone(), self.last_spike.clone(), self.buf.clone(),
                        rng, list(self.silenced))

    def restore(self, s: Snapshot):
        self.step = s.step
        self.v.copy_(s.v)
        self.g.copy_(s.g)
        self.last_spike.copy_(s.last_spike)
        self.buf.copy_(s.buf)
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(s.rng_state)
        else:
            self.gen.set_state(s.rng_state)
        self.unsilence()
        if s.silenced:
            self.silence(s.silenced)
        self.k_t.fill_(s.step)
        self.overflow.zero_()
