"""결정적 동치 테스트: 같은 입력 스파이크 패턴을 Brian2(원본 model.py) 와 brain/lif.py(CPU) 에 넣고 스파이크 열을 그대로 비교.

포아송 대신 미리 뽑은 Bernoulli(150Hz·dt) 이벤트를 양쪽에 SpikeGeneratorGroup / brain.inject 로 주입한다
(PoissonInput 과 같은 'synapses' 슬롯, 68.75 mV). RNG 차이를 없앤 상태에서 커널 의미론이 원본과 같은지 검증.

실행 (.venv-brian2 에 torch CPU 설치 필요: uv pip install --python .venv-brian2\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu):
    python brain/test_equivalence.py 1000                                    # 1초, silencing 없음
    python brain/test_equivalence.py 500 720575940622695448,720575940617937543   # 0.5초, 두 뉴런 silence
    python brain/test_equivalence.py 1000 --graph                             # lif_graph.py 고정 형상 스텝 검증
기대 출력: identical: True
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "external" / "Drosophila_brain_model"))
import brian2
from brian2 import prefs, ms, mV, Hz, NeuronGroup, Synapses, SpikeMonitor, Network, SpikeGeneratorGroup
prefs.codegen.target = "numpy"
import pandas as pd
import model
from data.ids import Connectome

args = [a for a in sys.argv[1:] if not a.startswith("--")]
USE_GRAPH = "--graph" in sys.argv   # brain/lif_graph.py (고정 형상 스텝) 검증
T_MS = float(args[0]) if len(args) > 0 else 200.0
SLNC = [int(x) for x in args[1].split(",")] if len(args) > 1 else []   # silence 할 root_id 목록
c = Connectome("630")
sugar = c.lists["neu_sugar"]
rng = np.random.default_rng(0)
# 입력: 각 sugar 뉴런에 150Hz Bernoulli(0.015)/step 이벤트 (미리 뽑음)
n_steps = int(T_MS / 0.1)
events = rng.random((n_steps, len(sugar))) < 0.015
ev_steps, ev_j = np.nonzero(events)
print("input events:", len(ev_steps))

# ---------------- Brian2
P = ROOT / "external" / "Drosophila_brain_model"
params = dict(model.default_params); params["t_run"] = T_MS * ms
neu, syn, spk_mon = model.create_model(P / "2023_03_23_completeness_630_final.csv", P / "2023_03_23_connectivity_630_final.parquet", params)
slnc_idx = [c.index_of(r) for r in SLNC]
syn = model.silence(slnc_idx, syn)
for i in sugar:
    neu[int(i)].rfc = 0 * ms
gen = SpikeGeneratorGroup(len(sugar), ev_j, ev_steps * 0.1 * ms)
inp = Synapses(gen, neu, on_pre="v += w_in", namespace={"w_in": params["w_syn"] * params["f_poi"]}, delay=0 * ms)
inp.connect(i=np.arange(len(sugar)), j=np.asarray(sugar))
net = Network(neu, syn, spk_mon, gen, inp)
t0 = time.time(); net.run(params["t_run"]); print("brian2 wall %.0fs" % (time.time() - t0))
b_i = np.asarray(spk_mon.i); b_t = np.round(np.asarray(spk_mon.t / ms) / 0.1).astype(int)
b = set(zip(b_t.tolist(), b_i.tolist()))
print("brian2 spikes", len(b), "active", len(set(b_i.tolist())), "MN9", int((b_i == c.index_of(720575940660219265)).sum()))

# ---------------- torch kernel (cpu), 같은 이벤트 주입
import torch
from brain.lif import LIFBrain
sugar_t = torch.as_tensor(sugar)
ev_t = torch.as_tensor(events)
if USE_GRAPH:
    from brain.lif_graph import LIFBrainGraph
    brain = LIFBrainGraph("630", device="cpu", seed=0)
    brain.silence(slnc_idx)
    brain.rfc_steps[sugar_t] = 0
    brain.inject_idx, brain.inject_events = sugar_t, ev_t
    t0 = time.time(); spk = brain.run(T_MS * 1e-3).numpy(); print("torch(fixed-shape, cpu) wall %.0fs" % (time.time() - t0))
else:
    brain = LIFBrain("630", device="cpu", seed=0)
    brain.silence(slnc_idx)
    brain.rfc_steps[sugar_t] = 0
    brain.inject = lambda k: (sugar_t, ev_t[k])
    out = []
    for k in range(n_steps):
        rows = brain.step_once()
        if rows.numel():
            out.append(torch.stack([torch.full_like(rows, k), rows], 1))
    spk = torch.cat(out).numpy() if out else np.zeros((0, 2), int)
t_ = set(zip(spk[:, 0].tolist(), spk[:, 1].tolist()))
print("torch spikes", len(t_), "active", len(set(spk[:, 1].tolist())), "MN9", int((spk[:, 1] == c.index_of(720575940660219265)).sum()))
print("identical:", b == t_, "| only brian2:", len(b - t_), "only torch:", len(t_ - b))
for r in SLNC:
    i = c.index_of(r)
    print(f"silenced {r}: brian2 spikes {int((b_i == i).sum())}, torch spikes {int((spk[:, 1] == i).sum())}")
if b != t_:
    ob = sorted(b - t_)[:10]; ot = sorted(t_ - b)[:10]
    print(" first only-brian2:", ob); print(" first only-torch:", ot)
