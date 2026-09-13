"""lif_graph.py 스텝의 커널별 시간 (torch.profiler, eager 로 fixed-shape 스텝 200회)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.lif_graph import LIFBrainGraph  # noqa: E402
from data.ids import Connectome  # noqa: E402

c = Connectome("630")
b = LIFBrainGraph("630", seed=0)
b.set_poisson(c.lists["neu_sugar"], 150)
b.use_graph = False
b.run(0.1)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    b.run(0.02)  # 200 steps
    torch.cuda.synchronize()
rows = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:20]
for e in rows:
    name = e.key
    for tag in ["scatter_gather", "index_elementwise", "Bucketization", "DeviceScan", "Indexing_cu", "vectorized_elementwise", "unrolled_elementwise", "elementwise_kernel", "Memcpy", "reduce", "fill"]:
        if tag in name:
            name = tag; break
    print(f"{e.self_device_time_total/1000/200:8.4f} ms/step  {e.count/200:5.1f}/step  {name[:60]}")
print("total CUDA ms/step:", sum(e.self_device_time_total for e in prof.key_averages())/1000/200)
