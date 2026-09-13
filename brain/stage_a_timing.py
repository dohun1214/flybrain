"""Stage A 사전 점검: 원본 Brian2 모델 1 trial 소요 시간·codegen 타깃 확인.

실행: .venv-brian2\\Scripts\\python.exe brain\\stage_a_timing.py [t_run_ms]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / "external" / "Drosophila_brain_model"
sys.path.insert(0, str(SHIU))

import brian2  # noqa: E402
import pandas as pd  # noqa: E402
from brian2 import prefs  # noqa: E402

import model  # noqa: E402  (Shiu 원본)

PATH_COMP = SHIU / "2023_03_23_completeness_630_final.csv"
PATH_CON = SHIU / "2023_03_23_connectivity_630_final.parquet"
ID_MN9 = 720575940660219265


def main():
    ids = json.loads((ROOT / "data" / "paper_ids.json").read_text(encoding="utf-8"))
    neu_sugar = ids["lists"]["neu_sugar"]

    df_comp = pd.read_csv(PATH_COMP, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    exc = [flyid2i[n] for n in neu_sugar]

    print("brian2", brian2.__version__, "codegen target:", prefs.codegen.target)
    params = dict(model.default_params)
    t_run = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    params["t_run"] = t_run * brian2.ms

    brian2.seed(0)
    t0 = time.time()
    spk = model.run_trial(exc, [], [], PATH_COMP, PATH_CON, params)
    dt = time.time() - t0
    n_spk = sum(len(v) for v in spk.values())
    print(f"t_run={t_run} ms  wall={dt:.1f}s  active_neurons={len(spk)}  spikes={n_spk}")
    print("MN9 spikes:", len(spk.get(flyid2i[ID_MN9], [])))


if __name__ == "__main__":
    main()
