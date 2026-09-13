"""하강뉴런 채널 정의 초안 생성 (이슈 #7) → brain/channels.json

주석 TSV(v783 root_id 기준) 의 cell_type 으로 집단을 뽑고, v783/v630 인덱스를 함께 기록한다.
논문 ID 기반 집단(MN9, aBN1, aDN1, aDN2, 당/쓴맛/물 GRN, JON)은 data/paper_ids.json 에서 온다.
각 채널의 `status` 는 계획서 라벨 규칙을 따른다 — 여기서 정한 채널은 전부 "DN 발화율 → 몸 명령" 디코더의 입력
후보일 뿐이며, LIF 모델에서 실제로 그 행동을 내는지는 2~3주차 [검증 과제].

실행 (.venv): python brain/build_channels.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.ids import Connectome  # noqa: E402

# cell_type 기반 채널 (FlyWire 주석 cell_type 명칭). evidence 는 문헌 근거, 모델 검증은 별도.
TYPE_CHANNELS = {
    "forward": dict(cell_types=["DNp09"], behavior="전진 보행 (계획서 Level B)",
                    status="근거 있음/모델 미검증",
                    evidence="Bidaye et al. 2020 Neuron: DNp09 활성 → 전진 보행. 단 Zacarias et al. 2018 Nat Commun 은 DNp09 를 "
                             "freezing(정지) 뉴런으로 보고(silencing 시 freezing 소실, running 은 유지) — 자극 세기·문맥 의존. "
                             "Sapkal et al. 2024 Nature 도 DNp09(P9) 를 walking-promotion 노드로 분류. 디코더에서 '전진' 으로 읽되 정지 채널과 충돌 가능성 기록"),
    "turn_left": dict(cell_types=["DNa01", "DNa02"], side="left", behavior="회전 (좌측 DN 활성 → 방향은 2주차 실측으로 결정)",
                      status="근거 있음/모델 미검증",
                      evidence="Rayshubskiy et al. 2020 bioRxiv / Chen et al. 2018: DNa01·DNa02 는 조향 DN. 좌/우 비대칭 발화가 회전 방향을 결정. "
                               "어느 쪽 DN 이 어느 방향 회전인지는 문헌마다 표기가 달라 FlyGym 신호 부호는 2주차에 실험으로 맞춘다"),
    "turn_right": dict(cell_types=["DNa01", "DNa02"], side="right", behavior="회전 (우측)", status="근거 있음/모델 미검증",
                       evidence="turn_left 와 동일"),
    "escape": dict(cell_types=["DNp01"], behavior="도망 점프 (Giant Fiber; FlyWire 명칭 DNp01)", status="근거 있음/모델 미검증",
                   evidence="Giant Fiber 회로 (von Reyn et al. 2014 등). LC4/LPLC2 → GF → 점프"),
    "backward": dict(cell_types=["MDN"], behavior="후진 (Moonwalker DN)", status="근거 있음/모델 미검증",
                     evidence="Bidaye et al. 2014 Science: MDN 활성 → 후진 보행. FlyGym 음수 신호로 후진 가능 (1주차 확인)"),
    "looming_LC4": dict(cell_types=["LC4"], behavior="시각 looming 입력 (인코더가 자극할 집단)", status="근거 있음/모델 미검증",
                        evidence="LC4 는 looming 크기/속도 반응 시각 투사 뉴런 (Ache et al. 2019). GF 에 직접 입력"),
    "looming_LPLC2": dict(cell_types=["LPLC2"], behavior="시각 looming 입력 (인코더가 자극할 집단)", status="근거 있음/모델 미검증",
                          evidence="LPLC2 는 방사 확장(looming) 선택적 시각 투사 뉴런 (Klapoetke et al. 2017). GF 에 입력"),
}
# 정지 DN: 미확정
STOP_NOTE = ("정지(halt) DN 은 미확정 [검증 과제]. 후보: Sapkal et al. 2024 Nature 의 Foxglove(FG)·Bluebell(BB) — SEZ 에서 내려가는 GABA성 "
             "하강뉴런으로 walking-promotion DN(DNp09 포함)을 억제. FlyWire cell_type 명칭은 논문 보충자료에서 확인해야 함 "
             "(주석 TSV synonyms 에 없음). DNp09 자체도 문맥에 따라 freezing 을 낸다는 보고가 있어(Zacarias 2018) 정지 채널 설계 시 함께 고려")

# 논문 ID 기반 채널 (data/paper_ids.json / data/ids.py)
PAPER_CHANNELS = {
    "proboscis_MN9": dict(list="MN9", behavior="주둥이 신전 운동뉴런 (Level A 1순위 출력)", status="검증됨"),
    "groom_aBN1": dict(list="aBN1", behavior="더듬이 그루밍 개재뉴런", status="검증됨 (논문)"),
    "groom_aDN1": dict(list="aDN1", behavior="더듬이 그루밍 하강뉴런 (DNg62)", status="검증됨 (논문)"),
    "groom_aDN2": dict(list="aDN2", behavior="더듬이 그루밍 하강뉴런 (DNge078)", status="검증됨 (논문)"),
    "input_sugar_GRN_R": dict(list="neu_sugar", behavior="당 GRN 우측 (인코더 입력)", status="검증됨"),
    "input_sugar_GRN_L": dict(list="neu_sugar_left", behavior="당 GRN 좌측 (인코더 입력)", status="검증됨 (논문 ID)"),
    "input_bitter_GRN": dict(list="neu_bitter", behavior="쓴맛 GRN (P1 이후)", status="검증됨 (논문 ID)"),
    "input_water_GRN": dict(list="neu_water", behavior="물 GRN (P1 이후)", status="검증됨 (논문 ID)"),
    "input_JON_CE": dict(list="neu_JON_CE", behavior="JON-CE (더듬이 기계감각 → 그루밍)", status="검증됨 (논문 ID)"),
    "input_JON_F": dict(list="neu_JON_F", behavior="JON-F", status="검증됨 (논문 ID)"),
}


def main():
    c783, c630 = Connectome("783"), Connectome("630")
    annot = c783.annot  # v783 인덱스 순서
    out = {"_note": "채널 = 디코더/인코더가 읽거나 자극할 뉴런 집단. root_id 는 FlyWire v783 (paper_* 는 v630 논문 ID). "
                    "status 는 계획서 라벨. 모델에서 행동을 내는지는 2~3주차 검증.",
           "_stop_dn": STOP_NOTE, "channels": {}}
    rows = []
    for name, spec in TYPE_CHANNELS.items():
        m = annot.cell_type.isin(spec["cell_types"])
        if "side" in spec:
            m &= annot.side == spec["side"]
        sub = annot[m]
        rids = [int(r) for r in sub.root_id]
        idx783 = [int(i) for i in sub.index]
        idx630, miss630 = c630.indices_of(rids, strict=False)
        out["channels"][name] = dict(
            source="annotation cell_type", cell_types=spec["cell_types"], side=spec.get("side", "both"),
            behavior=spec["behavior"], status=spec["status"], evidence=spec["evidence"],
            n=len(rids), n_by_side={k: int(v) for k, v in sub.side.value_counts().items()},
            root_ids_v783=rids, idx_v783=idx783, idx_v630=[int(i) for i in idx630],
            n_missing_v630=len(miss630), missing_v630_root_ids=miss630)
        rows.append((name, ", ".join(spec["cell_types"]), spec.get("side", "both"), len(rids), len(idx630), len(miss630), spec["status"]))
    for name, spec in PAPER_CHANNELS.items():
        i630 = [int(i) for i in c630.lists[spec["list"]]]
        rids630 = [int(c630.root_ids[i]) for i in i630]
        i783 = [int(i) for i in c783.lists[spec["list"]]]
        out["channels"][name] = dict(
            source=f"data/paper_ids.json:{spec['list']}", behavior=spec["behavior"], status=spec["status"],
            n=len(rids630), root_ids_v630=rids630, idx_v630=i630, idx_v783=i783,
            n_missing_v783=len(c783.missing[spec["list"]]), missing_v783_root_ids=c783.missing[spec["list"]])
        rows.append((name, f"paper:{spec['list']}", "-", len(i783), len(i630), len(rids630) - len(i783), spec["status"]))
    (ROOT / "brain" / "channels.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("| 채널 | 정의 | side | v783 뉴런 수 | v630 뉴런 수 | 누락 | 상태 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
