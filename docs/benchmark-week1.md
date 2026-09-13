# 1주차 측정·검증 결과

측정 환경: Windows 11 · i7-14700K (28 스레드) · 64GB RAM · RTX 4060 Ti 8GB · 2026-09-13
모든 수치는 이 PC에서 직접 실행한 값. 다른 시스템 수치는 쓰지 않는다.

## FlyGym 2.1.0 스모크 테스트 (이슈 #5) — `body/smoke_test.py`

실행: `.venv\Scripts\python.exe body\smoke_test.py` → `demo_output/smoke/` (영상·이미지·summary.json, gitignore)

### 결과 요약

| 항목 | 결과 | 상태 |
|---|---|---|
| 튜토리얼 4d 회전 (`[1.2,0.4]` 1초 → `[0.4,1.2]` 1초) | 실행 성공, 영상 500프레임 저장. 흉부 변위 (x +4.70, y −14.38 mm) — 회전함 | [검증됨] |
| 전진 `[1.0,1.0]` 2초 | 초기 heading 방향 **+26.6 mm** | [검증됨] |
| **후진 `[-1.0,-1.0]` 2초** | 초기 heading 방향 **−20.3 mm**, 자세 유지(흉부 z 0.80~1.25 mm, 전진 때와 같은 범위), 영상에서 넘어짐 없음 | [검증됨] → MDN 후진에 별도 CPG 불필요 |
| `get_raw_vision` | shape (2, 512, 450, 3) uint8, 어안 보정된 좌/우 눈 이미지. 자기 다리·바닥·하늘 보임 | [검증됨] |
| `get_ommatidia_readouts` | shape (2, 721, 2) — 눈당 낱눈 721개, yellow/pale 채널. `Retina.hex_pxls_to_human_readable`로 육각 격자 이미지 생성 | [검증됨] |
| `get_site_positions` (더듬이·주둥이·앞다리 coxa 사이트) | `fly.add_joint_sites([...])` 후 정상 반환. 예: 좌 funiculus (1.082, 0.108, 1.055) mm, rostrum (1.027, 0, 0.745) mm | [검증됨] → 후각 농도 직접 계산 가능 |
| `get_bodysegment_contact_forces` | 바닥 접촉만: 앞다리 tarsus5에 힘 있음, 더듬이·주둥이 0. 호출 0.4 ms | [검증됨] |
| 첫 실행 시 S3 메시 다운로드 | `make_locomotion_fly` 기본은 **번들된 간략 메시(SIMPLIFIED_MAX2000FACES)** 사용 → 다운로드 없이 동작. FULLSIZE 메시만 S3 lazy load | [검증됨] |

### 관절 구조 — CLAUDE.md/plan.md 정정 필요

FlyGym 2.x는 MJCF에 관절이 고정돼 있지 않고 `fly.add_joints(Skeleton(joint_preset=...))`로 **프로그램이 관절을 생성**한다.

| 골격 프리셋 | MJCF 관절 수 | 내용 |
|---|---|---|
| `make_locomotion_fly` 기본 (`LEGS_ONLY`) | 66 (능동 42 + 수동 tarsus 24) | 다리만 |
| `ALL_BIOLOGICAL` | 126 | 다리 66 + 머리·더듬이·눈·주둥이·날개·평균곤·복부 각 3축 |
| `ALL_POSSIBLE` | 204 | 모든 연결 쌍 3축 |

`ALL_BIOLOGICAL`/`ALL_POSSIBLE`에서 실제로 생성된 관절: `c_head-c_rostrum-{yaw,pitch,roll}`, `c_rostrum-c_haustellum-*`, `c_thorax-l_wing-*`, `c_thorax-r_wing-*`, `c_thorax-{l,r}_haltere-*`, `c_thorax-c_abdomen12-*` … `c_abdomen5-c_abdomen6-*`, `c_head-{l,r}_pedicel-*`, `*_pedicel-*_funiculus-*`, `*_funiculus-*_arista-*`, `c_head-{l,r}_eye-*`.

→ **"주둥이·날개·복부에 관절 없음"은 틀렸다.** 주둥이(rostrum/haustellum) 관절은 골격 프리셋만 바꾸면 생긴다. 다만 이 관절들에 **미리 만들어진 동작(preprogrammed step)이나 액추에이터 튜닝은 없으므로** 주둥이 신전 시퀀스는 우리가 각도 궤적을 직접 설계해야 한다. 날개도 관절은 생기지만 비행 물리는 없다.
(2026-09-13 확인. 이전 "관절 87개" 수치는 어떤 프리셋에서 센 것인지 불명 — 위 표로 대체)

### 속도 (평지, 다리 파리, 렌더 240×320 @ 25fps, 0.1× 재생)

| 구간 | 값 |
|---|---|
| 물리 timestep | 0.1 ms (1초 = 10,000 스텝) |
| `sim.step()`만 | 시뮬 2초에 1.8~2.0 s → **약 1.0× 실시간** |
| 전체 루프 (관측 `HybridControllerObservation.from_sim` + 컨트롤러 + 액션 + 스텝 + 몸 위치 읽기 + 렌더) | 시뮬 2초에 13.6~15.1 s → **0.13~0.15× 실시간** |
| `get_ommatidia_readouts` (첫 호출) | 5.0 s (렌더러·Retina 초기화, numba JIT 포함) |
| `get_ommatidia_readouts` (이후 평균) | **8.1 ms** / 호출 (좌우 눈 렌더 + 어안 보정 + 낱눈 샘플링) |
| `get_bodysegment_contact_forces` (6 부위) | 0.4 ms |

→ 물리 자체는 실시간이지만 컨트롤러/관측 오버헤드가 10배 크다. 이슈 #6에서 구간별로 쪼개서 측정하고, 겹눈은 10 ms 주기(=100 스텝마다) 호출 시 스텝당 0.08 ms로 부담 없음.

## Stage A — Shiu 원본 Brian2, v630 당 GRN → MN9 재현 (이슈 #2) — `brain/stage_a_brian2.py`

환경 `.venv-brian2` (Python 3.10, brian2 2.5.1, numpy 1.24.4, numpy 코드 생성 타깃), 원본 `model.py`를 그대로 import.
원본과 다른 점은 trial마다 `brian2.seed(seed*1000+k)`를 고정한 것뿐. 30 trial × 1초, loky 12 워커 → 실험 1개당 약 125~135초.
결과 parquet: `data/ground_truth/` (gitignore), 요약: `data/ground_truth/summary.json`.

### 저자 동봉 결과 파일의 자극 주파수 (발견)

리포 `results/example/sugarR.parquet`에서 당 GRN 21개의 평균 발화율은 **197 Hz** → 이 파일은 코드 기본값(150 Hz)이 아니라 **200 Hz**로 생성된 것 (노트북 본문 "excited at 200 Hz"와 일치). `sugarR_100Hz.parquet`과 `sugarR-<id>.parquet`(silencing)은 99 Hz. 따라서 저자 결과와 직접 비교하려면 200 Hz 조건이 필요해 추가로 실행했다.

### 결과 (이 PC, 2026-09-13)

| 실험 | 당 GRN 평균 Hz | 총 스파이크 | 활성 뉴런(30 trial 합) | trial당 활성 | MN9 Hz (mean±std) | MN9 발화 trial |
|---|---|---|---|---|---|---|
| sugarR_100Hz seed0 | 99 | 295,571 | 405 | 358.5 | 69.6 ± 3.6 | 30/30 |
| sugarR_100Hz seed1 | 100 | 292,266 | 410 | 357.3 | 67.0 ± 4.0 | 30/30 |
| sugarR_100Hz seed2 | 99 | 290,090 | 403 | 355.0 | 66.5 ± 4.1 | 30/30 |
| sugarR_100Hz seed3 | 99 | 291,525 | 414 | 354.9 | 66.6 ± 4.1 | 30/30 |
| sugarR_100Hz seed4 | 99 | 289,808 | 397 | 355.5 | 67.2 ± 4.7 | 30/30 |
| sugarR_150Hz seed0 | 148 | 409,883 | 431 | 379.5 | 83.5 ± 4.6 | 30/30 |
| sugarR_150Hz seed1 | 148 | 409,755 | 433 | 380.0 | 83.0 ± 5.3 | 30/30 |
| sugarR_150Hz seed2 | 148 | 410,949 | 441 | 382.3 | 83.0 ± 5.7 | 30/30 |
| sugarR_150Hz seed3 | 148 | 410,703 | 436 | 379.2 | 83.6 ± 5.9 | 30/30 |
| sugarR_150Hz seed4 | 148 | 411,229 | 434 | 377.8 | 84.5 ± 4.3 | 30/30 |
| sugarR_200Hz seed0 | 197 | 509,390 | 450 | 403.9 | 92.8 ± 5.5 | 30/30 |
| **저자 sugarR.parquet (200 Hz)** | 197 | 511,566 | 448 | 402.6 | 93.3 ± 3.2 | 30/30 |
| **저자 sugarR_100Hz.parquet** | 99 | 289,073 | 404 | 354.3 | 67.0 ± 6.6 | 30/30 |

→ **당 GRN → MN9 재현 [검증됨].** 200 Hz 조건에서 저자 파일과 총 스파이크 0.4% 차이, MN9 92.8 vs 93.3 Hz, 활성 집합 Jaccard 0.969, 뉴런별 발화율 상관 0.9998. 100 Hz도 MN9 66.5~69.6 vs 67.0 Hz, Jaccard 0.935, 상관 0.9995.

### 시드 간 편차 (Stage B 합격 기준의 근거)

| 조건 | 활성 집합 Jaccard (5시드 쌍별 min / mean) | 발화율 Pearson 상관 (min / mean) | MN9 Hz 범위 |
|---|---|---|---|
| 150 Hz | 0.953 / 0.969 | 0.9997 / 0.9998 | 83.0 ~ 84.5 |
| 100 Hz | 0.931 / 0.954 | 0.9995 / 0.9996 | 66.5 ~ 69.6 |

Jaccard가 0.93~0.97에 그치는 이유는 "30 trial 중 한 번이라도 쏜 뉴런" 집합에 드물게 쏘는 뉴런이 섞이기 때문(trial당 활성 수는 ±3 안에서 안정). 발화율 상관은 0.9995 이상.

**Stage B 합격 기준 제안 (시드 편차 기준으로 조정)**: 동일 조건에서 PyTorch 결과가 Brian2 5시드 분포 안에 들 것 — MN9 30/30 trial 발화, MN9 평균 Hz가 시드 범위 ±2 Hz 이내, 활성 집합 Jaccard ≥ 0.93, 발화율 상관 ≥ 0.999 (계획서 초안 0.9 / 0.95보다 엄격하게 잡아도 된다).

### silencing 정답 (Brain Surgery 검증용, 100 Hz, seed0)

| silenced 뉴런 (당 GRN) | 우리 MN9 Hz | 저자 MN9 Hz | Jaccard / 상관 (vs 저자) | silenced 뉴런 자신의 발화율 |
|---|---|---|---|---|
| 720575940617937543 | 64.8 ± 5.4 | 63.3 ± 4.7 | 0.911 / 0.9997 | 99.4 Hz (baseline 99.2) |
| 720575940621754367 | 67.0 ± 4.2 | 66.9 ± 6.1 | 0.976 / 0.9997 | 98.7 Hz (baseline 99.1) |
| 720575940622695448 | 70.9 ± 5.6 | 71.2 ± 4.4 | 0.962 / 0.9995 | 115.0 Hz (baseline 114.9) |
| 720575940620900446 (우리 top-3) | 61.6 ± 4.2 | — | — | — |
| 720575940629888530 (우리 top-3) | 63.8 ± 5.3 | — | — | — |

- 저자와 같은 3개 뉴런 silencing 결과가 시드 편차 안에서 일치 [검증됨].
- **silencing 의미 확인**: silenced 뉴런 자신은 계속 발화한다(출력 시냅스만 0). Stage B의 `silence()`도 같은 결과를 내야 한다 — 이 표가 정답.
- 당 GRN 1개를 끄면 MN9는 몇 Hz만 변한다(저자 결과와 동일). 시연용 Brain Surgery 프리셋으로는 효과가 작다 → FEEDING 프리셋 집단 선정은 계획대로 [검증 과제].
- 참고: `720575940622695448`은 100 Hz 포아송 자극인데 115 Hz로 발화 — 재귀 흥분 입력을 받는 GRN.

### 검증 메모
- numpy 타깃 vs Cython 타깃 결과 동일성: Claude 클라우드 환경(gcc 있음)에서 같은 시드로 1 trial 실행 → 두 타깃 모두 13,698 스파이크·375 활성·MN9 80으로 **완전히 동일**. 이 PC의 `sugarR_150Hz_seed0` trial 0(같은 시드 0)도 13,698 / 375 / 80 — 기기·타깃에 무관하게 시드가 결과를 결정한다. 이 PC의 numpy 타깃 결과를 그대로 써도 된다.
- 원본 코드는 그대로이며(2023-05-09 이후 model.py 변경 없음, 데이터 파일 동일), 저자 결과와의 차이는 자극 주파수(200 vs 150 Hz)뿐이었다.

## 데이터 파이프라인 (이슈 #3) — `data/build_csr.py`, `data/ids.py`

`.venv`에서 `python data/build_csr.py` → `data/processed/v{630,783}.npz` (CSR, 행 = presynaptic) + `.annot.parquet` (gitignore), 메타데이터 `data/v{630,783}.meta.json` (git 추적). 빌드 시간 각 6초. 빌드 중 `*_Index` ↔ `*_ID` 일관성, 중복 edge 없음, 계획서 §3.2 기대값을 assert로 재검증한다.

| | v630 | v783 |
|---|---|---|
| 뉴런 | 127,400 | 138,639 |
| 뉴런쌍 edge (임계값 없음) | 14,687,178 | 15,091,983 |
| 시냅스 합 | 52,793,639 | 54,492,922 |
| ≥5 시냅스 연결 | 2,614,028 | 2,700,513 |
| 흥분 / 억제 edge | 8,800,532 / 5,886,646 | 9,059,302 / 6,032,681 |
| 최대 출력 차수 | 9,615 | 9,783 |
| 출력 없는 뉴런 / 입력 없는 뉴런 | 385 / 607 | 634 / 1,549 |
| 주석 TSV 매칭 (flywire_annotations, 139,248행) | **106,214 / 127,400 = 83.37%** | **138,625 / 138,639 = 99.99%** |
| npz 크기 | 184 MB | 189 MB |

→ 계획서 수치와 전부 일치 [검증됨]. 시뮬 사용 edge 수 = 위 edge 수 전부(임계값 없음).

논문 ID 목록(`data/paper_ids.json`) → 인덱스 (`data/ids.py Connectome`): v630은 전 목록 100% 존재. v783은 당 GRN 20/21, 쓴맛 20/21, JON-CE 69/70, 좌측 당 GRN 9/10 (나머지·MN9·aBN1·aDN1·aDN2 전부 존재) — 계획서 §3.1 생존률과 일치. MN9 = v630 idx 127193 (`CB0701`, motor, right, 출력 edge 83), aBN1 = idx 94426 (`SAD093`), aDN1 = idx 24975 (`DNg62`), aDN2 = idx 88939 (`DNge078`).
