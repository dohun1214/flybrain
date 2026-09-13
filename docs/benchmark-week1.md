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

## Stage B — PyTorch CUDA LIF vs Brian2 정답 (이슈 #4) — `brain/lif.py`, `brain/compare.py`, `brain/test_equivalence.py`

### 1. 결정적 동치 테스트 (RNG 배제) — 스파이크 단위 완전 일치

포아송 대신 미리 뽑은 Bernoulli(150 Hz·dt) 이벤트를 Brian2(`SpikeGeneratorGroup`, 68.75 mV, 'synapses' 슬롯)와 우리 커널(`brain.inject`)에 똑같이 주입해 전뇌 스파이크 열을 집합으로 비교 (`.venv-brian2` + torch CPU).

| 조건 | Brian2 | PyTorch | 일치 |
|---|---|---|---|
| 1초, silencing 없음 | 13,975 스파이크 / 370 활성 / MN9 88 | 13,975 / 370 / 88 | **스파이크 (스텝, 뉴런) 전부 동일** |
| 0.5초, 720575940622695448·720575940617937543 silence | 6,538 / 374 / MN9 45, silenced 뉴런 67·69 스파이크 | 6,538 / 374 / 45, 67·69 | **전부 동일** |

이 테스트로 찾은 원본 의미론 (CLAUDE.md에 추가):
- **불응기 중(그 스텝에 스파이크한 뉴런 포함)에 도착한 시냅스 입력 `g += w`는 버려진다.** Brian2가 `(unless refractory)` 변수에 conditional write를 걸기 때문 — `on_pre='g += w'`도 not_refractory인 뉴런에만 적용된다. 이걸 빼고 구현하면 스파이크가 약 20% 많아진다(첫 구현에서 실제로 발생: 16,368 vs 13,698).
- 지연: 스파이크 스텝 k → k+18 스텝의 시냅스 슬롯에 도착(상태 갱신·임계값 뒤, 리셋 앞). 불응기 22스텝(`timestep(2.2ms)`), 스파이크 후 k+22부터 다시 갱신·발화 가능.

### 2. 통계적 비교 (GPU, 포아송 RNG는 torch, 30 trial × 1초, seed 0)

| 실험 | PyTorch 스파이크 | 활성 | MN9 Hz | Brian2 MN9 Hz (5시드 범위) | Jaccard vs 각 시드 | 발화율 상관 |
|---|---|---|---|---|---|---|
| sugarR_100Hz | 287,929 | 407 | 66.8 ± 3.0 (30/30) | 66.5 ~ 69.6 | 0.945 ~ 0.966 | 0.9994 ~ 0.9997 |
| sugarR_150Hz | 409,509 | 441 | 83.2 ± 4.0 (30/30) | 83.0 ~ 84.5 | 0.962 ~ 0.973 | 0.9997 ~ 0.9998 |
| sugarR_200Hz | 509,245 | 449 | 92.2 ± 4.3 (30/30) | 92.8 (1시드; 저자 93.3) | 0.976 | 0.9998 |
| slnc 720575940617937543 (100 Hz) | 277,585 | 419 | 63.7 ± 5.2 (30/30) | 64.8 (저자 63.3) | 0.945 | 0.9997 |
| slnc 720575940621754367 | 287,716 | 409 | 68.2 ± 4.4 (30/30) | 67.0 (저자 66.9) | 0.954 | 0.9996 |
| slnc 720575940622695448 | 302,013 | 410 | 71.2 ± 4.8 (30/30) | 70.9 (저자 71.2) | 0.978 | 0.9994 |

silenced 뉴런 자신의 발화율: 101.0 / 97.6 / 114.6 Hz (Brian2 99.4 / 98.7 / 115.0) — 출력 시냅스만 0이라는 의미론 일치.

**합격 기준(Stage A 시드 편차 기준: MN9 30/30, MN9 Hz 시드 범위 ±2, Jaccard ≥ 0.93, 상관 ≥ 0.999) 전 조건 통과 → Stage B [검증됨].** Brian2 시드끼리의 Jaccard(0.93~0.97)·상관(≥0.9995)과 같은 수준이므로, 남은 차이는 RNG 차이뿐이다.

### 3. 인터페이스·성능 (이 PC, RTX 4060 Ti)

| 항목 | 값 |
|---|---|
| `snapshot()` (v, g, last_spike, 지연 링버퍼, RNG 상태, silenced 목록) | 0.08 ~ 3.5 ms |
| `restore()` | 0.6 ~ 10.8 ms |
| restore 후 같은 시드로 재실행 → 스파이크 완전 동일 | True |
| restore 후 silence → 결과 달라짐 | True (2,764 → 2,285 스파이크 / 0.2초) |
| VRAM (v630, float64 상태 + float32 CSR) | 245 MB (단독) ~ 492 MB (비교 스크립트 최대) |
| 속도 (eager, 스텝당 Python 루프, 활성 ~400) | 생물 1초당 6.6 s (단독) ~ 15 s (CPU 경합 시) → **0.07 ~ 0.15× 실시간** |

→ 속도는 GPU 연산이 아니라 스텝당 커널 런치·Python 오버헤드(10,000 스텝/초)에 묶여 있다 (GPU 사용률 ~28%). 이슈 #6에서 CUDA graph / 스텝 융합으로 측정·개선.

## 성능 측정 (이슈 #6, 계획서 §7) — `scripts/bench_brain.py`, `bench_brain_graph.py`, `bench_flygym.py`, `bench_loop.py`

이 PC(i7-14700K, RTX 4060 Ti 8GB) 실측. 결과 json: `data/bench/` (gitignore).

### 뇌 (v630, dt 0.1 ms = 생물 1초당 10,000 스텝)

**eager 커널(`brain/lif.py`)** — 활성 뉴런 수와 무관하게 1.5~2.0 ms/step. 데이터 의존 형상(`nonzero`, `int(lens.sum())`) 때문에 스텝마다 CPU↔GPU 동기화가 생겨 GPU 사용률 ~28%.

| 조건 | 활성 뉴런 (1초) | 스파이크/스텝 | ms/step | 실시간 배율 |
|---|---|---|---|---|
| 자극 없음 (elementwise + 런치만) | 0 | 0 | 0.473 | 0.21× |
| 당 GRN 150 Hz | 317 | 1.3 | 1.55 | 0.064× |
| + 무작위 1k 뉴런 50 Hz | 9,284 | 34 | 2.07 | 0.048× |
| + 10k | 20,410 | 104 | 1.97 | 0.051× |
| + 50k | 59,221 | 317 | 2.03 | 0.049× |
| elementwise 만 CUDA graph 캡처 (하한) | — | — | **0.059** | 1.70× |

**고정 형상 + CUDA graph 커널(`brain/lif_graph.py`, 신규)** — 스파이크 압축·시냅스 전개를 고정 예산(S_MAX 2048 스파이크/스텝, E_MAX 131,072 시냅스/스텝)으로 바꾸고 100 스텝(10 ms)을 그래프 하나로 replay. 예산 초과는 overflow 플래그로 감지.

| 조건 | 활성 뉴런 | 스파이크/스텝 | ms/step | 실시간 배율 |
|---|---|---|---|---|
| 당 GRN 150 Hz | 374 | 1.4 | 0.187 | **0.53×** |
| + 1k 뉴런 50 Hz | 9,847 | 50 | 0.201 | 0.50× |
| + 10k | 21,941 | 110 | 0.200 | 0.50× |
| + 50k | 61,333 | 320 | 0.210 | 0.48× |

- 검증: (1) Brian2 결정적 동치 테스트 `--graph` 통과(0.5초 silencing 조건 6,538 스파이크 전부 동일) (2) 같은 시드에서 eager 커널과 스파이크 열 완전 동일 (13,543 / MN9 78), 30 trial 150 Hz 통계도 eager 와 동일(409,509 스파이크) → Brian2 정답 대비 Jaccard 0.962~0.973, 상관 ≥ 0.9997 (3) 같은 시드 2회 동일, 다른 시드 다름, snapshot 0.26 ms / restore 0.68 ms, restore 후 재실행 동일
- 프로파일(torch.profiler, 200 스텝): 첫 버전은 무효 시냅스 칸(가중치 0)을 전부 주소 0에 `index_add_` 해서 원자 연산 경합으로 그 커널 하나가 0.58 ms/step → 무효 칸을 서로 다른 주소(e mod N)로 흩어서 0.006 ms. 남은 0.25 ms/step 은 약 60개의 작은 커널(각 2~10 µs)의 합 → 다음 단계는 torch.compile/Triton 으로 elementwise 융합 (2주차 후보)
- eager 대비 8배(6.6 s → 1.8 s / 생물 1초). 30 trial 비교 450 s → 56 s.

**VRAM**: v630 238 MB(할당)/242 MB(피크), v783 247/252 MB (float64 상태 + float32 CSR + 18스텝 링버퍼). 비교·벤치 스크립트 피크 480~654 MB. 8 GB 중 여유 충분.

### FlyGym 2.1 (CPU, 물리 0.1 ms, 1초 시뮬)

| 지형 | 겹눈(10 ms 주기) | 렌더(480×640 @30fps) | 전체 실시간 배율 | 물리만 | µs/step: 관측 / 컨트롤러 / 액션 / 물리 / 렌더 / 겹눈 |
|---|---|---|---|---|---|
| 평지 | ✗ | ✗ | 0.117× (첫 실행, 워밍업 포함) | 1.02× | 193 / 553 / 11 / 98 / – / – |
| 평지 | ✗ | ✓ | **0.199×** | 1.41× | 111 / 308 / 6 / 71 / 7 / – (2.3 ms/프레임) |
| 평지 | ✓ | ✗ | 0.178× | 1.44× | 111 / 310 / 6 / 70 / – / 65 (6.5 ms/호출) |
| 평지 | ✓ | ✓ | 0.184× | 1.44× | 110 / 307 / 6 / 70 / 7 / 43 (2.1 ms/프레임, 4.3 ms/호출) |
| 블록 험지 | ✗ | ✗ | 0.137× | 0.39× | 137 / 332 / 6 / 254 / – / – |
| 블록 험지 | ✓ | ✓ | 0.130× | 0.39× | 128 / 308 / 6 / 257 / 10 / 58 |

- **병목은 물리가 아니라 `HybridTurningController.step`(약 310 µs/step, Python CPG+규칙 컨트롤러)과 `HybridControllerObservation.from_sim`(110 µs)** — 물리(70 µs)의 6배. 평지 물리만은 1.4× 실시간.
- 렌더 2.1~3.3 ms/프레임 → 30 fps 기준 스텝당 7~10 µs, 부담 없음. 겹눈 4~6.5 ms/호출 → 10 ms 주기면 스텝당 43~65 µs.
- 험지는 물리가 3.6배 느려짐(접촉 증가) → 0.39×.

### 결합 폐루프 (뇌 GPU 10 ms 청크 + 임시 디코더 + FlyGym 100 스텝, 순차 실행, 2초)

| 구성 | 실시간 배율 | 10 ms 청크당 ms: 뇌 / 디코더 / 몸 / 겹눈 |
|---|---|---|
| 평지, 렌더·겹눈 없음 | **0.095×** | 30.3 / 0.4 / 74.2 / – |
| 평지 + 겹눈 + 렌더 | 0.098× | 32.1 / 0.4 / 66.5 / 3.3 |

- 뇌 청크 30 ms(단독 실행 18.7 ms 보다 큼 — `run()` 호출당 기록 버퍼 복사·CPU 전송 오버헤드), 몸 66~74 ms. 순차 합산이라 두 쪽을 병렬(뇌는 GPU 비동기, 몸은 CPU)로 돌리면 max(30, 70) ≈ 0.13~0.15×.
- **판단 (계획서 §7 기준)**: 뇌 단독 0.5× 실시간 → 이벤트 구동 커널로 이미 전환한 상태이며 가지치기 불필요. 몸이 더 느리므로(0.15~0.2×) 2주차에 (a) 컨트롤러·관측을 numba/벡터화하거나 호출 주기를 낮추고 (b) 뇌·몸을 비동기로 겹치고 (c) 그래도 부족하면 계획대로 화면을 뇌와 같은 배율로 느리게 재생한다. 현재 예상 데모 속도는 실시간의 1/7 ~ 1/10.
- CPU: 몸·뇌 루프 모두 단일 스레드 Python 바운드. 뇌 비교 실행 중 `Win32_Processor.LoadPercentage` 는 2% (28 스레드 기준) — 나머지 코어는 유휴라 스트리밍 인코딩 등과 동시 실행 가능. 정밀한 코어별 사용률은 아직 안 잼.

## 하강뉴런 채널 초안 (이슈 #7) — `brain/build_channels.py` → `brain/channels.json`

주석 TSV cell_type(v783 root_id)으로 집단을 뽑고 v630 인덱스도 함께 기록. `status`·`evidence` 필드에 문헌 근거와 라벨을 넣었다. **전부 디코더 입력 후보이며, LIF 모델에서 실제로 그 행동을 내는지는 2~3주차 [검증 과제].**

| 채널 | 정의 | side | v783 | v630 | v630 누락 | 상태 |
|---|---|---|---|---|---|---|
| forward | DNp09 | both | 2 | 2 | 0 | 근거 있음/모델 미검증 |
| turn_left | DNa01, DNa02 | left | 2 | **1** | 1 (DNa01 left `720575940627787609` 가 v630 주석 매칭에 없음) | 근거 있음/모델 미검증 |
| turn_right | DNa01, DNa02 | right | 2 | 2 | 0 | 근거 있음/모델 미검증 |
| escape | DNp01 (Giant Fiber) | both | 2 | 2 | 0 | 근거 있음/모델 미검증 |
| backward | MDN | both | 4 | 4 | 0 | 근거 있음/모델 미검증 |
| looming_LC4 | LC4 | both | 104 | 92 | 12 | 근거 있음/모델 미검증 |
| looming_LPLC2 | LPLC2 | both | 210 | 176 | 34 | 근거 있음/모델 미검증 |
| proboscis_MN9 / groom_aBN1 / groom_aDN1 / groom_aDN2 | 논문 ID | — | 1 each | 1 each | 0 | 검증됨 (논문) |
| input_sugar_GRN_R / L, bitter, water, JON_CE, JON_F | 논문 ID | — | 20/9/20/18/69/60 | 21/10/21/18/70/60 | 1/1/1/0/1/0 | 검증됨 (논문 ID) |

- **정지 DN [검증 과제]**: Sapkal et al. 2024 (Nature, "Neural circuit mechanisms underlying context-specific halting") 의 **Foxglove·Bluebell** — SEZ 에서 내려가는 GABA성 하강뉴런으로 walking-promotion DN(DNp09 포함)을 억제한다. FlyWire cell_type 명칭은 논문 보충자료에서 확인해야 함(주석 TSV synonyms 에 없음) → 2주차. 같은 논문의 Brake 는 VNC 상행뉴런이라 뇌 모델 밖.
- **DNp09 주의**: Bidaye et al. 2020 은 전진 보행 DN 으로, Zacarias et al. 2018 은 freezing(정지) DN 으로 보고(silencing 시 freezing 소실, running 유지). 자극 세기·문맥 의존. 디코더에서 "전진"으로 읽되 정지 채널과 충돌 가능성을 기록해 둔다.
- turn_left 는 v630 에서 DNa01 이 빠져 DNa02 하나뿐 → v630 단계 STEERING 실험은 DNa02 L/R 로만, 좌우 대칭 비교는 v783(Stage C) 이후에.
- LC4/LPLC2 는 v630 에서 각각 12·34개 누락(주석 매칭 83% 영향) → Level C looming 실험은 v783 에서.

Sources: [Zacarias et al. 2018, Nat Commun](https://www.nature.com/articles/s41467-018-05875-1) · [Bidaye et al. 2020 (bioRxiv 판)](https://www.biorxiv.org/content/10.1101/798439v1.full) · [Sapkal et al. 2024, Nature](https://www.nature.com/articles/s41586-024-07854-7)
