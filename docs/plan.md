# 초파리 커넥톰 생태 시뮬레이션 — 실행 계획 v5.1

작성 2026-09-13 (v5 + 최종 검증 반영) · 경진대회 출품작(11월) · 개인 프로젝트 · 주 20h+

> 표기 규칙 — 모든 기술 항목은 다음 중 하나로 표시한다.
> **[검증됨]** 원논문·공식 문서·소스코드로 직접 확인 · **[근거 있음/모델 미검증]** 생물학 문헌 근거는 있으나 Shiu LIF 모델 또는 FlyGym에서 재현 미확인 · **[직접 설계]** 프로젝트가 만들어야 하는 것 · **[검증 과제]** 아직 확인 안 된 가정, 해당 주차에 확인

## 0. 한 줄 정의

**실제 초파리 커넥톰 기반 전뇌 LIF 시뮬레이션(Shiu et al. 2024 모델)을 자택 GPU에서 돌리고, 그 뇌의 하강뉴런 활동을 NeuroMechFly v2(FlyGym 2.x) 생체역학 몸의 명령으로 변환해, 3D 물리 세계에서 걷고·먹고·도망치고·그루밍하는 파리 한 마리를 — 파리가 느끼는 감각, 뇌 속 활동, 그리고 뇌의 일부를 껐을 때의 변화까지 — 한 화면에서 구경하고 실험할 수 있는 "embodied connectome simulator".**

연구가 아니라 "살아있는 것 같은 파리"가 목표. 단, 그것이 대본이 아니라 커넥톰에서 나온다는 것을 관객이 직접 확인할 수 있어야 한다.

### 핵심 축 4개

| 축 | 역할 | 관객이 보는 것 |
|---|---|---|
| **Digital Aquarium** | 몸이 있는 파리가 사는 3D 세계 | 파리가 걷고 먹고 도망치고 그루밍 |
| **Fly POV** | 파리에게 지금 들어오는 감각 입력 | 겹눈 시야, looming 점수, 좌우 냄새, 접촉 |
| **Brain Inspector** | 뇌 내부 활동과 행동 결정 | sensory → central → descending → behavior, Neural Event Replay |
| **Brain Surgery** | 뉴런 집단을 끄고 같은 자극을 다시 줌 | NORMAL vs SILENCED 나란히 비교 |

내부 검증: 행동 벤치마크 표(§6). 한 화면의 흐름: **환경 → Fly POV → 감각 인코더 → 뇌 → 하강 출력 → 행동**.

## 1. v4 → v5 변경 요약

| 항목 | v4 | v5 | 근거 |
|---|---|---|---|
| FlyGym 배포 | "PyPI 미배포, GitHub 설치" | **PyPI 배포됨** — `pip install flygym[examples]`. 2.0.0 2026-04-01, **2.1.0 2026-06-24**. Python ≥3.12,<3.15, MuJoCo ≥3.9,<3.10 | PyPI JSON·공식 설치 문서 [검증됨]. v4 오류 원인: 조사 환경의 pip이 Python 3.11이라 2.x가 목록에서 숨겨짐 |
| 핵심 축 | 수족관 + 인스펙터 (2) | **수족관 + Fly POV + 인스펙터 + Brain Surgery (4)** | §0 |
| Fly POV | 없음 | 신규. 감각 입력 디버거로 정의 | §5.2 |
| Brain Surgery | 6주차 "채택 여부 결정" | **6주차 구현 기본 목표(P0.5)**. silencing API·상태 스냅샷은 1주차 엔진 설계에 포함 | §5.4, §8 |
| 우선순위 | 암묵적 | P0 / P0.5 / P1 / P2 명시 | §9 |
| 시연 시나리오 | 한 줄 | 7단계 | §11 |
| 후각 | [검증 과제] | 2.x에 없음 확정 → 직접 계산 | 소스 확인 [검증됨] |

## 2. 확정 결정

| 결정 | 선택 | 상태 |
|---|---|---|
| 뇌 모델 | Shiu et al. 2024 LIF. V_rest = V_reset −52mV, V_th −45mV, τ_m 20ms, τ_syn 5ms(알파 시냅스, `dg/dt=-g/τ`), 불응기 2.2ms, 지연 1.8ms, 0.275mV/시냅스, 포아송 150Hz × 250배. **커널 구현 시 놓치기 쉬운 것 [검증됨 `model.py`]: 스파이크 시 `g`도 0으로 리셋 · 불응기 동안 v·g 모두 동결(`unless refractory`) · 포아송 대상 뉴런은 불응기 0 · 실행 1초 × 30회 반복** | [검증됨] |
| 뇌 엔진 | 직접 구현: PyTorch CUDA 스파스 LIF. 원본 Brian2(MIT)를 정답으로 비교. **1주차부터 포함할 인터페이스: `silence(neuron_ids)`, `snapshot()/restore()`, 시드 고정.** silencing은 원본 코드와 동일하게 **출력 시냅스만 0**(`model.py: syn.w['i == n'] = 0`; README의 "to and from"과 달리 코드는 outgoing만) | [직접 설계]; silencing 방식 [검증됨] |
| 커넥톰 | Stage A/B **v630**(원논문, 리포 동봉), Stage C **v783** | [검증됨] |
| 세포 타입 주석 | `flyconnectome/flywire_annotations` TSV(139,248행, v783 root_id 기준). **v783 뉴런의 99.99%(138,625/138,639) 매칭, v630은 83%(106,214/127,400)만 매칭** → 주석 기반 패널·검색은 v783(Stage C 이후)에서 완전해짐. **논문 핵심 뉴런의 v630 root_id는 리포 `figures.ipynb`에 전부 있음**(당 GRN 21, 쓴맛 21, 물 18, JON-CE 70, JON-F 60, MN9, aBN1, aDN1, aDN2) | [검증됨] |
| 몸·세계 | **FlyGym 2.1.0**(Apache-2.0, PyPI). `pip install flygym[examples]` (`flygym_demo` 패키지는 기본 설치에 포함, `examples`는 노트북용 pandas·tqdm 추가). 첫 실행 시 대형 메시를 S3에서 자동 다운로드(네트워크 필요) | [검증됨] |
| 보행 | `flygym_demo.complex_terrain.HybridTurningController.step(descending_signal: shape (2,), obs)` — 절댓값이 좌/우 CPG 진폭, **부호가 CPG 주파수 방향(음수 = 역방향 = 후진 후보)** | [검증됨] 코드; 음수 신호가 안정적 후진 보행을 내는지 [검증 과제] |
| 겹눈 | `Simulation.get_raw_vision` / `get_ommatidia_readouts` | [검증됨] 코드 존재; 2.x 튜토리얼 없음 → 1주차 동작 확인 [검증 과제] |
| 후각 | **2.x에 없음** — 냄새원↔더듬이·구기 사이트 거리로 농도 직접 계산 | [검증됨] 없음; 계산은 [직접 설계] |
| 접촉 | `get_ground_contact_info`, `get_bodysegment_contact_forces` | [검증됨] |
| 몸의 관절 | 2.x MJCF 관절 87개: 다리 6×7 + 머리(pitch/roll/yaw) + 더듬이(pedicel·funiculus·arista 각 3축 × 좌우). **주둥이(rostrum/haustellum)·날개·복부에는 관절 없음**(바디만 존재) | [검증됨] |
| 서버 | 자택 4060 Ti **8GB**, Python 단일 프로세스. 뇌만 GPU, FlyGym은 CPU(Warp 제외) | — |
| 뷰어 | 5주차: 서버 렌더(MuJoCo) 영상 스트리밍 — 바닥 텍스처·스카이박스·그림자·재질·카메라 워크 튜닝 포함. **7주차: Godot 렌더러(P1)** — 서버가 관절 각도·위치만 송출, Godot이 NeuroMechFly 메시(Apache-2)를 PBR 재질·실시간 조명·꾸민 세계로 그림 + 카메라 조작 + Surgery/상호작용 UI. flygym WASM 뷰어는 대안 | 영상 [직접 설계, 저위험]; Godot 메시 임포트·관절 매핑(87개) [검증 과제] |
| 파리 수 | 1마리 | — |
| LLM | P2. 엔진의 구조화 이벤트를 문장으로만. 인과 추론 금지. API | — |
| 쓰지 않는 것 | fly-escape(라이선스 없음), webgpu-fly(가중치 55배 축소), snedea/flybrain(DN 그룹 비어 있음), fly-brain-bench 엔진(가지치기·브라우저). eonsystems/fly-brain은 GPL-2라 읽기만 | — |

## 3. 뇌 검증 계획 (1~2주차)

### 3.1 3단계

**Stage A — Brian2 + v630 (원본 그대로)**
- 1순위: **당 GRN 자극 → MN9 발화** [검증됨: 논문 핵심 실험]
- 2순위: **JON → aBN1 → aDN1/aDN2 더듬이 그루밍** [검증됨: 논문]. root_id 확보됨 [검증됨]: MN9 `720575940660219265`(주석 cell_type `CB0701`, super_class motor), aBN1 `720575940630907434`(`SAD093`), aDN1 `720575940616185531`(`DNg62`), aDN2 `720575940629806974`(`DNge078`). **주의: 논문 명칭(aDN1/aBN1/GF)은 FlyWire cell_type에 없다 — ID로 찾는다.** 당·물 GRN은 둘 다 cell_type `LB3`라 타입으로 구분 불가 → 반드시 논문 ID 목록 사용
- 셔플 대조군(논문: 정상 100% vs 셔플 1/100 활성)은 선택 백로그(§10). **리포에 셔플 코드 없음 [검증됨]** → 하려면 직접 구현

**Stage B — PyTorch CUDA + v630 (동일 데이터·프로토콜)**
- 지표: 활성 뉴런 집합 Jaccard, 뉴런별 발화율 상관, 총 스파이크 비율, MN9 발화율, 집단 활동 시계열, 자극–반응 곡선
- 합격 기준(초안): MN9 활성 일치 100%, Jaccard ≥ 0.9, 상관 ≥ 0.95 — Brian2 시드 간 편차를 먼저 재서 조정
- **silencing 검증**: 원본의 silencing(시냅스 0)과 우리 `silence()`가 같은 결과를 내는지 이 단계에서 함께 확인 — Brain Surgery의 정답 기준

**Stage C — v783 이전**
논문 ID 목록의 v783 생존률 [검증됨]: 당 20/21, 쓴맛 20/21, 물 18/18, JON-CE 69/70, JON-F 60/60, MN9·aBN1·aDN1·aDN2 전부 생존. 사라진 소수는 같은 cell_type·side로 대체 후보를 찾고 기록. 벤치마크 반복 후 v630과 질적 일치 확인 [검증 과제]

### 3.2 데이터 숫자 표기 규칙

| 항목 | v630 | v783 |
|---|---|---|
| 뉴런 수 | **127,400** | **138,639**(Shiu 리포 `Completeness_783.csv`; Codex 공개치 139,255와 다름 — 우리는 리포 값을 씀) |
| 시냅스 합(`Connectivity` 열 합) | **52,793,639** | **54,492,922** |
| 뉴런쌍 edge 수(임계값 없음) | **14,687,178** | **15,091,983** |
| ≥5 시냅스 연결 수 | **2,614,028** | **2,700,513** |
| **시뮬 사용 edge 수·임계값** | 실행 시 기록 | 실행 시 기록 |

위 수치는 2026-09-13 리포 파일에서 직접 계산 [검증됨]. "15.1M connections" 단독 표기 금지. 프로그램 시작 시 재계산해 메타데이터에 기록. 부호는 `Excitatory` 열(+1/−1), 가중치는 `Excitatory x Connectivity` × 0.275mV.

## 4. 아키텍처

```
┌─ 서버 (자택 4060 Ti 8GB, Python 단일 프로세스) ────────────────────────┐
│                                                                        │
│  [FlyGym 2.1 / MuJoCo, CPU]                                             │
│   NeuroMechFly 몸 · 지형 · 먹이(냄새원) · 그림자 · 먼지                  │
│   관측: 겹눈 이미지[검증됨] · 접촉[검증됨] · 사이트 위치[검증됨]         │
│            ↓                                                            │
│  [감각 인코더] [직접 설계]                ──→  [Fly POV 데이터]          │
│   겹눈 → looming 점수·팽창률 → LC4/LPLC2 자극률      겹눈 시야(좌/우)   │
│   사이트 위치 → 좌/우 냄새 농도 → ORN 자극률         looming 점수       │
│   접촉 → 당 GRN / JON·강모 자극률                    좌/우 냄새, 접촉    │
│            ↓                                                            │
│  [뇌] PyTorch CUDA LIF, dt 0.1ms                                        │
│   silence 마스크 · snapshot/restore · 시드 고정  ←── [Brain Surgery]     │
│            ↓                                                            │
│  [운동 디코더] [직접 설계]  하강뉴런 집단 Hz(50ms 창) → 정규화 명령       │
│            ↓                                                            │
│  [행동 중재기] [직접 설계]                                               │
│   forward/turn → HybridTurningController 2값 신호                        │
│   backward → 음수 신호(같은 컨트롤러) · escape/proboscis/groom → 직접 구현  │
│            ↓                                                            │
│  [Brain Inspector 데이터] 활성 수 · super_class 흐름 · 채널 · 행동 ·      │
│                          Event Replay 링버퍼 · silencing 상태            │
│            ↓                                                            │
│  [화면 합성] 3D + Fly POV + Inspector (+ Surgery 비교 뷰) → 30fps 송출    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
     뷰어 N개: 영상 (5주차) → Godot 렌더러 + 상호작용·Surgery UI (7주차, P1)
```

**원칙: "뭘 할지"는 뇌가, "어떻게 움직일지"는 컨트롤러가.** DN 발화율 → FlyGym 신호 변환은 우리 디코더이지 FlyGym의 생물학적 매핑이 아니다. Fly POV는 인코더의 *입력*을, Brain Inspector는 뇌의 *출력*을 보여주므로 둘 사이에 인코더가 무엇을 했는지가 관객에게 드러난다.

## 5. 화면과 기능

### 5.1 레이아웃 (서버 합성, 16:9 기준)

```
┌────────────────────────────────┬──────────────────────┐
│                                │  FLY POV             │
│                                │  겹눈 좌 / 우        │
│          3D WORLD              │  looming ████░ 0.82  │
│                                ├──────────────────────┤
│                                │  ODOR  L ███ R █████ │
│                                │  CONTACT ant·prob·leg│
├────────────────────────────────┴──────────────────────┤
│  BRAIN INSPECTOR                                       │
│  sensory ▶ central ▶ descending ▶ [행동: ESCAPE]       │
│  채널 바 · 활성 뉴런 수 · Event Replay · 수술 상태      │
└────────────────────────────────────────────────────────┘
```
Surgery 비교 시에는 3D 영역을 좌우 분할(NORMAL | SILENCED)하고 Inspector도 두 줄로.

3D 퀄리티 단계: 1~5주차는 MuJoCo 렌더 + 튜닝(텍스처·조명·그림자·카메라). 7주차에 Godot 렌더러로 교체 — 관절 상태만 받아 그리므로 물리·뇌는 그대로. 5주차까지 뇌 연결이 안 되면 렌더러는 자동으로 밀린다.

### 5.2 Fly POV — 감각 입력 디버거

| 요소 | 내용 | 출처 | 상태 |
|---|---|---|---|
| 겹눈 시야 | `get_ommatidia_readouts` 낱눈 값을 육각 격자로 좌/우 표시 + 사람용 확대본 | FlyGym | [검증됨] API 존재, 시각화 [직접 설계] |
| 시각 특징 | looming 점수(0~1 막대), 팽창률(%/s) — LC4/LPLC2 자극으로 바뀌기 *직전* 값 | 인코더 | [직접 설계] |
| 냄새 | 좌/우 더듬이 농도 막대, 채널별 확장 가능 | 인코더(직접 계산) | [직접 설계] |
| 접촉 | 더듬이 L/R · 주둥이 · 앞다리 ON/OFF — 현재 행동과 관련된 것만 | FlyGym 접촉 | [검증됨] 데이터, 표시 [직접 설계] |

정의: 장식이 아니라 **"3D 물체 → 파리 망막 → looming 특징 → LC4/LPLC2 자극"** 과정을 눈으로 확인하는 도구. 3~4주차 디버깅에도 우리가 먼저 쓴다.

### 5.3 Brain Inspector

| 요소 | 내용 | 상태 |
|---|---|---|
| 활성 뉴런 수 / 집단 발화율 | 50ms 창, 시계열 | [직접 설계] |
| 클래스/영역 활동 | super_class·cell_class별 막대 + 좌표 점구름 | 주석 [검증됨], 표시 [직접 설계] |
| sensory → central → descending 흐름 | 3단 파이프, 폭 = 발화율 | [직접 설계] |
| 선택 뉴런 활동 | 고른 세포 타입 발화 궤적 | [직접 설계] |
| 디코더 출력 / 현재 행동 | 채널 바 + 중재기 상태 | [직접 설계] |
| **Neural Event Replay** | 행동 전환 직전 300~500ms 링버퍼를 타임라인으로. **개입 상태도 기록**(예: "−90ms DNp01 SILENCED / 0ms 도망 전환 없음"). 활동 순서 기록이며 인과 증명 아님 — "원인"이라 쓰지 않음 | [직접 설계] |
| 수술 상태 | silenced 집단은 발화율 대신 `SILENCED` 표시 | [직접 설계] |

### 5.4 Brain Surgery / Intervention Mode

목적: 같은 감각 입력에 대해 뉴런 집단을 껐을 때 네트워크 출력과 시뮬레이션 행동이 어떻게 변하는지 **관객이 직접 실험**하게 한다. 정확한 표현: *"커넥톰 기반 시뮬레이션에서 특정 뉴런/집단을 silencing했을 때 네트워크 출력과 시뮬레이션 행동이 어떻게 변하는지 관찰한다."* 실제 생물 인과성 주장은 기존 문헌이 있을 때만.

**기능 (첫 버전은 두 개만 확실히)**
- **Silencing** — 선택 집단의 입·출력 시냅스 0 (Shiu 방식과 동일 [검증됨]). Stage B에서 원본과 결과 비교로 검증
- **동일 조건 비교** — `snapshot()`으로 뇌 상태 + MuJoCo 상태 + 시드 저장 → 자극 실행(NORMAL) → `restore()` → silence → 같은 자극 궤적·타이밍 재실행(SILENCED) → 좌우 분할 표시. 포아송 때문에 완전 동일이 불가하면 시드 고정 + 반복 N회 비교
- Stimulation(직접 자극)은 P2

**UI (첫 버전: 검색 + 프리셋, 3D 선택 없음)**
```
BRAIN SURGERY
Search: [ DNp01 ]   Selected: DNp01 (Giant Fiber)   Status: NORMAL
[ SILENCE ] [ RESTORE ] [ RERUN SAME STIMULUS ]
```
프리셋 후보(각각 **ID·주석 확인 후에만 UI에 노출**, 미확인은 숨기거나 `Experimental` 표시):

| 프리셋 | 집단 | 노출 조건 |
|---|---|---|
| NORMAL FLY | — | 항상 |
| ESCAPE LESION | DNp01 | Level C looming이 3주차에 성립했을 때 |
| STEERING LESION | DNa01/DNa02 | 2주차 조향이 성립했을 때 |
| FORWARD LESION | DNp09 | 동일 |
| BACKWARD LESION | MDN | 후진이 성립했을 때 |
| FEEDING LESION | MN9 상류 개재뉴런(MN9 자체를 끄는 건 자명해서 덜 흥미로움; 집단 선정 [검증 과제]) | Level A 재현 후 |
| GROOMING LESION | aBN/aDN | Level A 2순위 재현 후 |

Inspector 연동: silencing 즉시 `SILENCED` 표시, Event Replay에 개입 기록.

### 5.5 상호작용 (관객 → 세계)
먹이 놓기 · 그림자 던지기 · 먼지 뿌리기 — 영상 옆 웹 버튼(7주차, 4주차 여유 시 5주차). 여러 명 동시 입력은 큐 + 쿨다운.

## 6. 행동 벤치마크 표

### Level A — Shiu 논문 재현 (뇌만)

| 회로 | 생물학 | LIF 모델 | 비고 |
|---|---|---|---|
| 당 GRN → MN9 | 확립 | **[검증됨]** | Stage A/B/C 1순위 |
| JON → aBN1/2 → aDN1/2 | 확립 | **[검증됨]** | 뉴런 ID [검증 과제] |
| 쓴맛 GRN → 주둥이 억제 | 확립 | [검증됨] | P1 이후 |
| 물 GRN → 주둥이 | 확립 | [검증됨] | P1 이후 |

### Level B — 확립된 DN 기능 + 우리가 만든 몸 연결

| 행동 | 읽는 뉴런 | LIF 모델 | FlyGym 쪽 |
|---|---|---|---|
| 전진 | DNp09 | [근거 있음/모델 미검증] | 디코더 → HybridTurningController [직접 설계] |
| 회전 | DNa01, DNa02 L/R | [근거 있음/모델 미검증] | 동일 |
| 멈춤 | 정지 DN(타입 [검증 과제]) | [근거 있음/모델 미검증] | 신호 0 |
| 후진 | MDN | [근거 있음/모델 미검증] | HybridTurningController에 **음수 신호** → CPG 역방향 [검증됨 코드]; 실제 보행 안정성 [검증 과제] |
| 도망 점프 | DNp01(FlyWire 명칭; `GF` 아님) | [근거 있음/모델 미검증] | **날개 관절 없음 → "이륙" 애니메이션 불가.** 점프 = 뒷다리 신전 시퀀스(관절 있음) [직접 설계] |
| 주둥이(몸) | MN9 경로 | [검증됨](Level A) | **2.x에 주둥이 관절 없음 [검증됨]** → v1: 정지 + 머리 pitch 숙임 + Fly POV/Inspector에 `PROBOSCIS EXT` 표시. 선택: MjSpec으로 c_rostrum에 힌지 관절 추가(FlyGym이 MjSpec 기반이라 가능성 있음) [검증 과제, P1] |
| 더듬이 그루밍(몸) | aDN 경로 | [검증됨](Level A) | 앞다리 시퀀스 + 더듬이 관절(pedicel 3축, 있음) 눌림 [직접 설계] |

### Level C — 프로젝트 고유 폐루프 확장 (전부 [직접 설계], 성공 보장 없음)

| 확장 | 확인 방법 | Fly POV 표시 |
|---|---|---|
| looming 인코딩 → LC4/LPLC2 → DNp01 → 점프 | 그림자 접근 시 DNp01 발화 유의 증가 (3주차) | looming 점수·팽창률 |
| 후각 정위 → ORN → DNa02 비대칭 → 회전 | 먹이 도달률 > 무작위 보행 (4주차) | 좌/우 냄새 |
| 먹이 접촉 → GRN → MN9 → 주둥이 | — | 접촉 |
| 다중 행동 중재 | 상태 전환 로그 | 현재 행동 |

Level C 실패 시 해당 감각은 "뇌를 거치지 않는 직접 규칙"으로 대체하고 Fly POV/Inspector에 그렇게 표시한다(속이지 않음). Brain Surgery 프리셋은 성립한 것만 노출.

## 7. 성능 측정 (1주차 필수, 타 시스템 수치 사용 금지)

```
brain: 생물 초 / 실제 초 (활성 뉴런 1k, 10k, 50k)  · VRAM
FlyGym(CPU): 시뮬 초 / 실제 초 (평지·험지, 겹눈 렌더 on/off) · 물리 스텝 시간 · 렌더 시간
snapshot/restore 소요 시간 (Surgery 비교의 체감 대기)
CPU 사용률 · 결합 폐루프 처리량 (뇌 + FlyGym + 인코더 + 합성)
```
참고치(우리 수치 아님): FlyGym 2.x README — CPU 약 2배속, Warp 약 60배속. 뇌 유사 구현 — M2에서 실시간 0.67배.

## 8. 일정 (9주, 9/14 → 11/15)

| 주 | 목표 | 완료 기준 | 검증 과제 |
|---|---|---|---|
| **1** | **Stage A + B, FlyGym 가동, 성능 측정** | (a) Brian2 v630 당 GRN→MN9 재현 (b) PyTorch v630이 §3.1 기준 일치, **`silence()` 결과가 원본 silencing과 일치, `snapshot/restore` 동작** (c) `pip install flygym[examples]`, 튜토리얼 4d 실행(`[1.2,0.4]` 회전) (d) `get_ommatidia_readouts`로 겹눈 이미지 저장, 신호 `[-1.0,-1.0]`으로 후진 시험 (e) §7 측정 | Python 3.12 환경, 음수 신호 후진 안정성, S3 메시 다운로드 |
| 2 | Stage C + 뇌→몸 연결 | v783 벤치마크 반복. DNa02 L/R·DNp09 Hz → HybridTurningController. 랜덤 자극만으로 돌아다님. 시간 동기화 | v630↔v783 재탐색, 정지 DN |
| 3 | 도망 + 그루밍 + **Fly POV v0** | Level A 그루밍 재현(ID 확보됨). 겹눈 → looming 점수 → LC4/LPLC2 → DNp01 유의 증가 → 뒷다리 점프. JON → 그루밍. **Fly POV에 겹눈·looming 점수 표시(디버깅용)** | LC4/LPLC2 v783 ID(주석 cell_type 있음 [검증됨]) |
| 4 | 후각 + 먹기 + **Fly POV 완성 + Inspector v1** | 냄새 농도 직접 계산 → ORN → 정위. 접촉 → GRN → 주둥이. Fly POV 4요소, Inspector 흐름·채널·행동 | — |
| **5** | **공개: Aquarium + Fly POV + Inspector** | 영상 스트림 공개 URL, 24시간 가동, 설문 10문항. 여유 시 웹 버튼 3개 + intervention 백엔드(API만) | 스트리밍 방식 |
| **6** | **Brain Surgery 구현 + 차별화 결정** | `집단 선택 → silence → 같은 자극 재실행 → NORMAL/SILENCED 분할 화면` 완성. 검증된 프리셋만 노출. **MaleCNS(구글·Janelia 수컷 데이터) 추가 여부를 이 주에 판단** — 기준은 §10 | FEEDING 프리셋 집단 |
| 7 | **Godot 렌더러 + Surgery UI + 상호작용** | NeuroMechFly 메시 임포트·관절 매핑, PBR 재질·조명·세계 꾸미기, 카메라 조작. 관절 상태 WebSocket 수신. Godot 안에 검색·프리셋·RERUN, 먹이·그림자·먼지 버튼. Event Replay 완성. (P2 여유 시 stimulation) | 메시 임포트 경로, 관절 축 일치 |
| 8 | 발표 준비 | 대표 개입 실험 결과 정리, 설문 10건, 시연 리허설, 실패 케이스 | — |
| 9 | 버퍼 | — | — |

**1~2주차가 성패를 가른다.** Stage B 불일치 → 커널 버그. 속도 부족 → 이벤트 구동 커널 → 최후 수단 임계값 가지치기(§3.2에 기록). 2주차 말까지 안 되면 재조정. **Surgery는 1~5주차에 구현하지 않되, 엔진 인터페이스(silence/snapshot/seed)만 1주차에 넣는다** — 나중에 붙이면 비싸고 지금 넣으면 싸다.

## 9. 우선순위

| 등급 | 항목 | 규칙 |
|---|---|---|
| **P0** | Shiu 모델 재현 · PyTorch 뇌 · FlyGym 몸 · 뇌→몸 · 기본 보행 · 폐루프 행동 최소 1개 · Brain Inspector | 이게 없으면 프로젝트 없음 |
| **P0.5** | Fly POV · Brain Surgery silencing · NORMAL vs SILENCED 비교 | 차별화. P0 다음 |
| **P1** | 먹기 · 그루밍 · 도망 · 후각 정위 · Neural Event Replay · 상호작용 · **Godot 렌더러(7주차)** | 행동 다양성 + 화면 퀄리티 |
| **P2** | stimulation UI · WASM 뷰어 · LLM 해설 · 여러 마리 · MaleCNS | P0/P0.5를 희생해 P2를 하지 않는다 |

## 10. 선택 백로그
- 셔플 대조군 재현(Stage A 위에서 100회, 발표 한 장)
- Level A 3·4순위(쓴맛·물)
- **MaleCNS 두 번째 뇌 추가** — 6주차에 판단. 조건: (1) 1~5주차가 일정대로 끝났고 (2) Brain Surgery가 6주차 안에 마무리될 전망일 때만. 방식: 엔진은 그대로, 데이터 파일 + 뉴런 ID 목록만 교체(Xenova MIT 엔진의 MaleCNS 로더 참고). 용도: 수컷 행동(구애·공격) 실험, "구글 데이터도 돌아간다"는 발표 대응. 주의: MaleCNS에는 검증된 LIF 벤치마크가 없으므로 [근거 있음/모델 미검증]으로 표시하고 FlyWire 결과와 같은 급으로 말하지 않는다
- FlyGym Warp(8GB라 제외)
- 발표용 Blender 오프라인 렌더 영상 — 하지 않기로 결정(2026-09-13)

## 11. 최종 시연 시나리오 (약 5분)

1. **Digital Aquarium** — 파리가 자유롭게 돌아다닌다. "이 뇌는 13만 뉴런짜리 실제 초파리 커넥톰입니다."
2. **Fly POV — 먹이** — 먹이를 오른쪽에 놓는다. 오른쪽 더듬이 냄새 막대 ↑ → Inspector에서 ORN → central → DNa02 L/R 비대칭 → 파리가 오른쪽으로 돈다. 도착하면 접촉 ON → 주둥이.
3. **위협** — 그림자를 던진다. Fly POV에 looming 점수 ↑ → Inspector에서 sensory → central → descending 순서 점등 → DNp01 → 파리 점프.
4. **Neural Event Replay** — 직전 0.3초를 되감아 순서를 보여준다. "순서이지 인과 증명은 아닙니다."
5. **Brain Surgery** — `ESCAPE LESION` 프리셋: DNp01 SILENCED.
6. **같은 자극 재실행** — 같은 그림자 궤적·타이밍. 좌우 분할: NORMAL은 점프, SILENCED는 Inspector에서 LC4/LPLC2까지 켜지고 DNp01에서 끊김, 행동 없음.
7. **결론** — "초파리 모양의 스크립트 에이전트가 아니라, 실제 커넥톰 기반 신경 시뮬레이션을 몸과 연결하고, 감각 입력·뇌 활동·행동을 한 화면에서 관찰하며 네트워크 개입까지 할 수 있는 embodied connectome simulator를 만들었습니다." 이어서 한계 슬라이드(§12).

5·6번은 3주차 Level C가 성립해야 가능. 실패 시 대체 시나리오: STEERING LESION(DNa02 한쪽 끄기 → 한 방향으로만 돎)으로 같은 구조를 보여준다.

## 12. 알려진 모델 한계 (Limitations 슬라이드)

Shiu LIF 모델은 완전한 기능 모델이 아니다: 모든 뉴런 단일 LIF · 형태·수용체 동역학·gap junction 무시 · 비스파이킹 뉴런 부정확 · 신경펩타이드·신경조절·내부 상태 없음 · 기저 발화율 0Hz · 신경전달물질 예측 불확실 · 절대 발화율 신뢰 불가 · 커넥톰은 동역학을 제약하지만 행동을 결정하지 않음. 해석은 **조건 간 차이, 출력 활성 여부, 상대 반응, 개입 전후 변화**로.
프로젝트 고유: 다리 협응은 CPG(뇌는 조절만) · Level C 전부 미검증 확장 · 시각은 망막이 아닌 LC 수준 주입 · 후각은 우리가 계산한 기하 농도 · Surgery 결과는 시뮬 내 관찰.

## 13. 리스크

| 리스크 | 대응 |
|---|---|
| Stage B 불일치 | Brian2 시드 편차 먼저 측정. 지연 링버퍼(18스텝), 불응기, 포아송 대상 불응기 0 [검증됨] 점검 |
| 뇌 GPU 속도 | 이벤트 구동 커널 → 임계값 가지치기 |
| GPU 8GB | 뇌만 GPU(v783 CSR 약 120MB + 상태 → 1GB 미만 예상, 1주차 실측). FlyGym CPU |
| snapshot/restore 느림 | 뇌 상태는 GPU 텐서 복사(수십 MB, ms 단위 예상), MuJoCo는 `mj_getState/setState`. 1주차 측정 |
| 겹눈 API 미작동/느림 | 2.x에 튜토리얼 없음. 1주차 확인. 렌더 주기를 뇌 dt와 분리(10ms마다) |
| 뇌·몸 시간축 | 뇌 N스텝 ↔ 물리 M스텝 고정 비율, 디코더 50ms 창. 실시간 미달 시 같은 배율로 |
| Level C 실패 | 직접 규칙 대체 + 표시. 시연은 STEERING LESION 대체 |
| 주둥이·날개 관절 없음 | 먹기는 정지+머리 숙임+표시로, 도망은 다리 점프로. 시각적 임팩트가 줄면 P1에서 rostrum 힌지 추가 시도 |
| v630 주석 매칭 83% | 1~2주차 패널은 매칭된 뉴런만 표시하고 "주석 없음" 집계를 따로 둠. Stage C 이후 완전 |
| 서버 단일 장애점 | systemd, 스냅샷, "파리 자는 중", 발표 당일 노트북 백업 |
| 라이선스 | flygym Apache-2.0, Shiu MIT, FlyWire CC-BY 4.0 [검증됨]; eonsystems GPL-2 읽기만 |
| 유사 프로젝트 | "원논문 모델 그대로 + 생체역학 몸 + 감각·뇌·개입이 한 화면" 조합은 미확인. 발표에서 출처 명시 |

## 14. 대회 평가 기준 대응
- 완성도 10점 — 5주차 공개 + 설문 10문항
- 성실성 10점 — 모듈(데이터/커널/검증/인코더/디코더/컨트롤러/POV/Inspector/Surgery/송출) 독립 → 일 단위 커밋, 주 5일+
- 상호+메타 10점 — §11 시연. Level 표·한계를 먼저 말해 신뢰 확보
- 생성형 AI 활용 — 코딩·디자인·문서로 충족. LLM 해설은 P2 가산

## 15. 1주차 할 일
1. 리포 생성: `brain/`(커널·검증·silence·snapshot), `body/`(flygym 연동·컨트롤러), `encoder/`, `panel/`(POV·Inspector), `stream/`, `data/`
2. **Stage A**: Shiu 리포 클론, conda, `example.ipynb`로 v630 당 GRN→MN9 재현(주의: 노트북 설명은 200Hz라 쓰지만 코드 기본값은 150Hz → 우리는 100Hz·150Hz 두 조건을 정답으로 저장). Brian2 시드 5개 편차. 원본 silencing 예제 1회 실행해 정답 저장. `figures.ipynb`에서 뉴런 ID 목록을 `data/paper_ids.json`으로 추출
3. 데이터: v630 → CSR(npz). 뉴런 수·edge 수·가중치 합 기록. 주석 TSV 조인(매칭률 기록)
4. **Stage B**: PyTorch CUDA LIF + `silence()` + `snapshot()/restore()` + 시드. §3.1 비교 스크립트
5. **FlyGym**: Python 3.12 venv, `pip install flygym[examples]`, 튜토리얼 4d 실행·영상 저장. `get_ommatidia_readouts` 호출해 겹눈 이미지 저장. 주둥이·더듬이 관절 확인
6. §7 측정 → `docs/benchmark-week1.md`
7. 하강뉴런 채널 정의 초안: DNp09, DNa01/DNa02 L/R, DNp01, MDN, LC4, LPLC2는 주석 cell_type에 있음 [검증됨] → v783 root_id로 뽑고, v630엔 그중 83%만 매칭되므로 v630 단계에선 매칭된 것만 사용. 정지 DN 타입은 [검증 과제]

## 참고
- 뇌 모델 코드: https://github.com/philshiu/Drosophila_brain_model (MIT; v630 기본)
- 몸: https://pypi.org/project/flygym/ (2.1.0, 2026-06-24) · https://github.com/NeLy-EPFL/flygym · https://neuromechfly.org · 1.x https://github.com/NeLy-EPFL/flygym-gymnasium
- NeuroMechFly v2 논문: https://www.nature.com/articles/s41592-024-02497-y
- 주석: https://github.com/flyconnectome/flywire_annotations
- GPU 참조(GPL-2, 읽기만): https://github.com/eonsystemspbc/fly-brain
- 채널 정의 참고(MIT): https://github.com/RaphaelSR/fly-brain-bench
- MaleCNS(P2): https://huggingface.co/spaces/Xenova/fruit-fly-simulation · https://github.com/ZeroXClem/closed-loop-fly
- 생태계: https://github.com/cobanov/awesome-fly
