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
