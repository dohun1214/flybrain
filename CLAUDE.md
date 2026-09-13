# flybrain — 프로젝트 작업 규칙 (Claude용)

## 프로젝트 한 줄
실제 초파리 커넥톰 기반 전뇌 LIF 시뮬레이션(Shiu et al. 2024)을 자택 GPU에서 돌리고, 하강뉴런 활동을 NeuroMechFly v2(FlyGym 2.1) 몸의 명령으로 변환해, 3D 물리 세계에서 사는 파리 한 마리를 — 감각 입력(Fly POV), 뇌 활동(Brain Inspector), 뉴런 끄기 실험(Brain Surgery)과 함께 — 누구나 구경하는 embodied connectome simulator. 교내 AI 경진대회(2026년 11월) 출품작.

**전체 계획서: `docs/plan.md` (v5.1).** 작업 전 반드시 읽을 것. 모든 기술 항목에 [검증됨]/[근거 있음/모델 미검증]/[직접 설계]/[검증 과제] 라벨이 있다 — 라벨을 존중하고, 미검증을 검증된 것처럼 쓰지 않는다.

## 절대 규칙
- **커밋 메시지에 Claude/AI 관련 문구·서명·트레일러를 넣지 않는다.** `Co-Authored-By`, `Generated with`, 세션 링크 전부 금지. 커밋 메시지는 변경 내용만.
- 작업 흐름: GitHub 이슈 → 브랜치(`feat/<이슈번호>-<짧은설명>`, `fix/…`, `chore/…`) → 커밋 → 푸시 → `gh pr create` → 사용자가 머지.
- 커밋은 작게, 매일. (대회 성실성 점수 = GitHub 커밋 참여일 수)
- 원논문 파라미터를 임의로 바꾸지 않는다. 바꿔야 하면 이유를 `docs/decisions.md`에 기록.
- GPL 코드(eonsystems/fly-brain)는 읽기만, 복사 금지.

## 환경
- Windows 11, RTX 4060 Ti **8GB**, gh CLI 로그인됨(dohun1214), git 2.54
- Python: 시스템 3.10 / 3.13 설치됨, `uv` 있음. **FlyGym 2.1은 Python ≥3.12 필요** → `uv venv --python 3.13 .venv` 로 프로젝트 venv 생성
- 뇌 시뮬만 GPU(PyTorch CUDA), FlyGym은 CPU. Warp 백엔드는 쓰지 않는다(GPU 공유 불가)
- 설치: `uv pip install flygym[examples] torch --index-url …(CUDA 휠)`, Shiu 원본 재현용은 별도 conda/venv에 brian2

## 디렉터리
```
brain/    PyTorch LIF 커널, silence/snapshot/seed, Brian2 비교 스크립트
body/     FlyGym 연동, 행동 중재기, 점프·그루밍 컨트롤러
encoder/  감각 인코더 (겹눈→looming, 냄새 농도 계산, 접촉→GRN/JON)
panel/    Fly POV, Brain Inspector, Event Replay 렌더러
stream/   화면 합성·영상 송출
data/     paper_ids.json(논문 뉴런 ID), CSR 변환 스크립트, (대용량 데이터는 gitignore)
scripts/  벤치마크·측정 스크립트
docs/     plan.md, benchmark-week1.md(측정 결과), decisions.md
```

## 1주차 이슈 (GitHub Issues #1~#7 참고)
1. 환경 세팅 (uv venv 3.13, torch CUDA, flygym)
2. Stage A: Shiu 원본 Brian2로 v630 당 GRN→MN9 재현, 시드 5개 편차, silencing 예제 정답 저장
3. 데이터: v630/v783 → CSR(npz), 뉴런·edge·시냅스 수 메타데이터, 주석 TSV 조인
4. Stage B: PyTorch CUDA LIF + silence/snapshot/seed, Brian2와 비교
5. FlyGym: 튜토리얼 4d 실행, 음수 신호 후진 시험, get_ommatidia_readouts 확인
6. 성능 측정 → docs/benchmark-week1.md
7. 하강뉴런 채널 정의 초안

## 커널 구현 시 놓치기 쉬운 것 (model.py에서 확인됨)
- 스파이크 시 `v = v_rst`, **`g = 0`** (시냅스 변수도 리셋)
- 불응기(2.2ms) 동안 **v와 g 모두 동결** (`unless refractory`)
- 포아송 자극 대상 뉴런은 **불응기 0**
- 포아송 이벤트 가중치 = 0.275mV × 250 = 68.75mV (한 번에 발화)
- 시냅스 지연 1.8ms = 0.1ms dt에서 18스텝 링버퍼
- silencing = 해당 뉴런의 **출력 시냅스만** 0 (README와 달리 코드는 outgoing만)
- 노트북 설명은 200Hz라 쓰지만 코드 기본 `r_poi`는 150Hz

## 데이터 사실 (2026-09-13 리포 파일에서 계산)
| | v630 | v783 |
|---|---|---|
| 뉴런 | 127,400 | 138,639 |
| 뉴런쌍 edge | 14,687,178 | 15,091,983 |
| 시냅스 합 | 52,793,639 | 54,492,922 |
| ≥5 시냅스 연결 | 2,614,028 | 2,700,513 |

- 논문 뉴런 ID: `data/paper_ids.json` (v630 root_id). 논문 명칭(aDN1, aBN1, GF, MN9)은 FlyWire cell_type에 **없다** — ID로 찾는다. aDN1=`DNg62`, aDN2=`DNge078`, aBN1=`SAD093`, MN9=`CB0701`(super_class motor). 당·물 GRN은 둘 다 cell_type `LB3`라 타입으로 구분 불가.
- 주석 TSV(`flyconnectome/flywire_annotations`)는 v783 기준: v783 99.99% 매칭, v630은 83%만.
- FlyGym 2.1 몸: 관절 87개(다리 42 + 머리 3 + 더듬이 18…). **주둥이·날개·복부에 관절 없음.** 후각 API 없음(직접 계산). `HybridTurningController.step(descending_signal(2,), obs)` — 음수 신호 = CPG 역방향.

## 참고 리포
- 뇌 모델 원본: https://github.com/philshiu/Drosophila_brain_model (MIT)
- 몸: https://github.com/NeLy-EPFL/flygym (2.1.0, Apache-2.0), 문서 https://neuromechfly.org
- 주석: https://github.com/flyconnectome/flywire_annotations
- 채널 정의 참고: https://github.com/RaphaelSR/fly-brain-bench (MIT)
