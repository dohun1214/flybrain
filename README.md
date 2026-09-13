# flybrain

실제 초파리 커넥톰(FlyWire) 기반 전뇌 LIF 시뮬레이션을 NeuroMechFly v2 생체역학 몸과 연결한 embodied connectome simulator.

뇌는 Shiu et al. 2024의 모델을 원본 파라미터 그대로 GPU에서 돌리고, 하강뉴런 활동을 FlyGym 2.1 몸의 명령으로 변환해 3D 물리 세계에서 파리 한 마리가 걷고·먹고·도망치고·그루밍한다. 화면에는 파리가 느끼는 감각 입력(Fly POV), 뇌 내부 활동(Brain Inspector), 그리고 뉴런 집단을 끄고 같은 자극을 다시 줬을 때의 변화(Brain Surgery)가 함께 보인다.

- 계획서: [docs/plan.md](docs/plan.md)
- 상태: 1주차 (뇌 모델 재현 + FlyGym 가동)

## 구성
```
brain/    PyTorch CUDA LIF (Shiu 2024 파라미터), silencing, 상태 스냅샷
body/     FlyGym 연동, 행동 중재기, 관절 컨트롤러
encoder/  감각 인코더 (겹눈 → looming, 냄새 농도, 접촉)
panel/    Fly POV, Brain Inspector, Neural Event Replay
stream/   화면 합성·송출
data/     논문 뉴런 ID, 데이터 변환 스크립트
scripts/  벤치마크
docs/     계획서, 측정 결과, 결정 기록
```

## 출처
- 뇌 모델: Shiu et al., *Nature* 2024 — https://github.com/philshiu/Drosophila_brain_model (MIT)
- 몸: NeuroMechFly v2 / FlyGym — https://github.com/NeLy-EPFL/flygym (Apache-2.0)
- 커넥톰·주석: FlyWire (CC-BY 4.0), https://github.com/flyconnectome/flywire_annotations
