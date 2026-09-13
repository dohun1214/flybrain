# 환경 세팅 (이슈 #1)

확인 일자: 2026-09-13 · Windows 11 · i7-14700K (28 스레드) · 64GB RAM · RTX 4060 Ti 8GB · NVIDIA 드라이버 610.88 · uv 0.11.14

## 메인 환경 `.venv` (Python 3.13) — PyTorch 뇌 + FlyGym 몸

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r pyproject.toml
```

설치 확인 결과:

| 패키지 | 버전 |
|---|---|
| Python | 3.13.13 |
| torch | 2.14.0+cu130 (`torch.cuda.is_available()` → True, `NVIDIA GeForce RTX 4060 Ti`, CUDA 13.0) |
| flygym | 2.1.0 (`flygym`, `flygym_demo` import 성공) |
| mujoco | 3.9.0 |
| numpy | 2.5.3 |

## Shiu 원본 재현 환경 `.venv-brian2` (Python 3.10)

```powershell
uv venv --python 3.10 .venv-brian2
uv pip install --python .venv-brian2\Scripts\python.exe -r requirements-brian2.txt
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git external/Drosophila_brain_model
```

- brian2 2.5.1 / numpy 1.24.4 (원본 `environment.yml`과 동일 버전)
- 이 PC에는 C++ 컴파일러가 없어 Brian2가 **numpy 코드 생성 타깃**으로 동작한다 (Cython 불가 경고). 결과는 같고 속도만 느리다. v630 전뇌 1 trial 100ms 실행 = 2.8초(모델 빌드 포함).
- `external/`은 gitignore. 원본 리포에 v630/v783 데이터와 저자 실행 결과(`results/example/*.parquet`)가 동봉되어 있다.
