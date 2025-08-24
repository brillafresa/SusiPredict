# 수시 합격 가능성 예측기 (Streamlit)

## 🎯 프로젝트 개요

수시 입시 결과 데이터를 기반으로 올해 합격 컷의 확률분포를 추정하고, 개인 등급에 대한 합격 확률 및 포지션을 시각화하는 웹 애플리케이션입니다.

## 🧮 핵심 수식 정의

### 기본 개념

- **경쟁률 (ratio) = 지원자수 (applicants) / 정원 (capacity)**
- **추가충원 (extra)**은 "정원 미달분에 대한 추가선발"이며 정원을 늘리지 않음
- **올해 선발인원 N = capacity + extra**
- **지원자수 M = ratio × capacity** (과거/올해 모두 동일 정의)

### 선발 비율 계산

- **선발비율 q_select = N / M = (capacity + extra) / (ratio × capacity)** (상한 1.0)
- **최초합격자 절단비율 q_init = capacity / M**

### 분포 모델링

- 전체 지원자 분포: 등급 범위 [1.0, 9.0]에서 **Beta(a,b)** 분포
- 피팅 전략:
  1. 경쟁률 데이터 있는 해: 전체 지원자 분포에서 상위 q_select 절단된 합격자 분포의 분위수/평균 조건을 역문제로 피팅
  2. 경쟁률 없는 해: 합격자 분포 자체를 Beta로 근사하여 입력 지표 맞춤

## 🚀 주요 기능

- **과거 데이터 분석**: 연도별 입시결과(정원/추가충원/컷/평균/최고 등)와 경쟁률 입력
- **예측 모델링**: Beta 분포 기반 확률적 모델로 올해 합격 컷 추정
- **시각화**:
  - 좌측: 연도별 추세 (원본값 ×마커 / 적합치 o마커)
  - 우측: 시뮬레이션 컷 분포 히스토그램 + 영역(안정/적정/소신/위험)
- **데이터 관리**: JSON 형태로 데이터 저장/불러오기
- **디버그 모드**: 전처리/피팅/투영/시뮬 결과를 상세 출력

## 📦 설치 및 실행

### 요구사항

- Python 3.10+
- pip

### 설치

```bash
git clone <repository-url>
cd SusiPredict
pip install -r requirements.txt
```

### 로컬 실행

```bash
streamlit run app.py
```

### Streamlit Cloud 배포

1. GitHub 저장소와 연동
2. `app.py`를 메인 파일로 지정
3. Python 버전 및 requirements.txt 자동 인식

## 📊 데이터 형식

### JSON 저장 구조

```json
{
  "department_name": "학과명",
  "current_year_inputs": {
    "capacity": 30,
    "extra": 5,
    "applicants": 150,
    "ratio": 5.0
  },
  "historical_data": [
    {
      "year": 2023,
      "capacity": 30,
      "extra": 3,
      "final_cut": 2.5,
      "mean_score": 3.2,
      "max_score": 4.1,
      "applicants": 120,
      "ratio": 4.0
    }
  ]
}
```

### 입력 모드

- **지원자 수 모드**: applicants > 0 & ratio == 0
- **경쟁률 모드**: ratio > 0

## 🪛 디버그 모드 사용법

사이드바의 "디버그 모드 (중간값 출력)" 체크 시 다음 정보가 노출됩니다:

1. **전처리 요약**: year, capacity, extra, ratio(raw/effective), N, q_select, M
2. **피팅 요약**: 각 연도의 beta(a,b), fitted*\*와 anchor*\* 비교
3. **올해 파라미터**: a,b, q_select_this, N, M, ratio_this
4. **컷 샘플/분위수**: g80/g50/g20과 컷 샘플 10개

### 문제 분석 체크리스트

1. **q_select_this**가 과도하게 낮은지 확인 (N/M이 0.05~0.1대로 수렴하면 컷이 매우 낮아짐)
2. ratio_this 산출 경로 확인 (입력 vs 과거 중앙값)
3. **anchor_final/anchor_best**의 시계열 예측값이 비정상적으로 낮은지 확인
4. weights 설정이 랜드마크를 충분히 제약하고 있는지 확인
5. 필요 시 **mix(회귀 vs 직전값)** 또는 **anchor_tol** 조정

## 🧪 테스트

```bash
pytest tests/
```

### 테스트 케이스

- 경쟁률 정의 불변성
- q_select/q_init 계산 검증
- 피팅 모드 분기
- 로버스트 앵커 유효성
- 극단 케이스 방어
- 시뮬 컷 분위수 합리성

## 📁 프로젝트 구조

```
.
├─ app.py                  # Streamlit 메인 앱
├─ requirements.txt        # Python 패키지 의존성
├─ README.md              # 프로젝트 문서
├─ CHANGELOG.md           # 변경 이력
├─ .streamlit/
│   └─ config.toml        # Streamlit 설정
├─ src/
│   ├─ model.py           # 분포/피팅/투영 로직 모듈
│   └─ utils.py           # 공통 유틸리티
├─ tests/                 # 테스트 파일들
├─ .github/workflows/     # CI/CD 워크플로우
└─ LICENSE                # 라이선스
```

## 🤝 기여 가이드

1. `dev` 브랜치에서 개발
2. 테스트 코드 작성 및 실행
3. PR 생성 시 문제 배경, 변경점, 디버그 캡처 포함
4. 코드 리뷰 후 `main` 브랜치로 머지

## 📈 버전 정책

- **Semantic Versioning** 사용
- 현재 버전: `v6.2.0`
- 주요 변경사항은 `CHANGELOG.md`에 기록

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
