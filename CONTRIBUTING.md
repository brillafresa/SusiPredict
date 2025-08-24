# Contributing to SusiPredict

수시 합격 가능성 예측기 프로젝트에 기여해주셔서 감사합니다! 이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 🚀 시작하기

### 1. 저장소 포크 및 클론

```bash
# GitHub에서 저장소를 포크한 후
git clone https://github.com/YOUR_USERNAME/SusiPredict.git
cd SusiPredict
```

### 2. 개발 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발 도구
```

### 3. 개발 브랜치 생성

```bash
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/your-bug-fix
```

## 🧪 개발 가이드라인

### 코드 스타일

- **Python**: PEP 8 준수
- **포맷팅**: Black 사용 (라인 길이 127자)
- **Import 정렬**: isort 사용
- **린팅**: flake8 사용

### 테스트 작성

- 모든 새로운 기능에 대한 테스트 작성
- 기존 테스트가 모두 통과하는지 확인
- 테스트 커버리지 유지

```bash
# 테스트 실행
pytest tests/ -v

# 커버리지 확인
pytest tests/ --cov=src/ --cov-report=html
```

### 커밋 메시지 규칙

```
type(scope): description

feat: 새로운 기능
fix: 버그 수정
docs: 문서 수정
style: 코드 스타일 변경
refactor: 코드 리팩토링
test: 테스트 추가/수정
chore: 빌드 프로세스 또는 보조 도구 변경
```

## 📋 Pull Request 프로세스

### 1. 변경사항 준비

- 코드 변경
- 테스트 작성 및 실행
- 문서 업데이트 (필요시)

### 2. PR 생성

- 명확한 제목과 설명
- 변경사항 요약
- 테스트 결과 포함
- 스크린샷 (UI 변경시)

### 3. 코드 리뷰

- 리뷰어의 피드백 반영
- CI/CD 체크 통과 확인

### 4. 머지

- 승인 후 `main` 브랜치로 머지

## 🐛 버그 리포트

버그를 발견하셨다면:

1. **이슈 검색**: 이미 보고된 버그인지 확인
2. **새 이슈 생성**: 명확한 제목과 설명
3. **재현 단계**: 버그 재현 방법 상세 설명
4. **환경 정보**: OS, Python 버전, 브라우저 등

## 💡 기능 제안

새로운 기능을 제안하고 싶다면:

1. **이슈 생성**: 기능 제안 이슈
2. **사용 사례**: 구체적인 사용 시나리오
3. **구현 방향**: 구현 아이디어 제시
4. **우선순위**: 기능의 중요도 설명

## 🔧 개발 환경 설정

### 필수 도구

- Python 3.10+
- Git
- pip

### 권장 도구

- VS Code 또는 PyCharm
- pre-commit hooks
- Docker (선택사항)

### pre-commit 설정

```bash
pip install pre-commit
pre-commit install
```

## 📚 문서화

### 코드 주석

- 복잡한 로직에 대한 설명
- 함수/클래스 docstring
- 중요한 수식이나 알고리즘 설명

### README 업데이트

- 새로운 기능 설명
- 사용법 예시
- 스크린샷 추가

### CHANGELOG

- 모든 변경사항 기록
- Semantic Versioning 준수

## 🚨 문제 해결

### 일반적인 문제들

#### 테스트 실패

```bash
# 의존성 재설치
pip install -r requirements.txt --force-reinstall

# 캐시 정리
pytest --cache-clear
```

#### Streamlit 실행 오류

```bash
# 포트 충돌 확인
netstat -an | findstr :8501

# 다른 포트 사용
streamlit run app.py --server.port 8502
```

#### Import 오류

```bash
# Python 경로 확인
python -c "import sys; print(sys.path)"

# 가상환경 활성화 확인
which python  # Unix
where python  # Windows
```

## 📞 도움 요청

문제가 해결되지 않는다면:

1. **이슈 검색**: 유사한 문제 해결 방법 확인
2. **Discussions**: GitHub Discussions에서 질문
3. **이슈 생성**: 구체적인 오류 메시지와 함께

## 🎯 기여 영역

### 우선순위 높음

- 버그 수정
- 성능 개선
- 테스트 커버리지 향상
- 문서 개선

### 우선순위 중간

- 새로운 기능 추가
- UI/UX 개선
- 코드 리팩토링

### 우선순위 낮음

- 코드 스타일 개선
- 주석 추가
- 로깅 개선

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 기여하시는 모든 코드는 동일한 라이선스를 따릅니다.

---

**감사합니다!** 🎉

프로젝트에 기여해주셔서 정말 감사합니다. 함께 더 나은 수시 합격 가능성 예측기를 만들어가요!
