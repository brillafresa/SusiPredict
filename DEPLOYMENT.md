# Deployment Guide

수시 합격 가능성 예측기 프로젝트의 배포 가이드입니다.

## 🚀 Streamlit Cloud 배포

### 1. GitHub 저장소 준비

#### 필수 파일 확인

- ✅ `app.py` (메인 Streamlit 앱)
- ✅ `requirements.txt` (Python 의존성)
- ✅ `.streamlit/config.toml` (Streamlit 설정)

#### 권장 파일

- ✅ `README.md` (프로젝트 설명)
- ✅ `LICENSE` (라이선스)
- ✅ `.gitignore` (Git 무시 파일)

### 2. Streamlit Cloud 설정

#### 1) Streamlit Cloud 접속

- [share.streamlit.io](https://share.streamlit.io) 방문
- GitHub 계정으로 로그인

#### 2) 새 앱 생성

- **New app** 클릭
- **Repository**: `YOUR_USERNAME/SusiPredict` 선택
- **Branch**: `main` 선택
- **Main file path**: `app.py` 입력

#### 3) 고급 설정 (선택사항)

- **Python version**: 3.10 또는 3.11 선택
- **App URL**: 원하는 서브도메인 설정

### 3. 배포 확인

#### 성공적인 배포 후

- 앱이 정상적으로 로드되는지 확인
- 모든 기능이 작동하는지 테스트
- 에러 로그 확인 (Streamlit Cloud 대시보드)

## 🌐 로컬 배포

### 1. 직접 실행

```bash
# 기본 실행
streamlit run app.py

# 특정 포트로 실행
streamlit run app.py --server.port 8501

# 헤드리스 모드 (백그라운드)
streamlit run app.py --server.headless true
```

### 2. Docker 배포

#### Dockerfile 생성

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Docker 실행

```bash
# 이미지 빌드
docker build -t susipredict .

# 컨테이너 실행
docker run -p 8501:8501 susipredict
```

### 3. 가상환경 배포

```bash
# 가상환경 생성
python -m venv venv

# 활성화
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
```

## 🔧 환경별 설정

### 개발 환경

```toml
# .streamlit/config.toml
[server]
headless = false
port = 8501
enableCORS = true

[browser]
gatherUsageStats = true
```

### 프로덕션 환경

```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## 📊 모니터링 및 로깅

### Streamlit Cloud 모니터링

- **App Health**: 앱 상태 확인
- **Error Logs**: 오류 로그 모니터링
- **Performance**: 성능 지표 확인

### 로컬 모니터링

```bash
# 포트 사용량 확인
netstat -an | findstr :8501  # Windows
netstat -an | grep :8501     # Unix

# 프로세스 확인
tasklist | findstr python    # Windows
ps aux | grep streamlit      # Unix
```

## 🚨 문제 해결

### 일반적인 배포 문제

#### 1. Import 오류

```bash
# 의존성 재설치
pip install -r requirements.txt --force-reinstall

# 가상환경 확인
python -c "import sys; print(sys.executable)"
```

#### 2. 포트 충돌

```bash
# 포트 사용 중인 프로세스 확인
netstat -ano | findstr :8501

# 프로세스 종료
taskkill /PID <PID> /F
```

#### 3. 메모리 부족

```bash
# Python 메모리 제한 설정
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
streamlit run app.py --server.maxUploadSize=200
```

### Streamlit Cloud 특정 문제

#### 1. 배포 실패

- GitHub 저장소 권한 확인
- `requirements.txt` 문법 오류 확인
- Python 버전 호환성 확인

#### 2. 앱 로딩 실패

- `app.py` 파일 경로 확인
- 의존성 설치 오류 확인
- 로그에서 구체적인 오류 메시지 확인

## 🔒 보안 고려사항

### 프로덕션 환경

- **HTTPS**: SSL/TLS 인증서 설정
- **인증**: 필요시 사용자 인증 추가
- **데이터 보호**: 민감한 데이터 암호화

### 환경 변수

```bash
# .env 파일 (로컬 개발용)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Streamlit Cloud secrets.toml
[api_keys]
openai_api_key = "your-api-key"
```

## 📈 성능 최적화

### 앱 최적화

- **캐싱**: `@st.cache_data` 사용
- **지연 로딩**: 필요한 시점에 데이터 로드
- **이미지 최적화**: 적절한 이미지 크기 사용

### 배포 최적화

- **CDN**: 정적 파일 CDN 사용
- **로드 밸런싱**: 여러 인스턴스 분산
- **모니터링**: 성능 지표 지속적 모니터링

## 🔄 업데이트 및 유지보수

### 정기 업데이트

- **의존성**: 주기적 보안 업데이트
- **Streamlit**: 최신 버전으로 업그레이드
- **코드**: 정기적인 코드 리뷰 및 리팩토링

### 롤백 전략

- **Git 태그**: 안정적인 버전 태그 생성
- **백업**: 배포 전 백업 생성
- **단계적 배포**: 점진적 배포로 리스크 최소화

---

## 📞 지원

배포 관련 문제가 있으시면:

1. **GitHub Issues**: 구체적인 오류 메시지와 함께 이슈 생성
2. **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)에서 질문
3. **프로젝트 Discussions**: GitHub Discussions 활용

**행운을 빕니다!** 🚀
