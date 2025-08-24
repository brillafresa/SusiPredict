# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 프로젝트 구조화 및 문서화
- 테스트 프레임워크 구축
- CI/CD 워크플로우 설정

## [6.2.0] - 2024-12-19

### Added

- 수시 합격 가능성 예측기 Streamlit 앱
- Beta 분포 기반 확률적 모델링
- 과거 데이터 기반 예측 알고리즘
- 시각화 차트 (연도별 추세, 합격 가능성 분포)
- JSON 데이터 저장/불러오기 기능
- 디버그 모드 (전처리/피팅/투영/시뮬 결과 출력)

### Features

- **분포 가정**: 전체 지원자 분포를 등급 범위 [1.0, 9.0]에서 Beta(a,b)로 모형화
- **피팅 전략**:
  - 경쟁률/충원 데이터 있는 해: underlying + truncation 방식
  - 경쟁률 없는 해: admitted-only 방식
- **로버스트 앵커**: 입력값 vs 적합치 비교로 anchor\_\* 열 생성
- **올해 투영**: 최근성 가중 회귀로 anchor 시계열 예측
- **시뮬레이션**: N, M 기반으로 최종컷 샘플링 및 합격 확률 계산

### Technical Details

- **핵심 정의**: 경쟁률 = 지원자수/정원, 추가충원은 정원 미달분 추가선발
- **선발비율**: q_select = (capacity + extra) / (ratio × capacity)
- **최초합격자 절단비율**: q_init = capacity / M
- **지원자수 계산**: M = ratio × capacity (과거/올해 모두 동일)

## [6.1.0] - 2024-12-01

### Added

- 초기 프로토타입 개발
- 기본 수학적 모델링 로직

## [6.0.0] - 2024-11-15

### Added

- 프로젝트 초기 설정
- 기본 아키텍처 설계
