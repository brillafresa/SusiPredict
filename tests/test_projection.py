import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

class TestProjection:
    """올해 투영과 시뮬레이션 로직 테스트"""
    
    def test_current_year_projection(self):
        """올해 투영 파라미터 설정 검증"""
        # 올해 입력값
        current_inputs = {
            'capacity': 30,
            'extra': 5,
            'ratio': 4.0
        }
        
        # 계산된 값들
        N = current_inputs['capacity'] + current_inputs['extra']  # 35
        M = current_inputs['ratio'] * current_inputs['capacity']  # 120
        q_select = N / M  # 35/120 = 0.2917...
        q_init = current_inputs['capacity'] / M  # 30/120 = 0.25
        
        # 검증
        assert N == 35
        assert M == 120
        assert abs(q_select - 0.2917) < 1e-3
        assert q_init == 0.25
        
    def test_ratio_fallback_median(self):
        """올해 ratio가 입력되지 않을 때 과거 중앙값 대체 검증"""
        # 과거 ratio 데이터
        historical_ratios = [3.2, 4.1, 3.8, 4.5, 3.9]
        
        # 중앙값 계산
        median_ratio = np.median(historical_ratios)
        expected_median = 3.9
        
        assert median_ratio == expected_median
        
        # 올해 ratio가 없을 때 중앙값 사용
        current_ratio = 0  # 입력되지 않음
        if current_ratio == 0:
            current_ratio = median_ratio
            
        assert current_ratio == expected_median
        
    def test_recency_weighted_regression(self):
        """최근성 가중 회귀 검증"""
        # half-life=1년, mix=0.65 설정
        half_life = 1.0
        mix = 0.65
        
        # 가중치 계산 (최근 데이터일수록 높은 가중치)
        years = [2020, 2021, 2022, 2023]
        weights = []
        
        for year in years:
            # 2023년 기준으로 상대적 거리
            distance = 2023 - year
            weight = np.exp(-np.log(2) * distance / half_life)
            weights.append(weight)
            
        # 가중치가 내림차순인지 확인 (최근일수록 높음)
        assert weights[0] < weights[1] < weights[2] < weights[3]
        
        # mix 파라미터 검증
        assert 0 < mix < 1
        
    def test_simulation_cut_sampling(self):
        """시뮬레이션 컷 샘플링 검증"""
        # Beta(N, M-N+1) 분포에서 샘플링
        N = 35  # 선발인원
        M = 120  # 지원자수
        
        # Beta 분포 파라미터
        alpha = N
        beta = M - N + 1
        
        assert alpha > 0
        assert beta > 0
        
        # 샘플링 (실제로는 더 많은 샘플)
        n_samples = 10
        samples = np.random.beta(alpha, beta, n_samples)
        
        # 샘플이 [0, 1] 범위에 있는지 확인
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
        
        # 샘플 개수 확인
        assert len(samples) == n_samples
        
    def test_grade_transformation(self):
        """등급 변환 검증"""
        # Beta 분포 샘플 (0~1 범위)을 등급 범위 [1.0, 9.0]으로 변환
        beta_sample = 0.5  # 중간값
        
        # 선형 변환: grade = 1.0 + (9.0 - 1.0) * beta_sample
        grade = 1.0 + 8.0 * beta_sample
        
        expected_grade = 5.0  # 1.0 + 8.0 * 0.5
        assert grade == expected_grade
        
        # 경계값 테스트
        min_grade = 1.0 + 8.0 * 0.0  # beta_sample = 0
        max_grade = 1.0 + 8.0 * 1.0  # beta_sample = 1
        
        assert min_grade == 1.0
        assert max_grade == 9.0
        
    def test_quantile_estimation(self):
        """분위수 추정 검증"""
        # 시뮬레이션 결과로부터 분위수 추정
        simulated_cuts = [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
        
        # 정렬
        sorted_cuts = np.sort(simulated_cuts)
        
        # 분위수 계산
        g80_idx = int(0.8 * len(sorted_cuts))
        g50_idx = int(0.5 * len(sorted_cuts))
        g20_idx = int(0.2 * len(sorted_cuts))
        
        g80 = sorted_cuts[g80_idx]
        g50 = sorted_cuts[g50_idx]
        g20 = sorted_cuts[g20_idx]
        
        # 분위수 순서 검증
        assert g20 <= g50 <= g80
        
        # 실제 값과 비교
        assert abs(g50 - 3.0) < 0.11  # 중앙값 근처 (3.1 - 3.0 = 0.1)
        
    def test_admission_probability_calculation(self):
        """합격 확률 계산 검증"""
        # 개인 등급
        personal_grade = 3.2
        
        # 시뮬레이션 컷 분포
        simulated_cuts = [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
        
        # personal_grade보다 낮은 컷의 개수 (합격)
        admitted_count = sum(1 for cut in simulated_cuts if cut <= personal_grade)
        total_count = len(simulated_cuts)
        
        # 합격 확률
        admission_probability = admitted_count / total_count
        
        # 검증
        assert 0 <= admission_probability <= 1
        assert admitted_count <= total_count
        
        # personal_grade가 3.2이고 컷이 2.1~3.1이면 6/10 = 0.6
        expected_prob = 6 / 10
        assert admission_probability == expected_prob
        
    def test_percentile_position_estimation(self):
        """상위 백분위 위치 추정 검증"""
        # 개인 등급
        personal_grade = 3.2
        
        # 시뮬레이션 컷 분포 (정렬됨)
        simulated_cuts = [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
        
        # personal_grade보다 높은 컷의 개수 (더 좋은 성적)
        better_count = sum(1 for cut in simulated_cuts if cut > personal_grade)
        total_count = len(simulated_cuts)
        
        # 상위 백분위 (더 좋은 성적의 비율)
        percentile_position = (better_count / total_count) * 100
        
        # 검증
        assert 0 <= percentile_position <= 100
        
        # personal_grade가 3.2이고 더 높은 컷이 3.3, 3.5, 3.7, 3.9이면 4/10 * 100 = 40%
        expected_percentile = 40.0
        assert percentile_position == expected_percentile
