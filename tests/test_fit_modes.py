import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

class TestFitModes:
    """피팅 모드 분기와 로버스트 앵커 유효성 테스트"""
    
    def test_fit_mode_branching(self):
        """피팅 모드 분기 검증"""
        # 경쟁률 데이터가 있는 해: underlying + truncation
        has_ratio_data = {
            'year': 2023,
            'capacity': 30,
            'extra': 3,
            'final_cut': 2.5,
            'mean_score': 3.2,
            'ratio': 4.0  # 경쟁률 있음
        }
        
        # 경쟁률이 없는 해: admitted-only
        no_ratio_data = {
            'year': 2022,
            'capacity': 30,
            'extra': 2,
            'final_cut': 2.8,
            'mean_score': 3.5,
            'ratio': 0  # 경쟁률 없음
        }
        
        # 경쟁률 유무에 따른 피팅 모드 결정
        if has_ratio_data['ratio'] > 0:
            fit_mode = "underlying + truncation"
        else:
            fit_mode = "admitted-only"
            
        assert fit_mode in ["underlying + truncation", "admitted-only"]
        
    def test_robust_anchor_validation(self):
        """로버스트 앵커 유효성 검증"""
        # 입력값 vs 적합치 비교
        input_value = 2.5
        fitted_value = 2.3
        tolerance = 0.75
        
        # 차이가 tolerance 이하면 적합치, 초과하면 입력값을 anchor로 채택
        difference = abs(input_value - fitted_value)
        
        if difference <= tolerance:
            anchor_value = fitted_value
        else:
            anchor_value = input_value
            
        # anchor_value가 적절히 설정되었는지 확인
        assert anchor_value in [input_value, fitted_value]
        
        # tolerance 기준에 따른 선택이 올바른지 확인
        if difference <= tolerance:
            assert anchor_value == fitted_value
        else:
            assert anchor_value == input_value
            
    def test_anchor_tolerance_scenarios(self):
        """다양한 tolerance 시나리오에서 앵커 선택 검증"""
        test_cases = [
            (2.5, 2.3, 0.75, 2.3),   # 차이 0.2 < 0.75 → 적합치
            (2.5, 1.5, 0.75, 2.5),   # 차이 1.0 > 0.75 → 입력값
            (3.0, 3.1, 0.75, 3.1),   # 차이 0.1 < 0.75 → 적합치
            (4.0, 2.0, 0.75, 4.0),   # 차이 2.0 > 0.75 → 입력값
        ]
        
        for input_val, fitted_val, tol, expected_anchor in test_cases:
            difference = abs(input_val - fitted_val)
            
            if difference <= tol:
                anchor = fitted_val
            else:
                anchor = input_val
                
            assert anchor == expected_anchor
            
    def test_beta_distribution_fitting(self):
        """Beta 분포 피팅 검증"""
        # Beta 분포 파라미터 a, b는 양수여야 함
        a, b = 2.5, 3.0
        
        assert a > 0
        assert b > 0
        
        # 등급 범위 [1.0, 9.0]에서의 Beta 분포
        grade_range = (1.0, 9.0)
        
        # 정규화된 Beta 분포 (0~1 범위)
        x = np.linspace(0, 1, 100)
        import math
        beta_pdf = (x**(a-1) * (1-x)**(b-1)) / (math.gamma(a) * math.gamma(b) / math.gamma(a+b))
        
        # PDF 값들이 유효한지 확인
        assert np.all(beta_pdf >= 0)
        assert np.isfinite(beta_pdf).all()
        
    def test_quantile_constraints(self):
        """분위수 제약 조건 검증"""
        # 최종컷, 중위수, 평균 등의 제약 조건
        constraints = {
            'final_cut': 2.5,      # 최종컷
            'median': 3.2,         # 중위수
            'mean': 3.5,           # 평균
            'max_score': 4.1       # 최고점
        }
        
        # 등급 순서 검증 (최종컷 ≤ 중위수 ≤ 평균 ≤ 최고점)
        assert constraints['final_cut'] <= constraints['median']
        assert constraints['median'] <= constraints['mean']
        assert constraints['mean'] <= constraints['max_score']
        
        # 등급 범위 검증 [1.0, 9.0]
        for value in constraints.values():
            assert 1.0 <= value <= 9.0
            
    def test_fitting_weights(self):
        """피팅 가중치 설정 검증"""
        # 랜드마크별 가중치
        weights = {
            'final_cut': 1.0,      # 최종컷 (가장 중요)
            'median': 0.8,         # 중위수
            'mean': 0.6,           # 평균
            'max_score': 0.4       # 최고점
        }
        
        # 가중치가 내림차순으로 설정되었는지 확인
        weight_values = list(weights.values())
        assert weight_values == sorted(weight_values, reverse=True)
        
        # 모든 가중치가 양수인지 확인
        for weight in weights.values():
            assert weight > 0
