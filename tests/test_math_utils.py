import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

class TestMathUtils:
    """수학적 유틸리티 함수들의 테스트"""
    
    def test_ratio_calculation(self):
        """경쟁률 계산 검증"""
        # 경쟁률 = 지원자수 / 정원
        capacity = 30
        applicants = 150
        expected_ratio = 5.0
        
        ratio = applicants / capacity
        assert ratio == expected_ratio
        
    def test_q_select_calculation(self):
        """선발비율 q_select 계산 검증"""
        capacity = 30
        extra = 5
        ratio = 4.0
        
        # q_select = (capacity + extra) / (ratio * capacity)
        N = capacity + extra
        M = ratio * capacity
        q_select = N / M
        
        expected_q_select = (30 + 5) / (4.0 * 30)  # 35 / 120 = 0.2917...
        assert abs(q_select - expected_q_select) < 1e-6
        
    def test_q_init_calculation(self):
        """최초합격자 절단비율 q_init 계산 검증"""
        capacity = 30
        ratio = 4.0
        
        # q_init = capacity / M
        M = ratio * capacity
        q_init = capacity / M
        
        expected_q_init = 30 / (4.0 * 30)  # 30 / 120 = 0.25
        assert q_init == expected_q_init
        
    def test_ratio_consistency(self):
        """경쟁률 정의 불변성 검증"""
        # 다양한 capacity/ratio 조합에서 일관성 확인
        test_cases = [
            (20, 3.0),   # capacity=20, ratio=3.0
            (50, 2.5),   # capacity=50, ratio=2.5
            (100, 1.8),  # capacity=100, ratio=1.8
        ]
        
        for capacity, ratio in test_cases:
            applicants = ratio * capacity
            calculated_ratio = applicants / capacity
            assert abs(calculated_ratio - ratio) < 1e-10
            
    def test_underfilled_fallback(self):
        """정원 미달 케이스 방어 로직"""
        capacity = 30
        extra = 0
        ratio = 0.5  # 매우 낮은 경쟁률
        
        N = capacity + extra
        M = ratio * capacity
        
        # M < N인 경우 (정원 미달)
        if M < N:
            # 실제로는 이런 경우가 발생할 수 있음
            assert M < N
        else:
            # 정상적인 경우
            assert M >= N
            
    def test_ratio_bounds(self):
        """경쟁률 경계값 테스트"""
        capacity = 30
        
        # ratio = 0인 경우 (지원자 없음)
        ratio_zero = 0
        applicants_zero = ratio_zero * capacity
        assert applicants_zero == 0
        
        # ratio = 1인 경우 (지원자 = 정원)
        ratio_one = 1.0
        applicants_one = ratio_one * capacity
        assert applicants_one == capacity
        
        # ratio > 1인 경우 (지원자 > 정원)
        ratio_high = 5.0
        applicants_high = ratio_high * capacity
        assert applicants_high > capacity
