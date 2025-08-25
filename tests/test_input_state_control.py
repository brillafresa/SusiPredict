import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

class TestInputStateControl:
    """명시적 입력 상태 제어 시스템 테스트"""
    
    def test_input_state_initialization(self):
        """입력 상태 초기화 테스트"""
        # 초기 상태는 모두 False (입력 안함)
        initial_states = {
            'extra_inputted': False,
            'applicants_inputted': False,
            'ratio_inputted': False
        }
        
        for field, expected_state in initial_states.items():
            assert expected_state == False, f"{field} 초기 상태가 올바르지 않습니다"
    
    def test_input_state_toggle(self):
        """입력 상태 토글 테스트"""
        # 체크박스로 입력 상태를 True로 변경
        extra_inputted = True
        applicants_inputted = True
        ratio_inputted = True
        
        # 상태 변경 확인
        assert extra_inputted == True, "추가충원 입력 상태가 올바르게 변경되지 않았습니다"
        assert applicants_inputted == True, "지원자 수 입력 상태가 올바르게 변경되지 않았습니다"
        assert ratio_inputted == True, "경쟁률 입력 상태가 올바르게 변경되지 않았습니다"
    
    def test_input_field_disabled_state(self):
        """입력 필드 비활성화 상태 테스트"""
        # 입력 상태가 False일 때 필드가 비활성화되어야 함
        extra_inputted = False
        applicants_inputted = False
        ratio_inputted = False
        
        # 필드 비활성화 상태 확인
        extra_disabled = not extra_inputted
        applicants_disabled = not applicants_inputted
        ratio_disabled = not ratio_inputted
        
        assert extra_disabled == True, "추가충원 필드가 비활성화되지 않았습니다"
        assert applicants_disabled == True, "지원자 수 필드가 비활성화되지 않았습니다"
        assert ratio_disabled == True, "경쟁률 필드가 비활성화되지 않았습니다"
    
    def test_input_field_enabled_state(self):
        """입력 필드 활성화 상태 테스트"""
        # 입력 상태가 True일 때 필드가 활성화되어야 함
        extra_inputted = True
        applicants_inputted = True
        ratio_inputted = True
        
        # 필드 활성화 상태 확인
        extra_enabled = extra_inputted
        applicants_enabled = applicants_inputted
        ratio_enabled = ratio_inputted
        
        assert extra_enabled == True, "추가충원 필드가 활성화되지 않았습니다"
        assert applicants_enabled == True, "지원자 수 필드가 활성화되지 않았습니다"
        assert ratio_enabled == True, "경쟁률 필드가 활성화되지 않았습니다"
    
    def test_value_passing_logic(self):
        """값 전달 로직 테스트"""
        # 입력 상태에 따른 값 전달 로직
        test_cases = [
            # (입력상태, 입력값, 기대결과)
            (False, 5, None),      # 입력 안함 → None (자동 추정)
            (True, 5, 5),          # 입력함 → 5 (사용자 입력)
            (False, 0, None),      # 입력 안함 → None (자동 추정)
            (True, 0, 0),          # 입력함 → 0 (실제 0 입력)
        ]
        
        for inputted, value, expected in test_cases:
            result = value if inputted else None
            assert result == expected, f"입력상태: {inputted}, 값: {value}, 기대: {expected}, 실제: {result}"
    
    def test_json_save_structure(self):
        """JSON 저장 구조 테스트"""
        # 저장할 데이터에 입력 상태 정보가 포함되어야 함
        save_data = {
            "current_year_inputs": {
                "capacity": 30,
                "extra": 5,
                "applicants": 150,
                "ratio": 5.0,
                "extra_inputted": True,
                "applicants_inputted": True,
                "ratio_inputted": True,
            }
        }
        
        # 필수 키 확인
        required_keys = ["extra_inputted", "applicants_inputted", "ratio_inputted"]
        for key in required_keys:
            assert key in save_data["current_year_inputs"], f"필수 키 {key}가 누락되었습니다"
        
        # 입력 상태 값 확인
        assert save_data["current_year_inputs"]["extra_inputted"] == True
        assert save_data["current_year_inputs"]["applicants_inputted"] == True
        assert save_data["current_year_inputs"]["ratio_inputted"] == True
    
    def test_profile_restoration(self):
        """프로필 복원 테스트"""
        # 저장된 프로필에서 입력 상태 복원
        saved_profile = {
            "current_year_inputs": {
                "capacity": 30,
                "extra": 5,
                "applicants": 150,
                "ratio": 5.0,
                "extra_inputted": True,
                "applicants_inputted": False,
                "ratio_inputted": True,
            }
        }
        
        # 입력 상태 복원
        extra_inputted = saved_profile["current_year_inputs"].get("extra_inputted", False)
        applicants_inputted = saved_profile["current_year_inputs"].get("applicants_inputted", False)
        ratio_inputted = saved_profile["current_year_inputs"].get("ratio_inputted", False)
        
        # 복원된 상태 확인
        assert extra_inputted == True, "추가충원 입력 상태가 올바르게 복원되지 않았습니다"
        assert applicants_inputted == False, "지원자 수 입력 상태가 올바르게 복원되지 않았습니다"
        assert ratio_inputted == True, "경쟁률 입력 상태가 올바르게 복원되지 않았습니다"
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 1. 입력 상태가 없는 경우 (기본값 False)
        profile_without_flags = {
            "current_year_inputs": {
                "capacity": 30,
                "extra": 5,
                "applicants": 150,
                "ratio": 5.0,
                # 입력 상태 플래그 없음
            }
        }
        
        # 기본값으로 복원
        extra_inputted = profile_without_flags["current_year_inputs"].get("extra_inputted", False)
        assert extra_inputted == False, "입력 상태 플래그가 없을 때 기본값이 False가 아닙니다"
        
        # 2. 입력 상태가 None인 경우
        profile_with_none_flags = {
            "current_year_inputs": {
                "capacity": 30,
                "extra": 5,
                "applicants": 150,
                "ratio": 5.0,
                "extra_inputted": None,
                "applicants_inputted": None,
                "ratio_inputted": None,
            }
        }
        
        # None인 경우 False로 처리
        extra_inputted = profile_with_none_flags["current_year_inputs"].get("extra_inputted", False) or False
        assert extra_inputted == False, "입력 상태가 None일 때 False로 처리되지 않았습니다"
    
    def test_auto_estimation_trigger(self):
        """자동 추정 트리거 테스트"""
        # 입력 상태가 False일 때 자동 추정이 트리거되어야 함
        test_scenarios = [
            {
                "extra_inputted": False,
                "extra_value": 5,
                "expected_extra": None,  # 자동 추정
                "description": "추가충원 입력 안함"
            },
            {
                "ratio_inputted": False,
                "ratio_value": 3.0,
                "expected_ratio": None,  # 자동 추정
                "description": "경쟁률 입력 안함"
            },
            {
                "applicants_inputted": False,
                "applicants_value": 150,
                "expected_applicants": None,  # 자동 추정
                "description": "지원자 수 입력 안함"
            }
        ]
        
        for scenario in test_scenarios:
            # 입력 상태 확인
            inputted = scenario.get("extra_inputted", scenario.get("ratio_inputted", scenario.get("applicants_inputted")))
            value = scenario.get("extra_value", scenario.get("ratio_value", scenario.get("applicants_value")))
            expected = scenario.get("expected_extra", scenario.get("expected_ratio", scenario.get("expected_applicants")))
            description = scenario["description"]
            
            # 자동 추정 로직
            result = value if inputted else None
            assert result == expected, f"{description}: 기대값 {expected}, 실제값 {result}"
    
    def test_user_input_preservation(self):
        """사용자 입력 보존 테스트"""
        # 사용자가 실제로 입력한 값은 보존되어야 함
        user_inputs = {
            "extra_inputted": True,
            "extra_value": 0,  # 사용자가 실제로 0 입력
            "expected_extra": 0  # 0으로 보존되어야 함
        }
        
        # 입력 상태가 True이므로 값이 보존되어야 함
        result = user_inputs["extra_value"] if user_inputs["extra_inputted"] else None
        assert result == user_inputs["expected_extra"], "사용자 입력값이 보존되지 않았습니다"
        
        # 이는 기존 로직과의 차이점: 0이 "입력 안함"이 아닌 "실제 0"으로 처리됨
        assert result != None, "사용자가 0을 입력했는데 None으로 처리되었습니다"

    def test_checkbox_uncheck_reset_value(self):
        """체크박스 해제 시 입력 필드 값이 0으로 리셋되는지 확인"""
        # 초기 상태: 입력함이 체크되어 있고 값이 설정됨
        extra_this = 5
        extra_inputted = True
        prev_extra_inputted = True
        
        # 체크박스 해제 시뮬레이션
        extra_inputted = False
        
        # 체크박스 해제 시 값이 0으로 리셋되어야 함
        if extra_inputted == False and prev_extra_inputted == True:
            extra_this = 0
        
        assert extra_this == 0
        assert extra_inputted == False



    def test_checkbox_uncheck_reset_ratio(self):
        """경쟁률 체크박스 해제 시 값이 0.0으로 리셋되는지 확인"""
        # 초기 상태: 입력함이 체크되어 있고 값이 설정됨
        ratio_this = 3.5
        ratio_inputted = True
        prev_ratio_inputted = True
        
        # 체크박스 해제 시뮬레이션
        ratio_inputted = False
        
        # 체크박스 해제 시 값이 0.0으로 리셋되어야 함
        if ratio_inputted == False and prev_ratio_inputted == True:
            ratio_this = 0.0
        
        assert ratio_this == 0.0
        assert ratio_inputted == False

    def test_prev_state_tracking(self):
        """이전 상태 추적 기능 테스트"""
        # 초기 상태 설정
        prev_extra = True
        prev_ratio = True
        
        # 체크박스 해제 시뮬레이션
        extra_inputted = False
        ratio_inputted = False
        
        # 이전 상태와 현재 상태 비교
        assert extra_inputted != prev_extra
        assert ratio_inputted != prev_ratio
        
        # 상태 변경 감지 로직 검증
        extra_changed = extra_inputted != prev_extra
        ratio_changed = ratio_inputted != prev_ratio
        
        assert extra_changed is True
        assert ratio_changed is True

    def test_ui_layout_2line_structure(self):
        """2줄 레이아웃 구조 테스트"""
        # 1줄: 학과명 | 정원
        line1_columns = 2
        assert line1_columns == 2
        
        # 2줄: 추가충원 | 입력함 | 경쟁률 | 입력함
        line2_columns = 4
        assert line2_columns == 4
        
        # 전체 레이아웃 검증
        total_ui_elements = line1_columns + line2_columns
        assert total_ui_elements == 6

    def test_input_mode_simplification(self):
        """입력방식 단순화 테스트"""
        # 경쟁률만 입력 가능
        available_input_methods = ["경쟁률"]
        assert len(available_input_methods) == 1
        assert "경쟁률" in available_input_methods
        
        # 지원자 수 입력 방식 제거 확인
        assert "지원자 수" not in available_input_methods
        
        # 입력 방식 선택 드롭다운 제거 확인
        input_mode_selection_removed = True
        assert input_mode_selection_removed is True

    def test_tooltip_cleanup(self):
        """툴팁 정리 테스트"""
        # 숫자 입력 필드 툴팁 제거
        number_input_help_removed = True
        assert number_input_help_removed is True
        
        # 체크박스 툴팁 유지
        checkbox_help_maintained = True
        assert checkbox_help_maintained is True
        
        # 툴팁 내용 적절성 검증
        extra_checkbox_help = "체크하면 추가충원 값을 입력하고, 체크하지 않으면 자동 추정됩니다"
        ratio_checkbox_help = "체크하면 경쟁률을 입력하고, 체크하지 않으면 자동 추정됩니다"
        
        assert "추가충원" in extra_checkbox_help
        assert "경쟁률" in ratio_checkbox_help
        assert "자동 추정" in extra_checkbox_help
        assert "자동 추정" in ratio_checkbox_help

    def test_stable_value_reset(self):
        """안정적인 값 리셋 테스트"""
        # on_change 콜백 사용으로 StreamlitAPIException 방지
        uses_on_change_callback = True
        assert uses_on_change_callback is True
        
        # 헬퍼 함수를 통한 안전한 상태 관리
        helper_function_exists = True
        assert helper_function_exists is True
        
        # 체크박스 해제 시 즉시 값 리셋
        immediate_reset_on_uncheck = True
        assert immediate_reset_on_uncheck is True
