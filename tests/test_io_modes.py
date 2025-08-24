import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock

class TestIOModes:
    """JSON 저장/불러오기와 입력모드 복원 로직 테스트"""
    
    def test_json_save_structure(self):
        """JSON 저장 구조 검증"""
        # 저장할 데이터 구조
        save_data = {
            "department_name": "컴퓨터공학과",
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
        
        # JSON 문자열로 변환
        json_str = json.dumps(save_data, ensure_ascii=False, indent=2)
        
        # 다시 파싱하여 구조 검증
        parsed_data = json.loads(json_str)
        
        # 필수 키들이 존재하는지 확인
        assert "department_name" in parsed_data
        assert "current_year_inputs" in parsed_data
        assert "historical_data" in parsed_data
        
        # current_year_inputs의 필수 필드 확인
        current_inputs = parsed_data["current_year_inputs"]
        required_fields = ["capacity", "extra", "applicants", "ratio"]
        for field in required_fields:
            assert field in current_inputs
            
        # historical_data가 리스트인지 확인
        assert isinstance(parsed_data["historical_data"], list)
        
    def test_input_mode_detection(self):
        """입력모드 감지 로직 검증"""
        # 지원자 수 모드: applicants > 0 & ratio == 0
        applicants_mode = {
            "applicants": 150,
            "ratio": 0
        }
        
        # 경쟁률 모드: ratio > 0
        ratio_mode = {
            "applicants": 0,
            "ratio": 5.0
        }
        
        # 입력모드 판별
        def detect_input_mode(data):
            if data["applicants"] > 0 and data["ratio"] == 0:
                return "지원자 수 모드"
            elif data["ratio"] > 0:
                return "경쟁률 모드"
            else:
                return "미정"
        
        assert detect_input_mode(applicants_mode) == "지원자 수 모드"
        assert detect_input_mode(ratio_mode) == "경쟁률 모드"
        
    def test_input_mode_restoration(self):
        """입력모드 복원 로직 검증"""
        # 저장된 데이터
        saved_data = {
            "current_year_inputs": {
                "capacity": 30,
                "extra": 5,
                "applicants": 150,
                "ratio": 0  # 저장 시 0으로 저장됨
            }
        }
        
        # 불러오기 시 입력모드 복원
        current_inputs = saved_data["current_year_inputs"]
        
        # applicants > 0 & ratio == 0이면 "지원자 수 모드"
        if current_inputs["applicants"] > 0 and current_inputs["ratio"] == 0:
            restored_mode = "지원자 수 모드"
            # ratio를 계산하여 복원
            restored_ratio = current_inputs["applicants"] / current_inputs["capacity"]
        else:
            restored_mode = "경쟁률 모드"
            restored_ratio = current_inputs["ratio"]
            
        # 검증
        assert restored_mode == "지원자 수 모드"
        expected_ratio = 150 / 30  # 5.0
        assert restored_ratio == expected_ratio
        
    def test_column_order_preservation(self):
        """컬럼 순서 보존 검증"""
        # COLUMN_ORDER 정의
        COLUMN_ORDER = [
            "year", "capacity", "extra", "final_cut", 
            "mean_score", "max_score", "applicants", "ratio"
        ]
        
        # historical_data의 각 행이 올바른 순서를 가지는지 확인
        historical_row = {
            "year": 2023,
            "capacity": 30,
            "extra": 3,
            "final_cut": 2.5,
            "mean_score": 3.2,
            "max_score": 4.1,
            "applicants": 120,
            "ratio": 4.0
        }
        
        # DataFrame으로 변환하여 순서 확인
        df = pd.DataFrame([historical_row])
        
        # 컬럼 순서가 COLUMN_ORDER와 일치하는지 확인
        for i, col in enumerate(COLUMN_ORDER):
            if col in df.columns:
                assert df.columns[i] == col
                
    def test_data_validation(self):
        """데이터 유효성 검증"""
        # 유효한 데이터
        valid_data = {
            "capacity": 30,
            "extra": 5,
            "final_cut": 2.5,
            "mean_score": 3.2,
            "max_score": 4.1
        }
        
        # 유효성 검사
        def validate_data(data):
            errors = []
            
            # capacity는 양의 정수
            if not isinstance(data["capacity"], int) or data["capacity"] <= 0:
                errors.append("capacity는 양의 정수여야 합니다")
                
            # extra는 음이 아닌 정수
            if not isinstance(data["extra"], int) or data["extra"] < 0:
                errors.append("extra는 음이 아닌 정수여야 합니다")
                
            # 등급은 [1.0, 9.0] 범위
            grade_fields = ["final_cut", "mean_score", "max_score"]
            for field in grade_fields:
                if field in data:
                    grade = data[field]
                    if not (1.0 <= grade <= 9.0):
                        errors.append(f"{field}는 1.0~9.0 범위여야 합니다")
                        
            return errors
        
        errors = validate_data(valid_data)
        assert len(errors) == 0
        
        # 잘못된 데이터 테스트
        invalid_data = {
            "capacity": -5,  # 음수
            "extra": 3,
            "final_cut": 10.0,  # 범위 초과
            "mean_score": 3.2,
            "max_score": 4.1
        }
        
        errors = validate_data(invalid_data)
        assert len(errors) > 0
        assert "capacity는 양의 정수여야 합니다" in errors
        assert "final_cut는 1.0~9.0 범위여야 합니다" in errors
        
    def test_json_roundtrip(self):
        """JSON 저장-불러오기 라운드트립 검증"""
        # 원본 데이터
        original_data = {
            "department_name": "전자공학과",
            "current_year_inputs": {
                "capacity": 25,
                "extra": 2,
                "applicants": 100,
                "ratio": 4.0
            },
            "historical_data": [
                {
                    "year": 2023,
                    "capacity": 25,
                    "extra": 2,
                    "final_cut": 2.8,
                    "mean_score": 3.4,
                    "max_score": 4.2,
                    "applicants": 100,
                    "ratio": 4.0
                }
            ]
        }
        
        # JSON으로 저장
        json_str = json.dumps(original_data, ensure_ascii=False, indent=2)
        
        # 파일에 저장 (시뮬레이션)
        with open("test_data.json", "w", encoding="utf-8") as f:
            f.write(json_str)
            
        # 파일에서 불러오기
        with open("test_data.json", "r", encoding="utf-8") as f:
            loaded_json = f.read()
            
        # JSON 파싱
        loaded_data = json.loads(loaded_json)
        
        # 데이터 일치성 검증
        assert loaded_data["department_name"] == original_data["department_name"]
        assert loaded_data["current_year_inputs"]["capacity"] == original_data["current_year_inputs"]["capacity"]
        assert len(loaded_data["historical_data"]) == len(original_data["historical_data"])
        
        # 파일 정리
        import os
        if os.path.exists("test_data.json"):
            os.remove("test_data.json")
            
    def test_edge_cases(self):
        """극단 케이스 처리 검증"""
        # 빈 historical_data
        empty_data = {
            "department_name": "신설학과",
            "current_year_inputs": {
                "capacity": 20,
                "extra": 0,
                "applicants": 0,
                "ratio": 0
            },
            "historical_data": []
        }
        
        # 빈 데이터 처리
        assert len(empty_data["historical_data"]) == 0
        
        # ratio가 0인 경우
        zero_ratio_data = {
            "capacity": 30,
            "extra": 0,
            "applicants": 0,
            "ratio": 0
        }
        
        # M = ratio * capacity = 0
        M = zero_ratio_data["ratio"] * zero_ratio_data["capacity"]
        assert M == 0
        
        # N = capacity + extra = 30
        N = zero_ratio_data["capacity"] + zero_ratio_data["extra"]
        assert N == 30
        
        # M < N인 경우 (정원 미달)
        if M < N:
            assert M < N  # 0 < 30
