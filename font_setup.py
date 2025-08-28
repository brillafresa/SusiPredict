# font_setup.py
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

def _register_local_fonts(font_paths: List[Path]) -> list:
    """동봉 폰트를 addfont로 등록하고, 등록된 패밀리명을 반환"""
    families = []
    for p in font_paths:
        if p.is_file():
            try:
                font_manager.addfont(str(p))
                fam = font_manager.FontProperties(fname=str(p)).get_name()
                if fam and fam not in families:
                    families.append(fam)
            except Exception:
                # 문제가 있어도 무시하고 다음 후보 진행
                pass
    return families

def setup_korean_fonts(prefer_local_first: bool = True) -> None:
    """
    Matplotlib 한글 폰트 깨짐 방지 공통 설정.
    1) 레포 동봉 폰트(Noto/Nanum)를 addfont로 등록
    2) OS별 시스템 후보를 준비 (Windows=Malgun Gothic, macOS=AppleGothic, Linux=Noto/Nanum 등)
    3) font.sans-serif 우선순위 체인을 일괄 지정
    """
    base_dir = Path(__file__).resolve().parent
    fonts_dir = base_dir / "fonts"

    # 1) 레포 동봉 폰트 등록 (배포 자유 폰트만)
    local_font_paths = [
        fonts_dir / "NanumGothic.ttf",
        # 필요시 추가 폰트 파일들
        # fonts_dir / "NotoSansCJKkr-Regular.otf",
        # fonts_dir / "NanumGothicBold.ttf",
    ]
    local_families = _register_local_fonts(local_font_paths)

    # 2) OS별 시스템 후보
    if sys.platform.startswith("win"):
        system_families = ["Malgun Gothic", "Malgun Gothic Semilight"]
    elif sys.platform == "darwin":
        system_families = ["AppleGothic"]
    else:
        # Linux(Streamlit Cloud). packages.txt로 설치했다면 유효
        system_families = [
            "NanumGothic",       # fonts-nanum
            "Noto Sans CJK KR",  # fonts-noto-cjk
            "Noto Serif CJK KR",
            "DejaVu Sans"        # 최후 보루
        ]

    # 3) 우선순위 체인 구성
    chain = (local_families + system_families) if prefer_local_first else (system_families + local_families)

    # 중복 제거
    seen = set()
    ordered_chain = []
    for name in chain:
        if name and name not in seen:
            seen.add(name)
            ordered_chain.append(name)

    # 4) Matplotlib에 반영
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ordered_chain if ordered_chain else ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # (선택) 캐시 강제 리빌드가 필요하면 아래 주석 해제
    # font_manager._load_fontmanager(try_read_cache=False)
