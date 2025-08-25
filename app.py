# -*- coding: utf-8 -*-
# app.py
# 🎓 수시 합격 가능성 예측기 (Streamlit)
# v6.2.0 — 디버그모드 복원(중간계산 표시) + 로버스트 앵커 + 겹침표시 + 확률기준 도움말

import numpy as np
import pandas as pd
import math
import json
from typing import Dict, List, Tuple, Optional, NamedTuple

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import optimize, stats
from scipy.special import betainc

import streamlit as st

# ===================== 페이지/스타일 설정 =====================
st.set_page_config(page_title="수시 합격 가능성 예측기", layout="wide")

GRADE_BOUNDS = (1.0, 9.0)

# 한글 폰트 (간단 우선순위)
for fam in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "sans-serif"]:
    try:
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

# 기본 가중치
BASE_WEIGHTS = {
    "final_cut": 10.0, "best": 5.0, "median": 5.0, "mean": 4.0, "p70": 3.0,
    "init_mean": 2.0, "init_best": 2.0, "init_worst": 2.0,
    "app_mean": 0.0, "app_best": 0.0, "app_worst": 0.0,
}

# ---- 컬럼/표시명 정의 ----
COLUMN_CONFIG = {
    "year": "입시학년도",
    "best": "최종등록자 최고",
    "median": "최종등록자 50%",
    "p70": "최종등록자 70%",
    "final_cut": "최종등록자 최저(컷)",
    "mean": "최종등록자 평균",
    "init_best": "최초합격자 최고",
    "init_worst": "최초합격자 최저",
    "init_mean": "최초합격자 평균",
    "app_best": "지원자 최고",
    "app_worst": "지원자 최저",
    "app_mean": "지원자 평균",
    "capacity": "정원",
    "extra": "추가충원",
    "competition_ratio": "경쟁률",
}
COLUMN_ORDER = [
    "year", "best", "median", "p70", "final_cut", "mean",
    "init_best", "init_worst", "init_mean",
    "app_best", "app_worst", "app_mean",
    "capacity", "extra", "competition_ratio"
]
REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_CONFIG.items()}

# ===================== 수학 유틸/구조 =====================
class QuantileTerm(NamedTuple):
    kind: str
    p: float
    value: float

class MeanTerm(NamedTuple):
    kind: str
    value: float
    q_trunc: Optional[float]

def _clip(x, lo, hi): return np.clip(x, lo, hi)
def _safe_ppf_p(p): return _clip(p, 1e-6, 1 - 1e-6)
def _beta_ppf_std(p, a, b): return stats.beta.ppf(_safe_ppf_p(p), a, b)
def _beta_quantile(p, a, b, lo, hi): return _beta_ppf_std(p, a, b) * (hi - lo) + lo
def _beta_cdf(x, a, b, lo, hi): return stats.beta.cdf(_clip((x - lo) / (hi - lo), 0.0, 1.0), a, b)
def _beta_pdf(x, a, b, lo, hi): return stats.beta.pdf(_clip((x - lo) / (hi - lo), 0.0, 1.0), a, b) / (hi - lo)

def _beta_trunc_mean_std(a, b, z_trunc):
    denominator = betainc(a, b, z_trunc)
    if denominator <= 1e-12:
        return z_trunc * 0.5
    numerator = (a / (a + b)) * betainc(a + 1, b, z_trunc)
    return numerator / denominator

def _admitted_quantile_from_underlying(p_cond, a, b, lo, hi, q_sel):
    return _beta_quantile(p_cond * q_sel, a, b, lo, hi)

def _is_bad(a, b):
    return not (np.isfinite(a) and np.isfinite(b) and 0.2 < a < 100.0 and 0.2 < b < 100.0)

# 🆕 체크박스 해제 시 값을 리셋하는 헬퍼 함수
def _reset_value_on_uncheck(checkbox_key: str, value_key: str, default_value):
    """체크박스가 해제되었을 때 해당 값 필드를 기본값으로 리셋합니다."""
    if not st.session_state.get(checkbox_key, False):
        st.session_state[value_key] = default_value

def _solve_beta_two_quantiles(q1: QuantileTerm, q2: QuantileTerm, lo, hi, init_ab=(3.0, 6.0)):
    def objective(v):
        a, b = math.exp(v[0]), math.exp(v[1])
        e1 = _beta_quantile(q1.p, a, b, lo, hi) - q1.value
        e2 = _beta_quantile(q2.p, a, b, lo, hi) - q2.value
        return e1*e1 + e2*e2
    try:
        res = optimize.minimize(objective, np.log(np.array(init_ab)), method="Nelder-Mead")
        ab = np.exp(res.x)
        if not np.all(np.isfinite(ab)):
            return init_ab
        return float(ab[0]), float(ab[1])
    except Exception:
        return init_ab

def _get_stochastic_forecast(years, values, x_new, n_scenarios, min_val=0.0):
    valid_mask = np.isfinite(years) & np.isfinite(values)
    years, values = years[valid_mask], values[valid_mask]
    if len(values) < 4:
        return np.random.choice(values, n_scenarios, replace=True) if len(values) > 0 else np.array([])
    slope, intercept, _, _ = stats.theilslopes(values, years)
    forecast_center = intercept + slope * x_new
    predicted_values = intercept + slope * years
    residuals = values - predicted_values
    scenarios = forecast_center + np.random.choice(residuals, n_scenarios, replace=True)
    return np.maximum(min_val, scenarios)

# ===================== 전처리/피팅/투영 =====================
def prepare_years(df_input: List[Dict]) -> pd.DataFrame:
    """과거 데이터 가공 + 유효 경쟁률 보정(지원자 평균 기반) + q_select 계산
       경쟁률 정의: ratio = applicants / capacity
    """
    if not df_input:
        return pd.DataFrame()
    df_temp = pd.DataFrame(df_input)

    valid_app_means = df_temp['app_mean'].dropna()
    baseline_app_mean = valid_app_means.mean() if len(valid_app_means) >= 2 else None

    rows = []
    for r in df_input:
        cap = r.get("capacity", None)
        ext = r.get("extra", None)
        cap = int(cap) if cap is not None else None
        ext = int(ext) if ext is not None else 0
        total = (cap or 0) + (ext or 0)  # 최종 선발 인원 N

        ratio = r.get("competition_ratio", None)  # ratio = applicants / capacity
        applicants = r.get("applicants", None)
        if ratio is None and applicants is not None and cap:
            try:
                ratio = float(applicants) / float(cap) if cap > 0 else None
            except Exception:
                ratio = None

        effective_ratio = ratio
        if (ratio is not None) and (baseline_app_mean is not None) and pd.notnull(r.get("app_mean")):
            deviation = (r["app_mean"] - baseline_app_mean) / baseline_app_mean
            effective_ratio = ratio * (1 + np.clip(deviation, -0.3, 0.3))

        M = None
        if effective_ratio is not None and cap:
            M = float(effective_ratio) * float(cap)

        q_sel = min(1.0, float(total) / float(M)) if (M and M > 0) else None

        row_data = r.copy()
        row_data.update({
            "total_seats": total if total > 0 else None,      # N
            "competition_ratio": ratio,                        # ratio = applicants / capacity
            "effective_competition_ratio": effective_ratio,    # 보정 경쟁률
            "q_select": q_sel,                                 # N/M
        })
        rows.append(row_data)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

def _objective_function(params, quantile_terms, mean_terms, lo, hi, weights):
    a, b = math.exp(params[0]), math.exp(params[1])
    loss = 0.0
    for term in quantile_terms:
        pred = _beta_quantile(term.p, a, b, lo, hi)
        w = weights.get(term.kind, 1.0)
        if w > 0:
            loss += w * (pred - term.value)**2
    for term in mean_terms:
        w = weights.get(term.kind, 1.0)
        if w == 0:
            continue
        if term.q_trunc is None:
            pred = lo + (hi - lo) * (a / (a + b))
        else:
            z_trunc = _beta_ppf_std(term.q_trunc, a, b)
            mean_std = _beta_trunc_mean_std(a, b, z_trunc)
            pred = lo + (hi - lo) * mean_std
        loss += w * (pred - term.value)**2
    return loss

def _fit_underlying_from_competitive(row, lo, hi, weights):
    """경쟁률/충원 정보가 있는 연도: 전체 지원자분포(Beta) 역추정"""
    q_sel = row["q_select"]
    total, cap = row["total_seats"], row["capacity"]  # total=N, cap=정원
    eff_ratio = row.get("effective_competition_ratio", None)

    M = int(round(eff_ratio * cap)) if (eff_ratio is not None and cap) else None
    q_init = (cap / M) if (cap and M and M > 0) else None

    q_terms: List[QuantileTerm] = []
    m_terms: List[MeanTerm] = []

    if pd.notnull(row.get("median")):
        q_terms.append(QuantileTerm("median", 0.5 * q_sel, row["median"]))
    if pd.notnull(row.get("p70")):
        q_terms.append(QuantileTerm("p70", 0.7 * q_sel, row["p70"]))
    if pd.notnull(row.get("final_cut")):
        q_terms.append(QuantileTerm("final_cut", 1.0 * q_sel, row["final_cut"]))
    if pd.notnull(row.get("best")) and total:
        q_terms.append(QuantileTerm("best", (1/(total+1)) * q_sel, row["best"]))
    if pd.notnull(row.get("mean")):
        m_terms.append(MeanTerm("mean", row["mean"], q_sel))

    if q_init and cap:
        if pd.notnull(row.get("init_best")):
            q_terms.append(QuantileTerm("init_best", (1/(cap+1)) * q_init, row["init_best"]))
        if pd.notnull(row.get("init_worst")):
            q_terms.append(QuantileTerm("init_worst", (cap/(cap+1)) * q_init, row["init_worst"]))
        if pd.notnull(row.get("init_mean")):
            m_terms.append(MeanTerm("init_mean", row["init_mean"], q_init))

    if M:
        if pd.notnull(row.get("app_best")):
            q_terms.append(QuantileTerm("app_best", 1/(M+1), row["app_best"]))
        if pd.notnull(row.get("app_worst")):
            q_terms.append(QuantileTerm("app_worst", M/(M+1), row["app_worst"]))
        if pd.notnull(row.get("app_mean")):
            m_terms.append(MeanTerm("app_mean", row["app_mean"], None))

    if len(q_terms) < 2:
        return 3.0, 6.0, np.nan, {"M": M, "q_init": q_init}

    q_terms.sort(key=lambda t: t.p)
    init_a, init_b = _solve_beta_two_quantiles(q_terms[0], q_terms[-1], lo, hi)

    res = optimize.minimize(
        _objective_function,
        np.log([init_a, init_b]),
        args=(q_terms, m_terms, lo, hi, weights),
        method="Nelder-Mead",
        options={"maxiter": 2500}
    )
    a, b = np.exp(res.x)
    if (not res.success) or _is_bad(a, b):
        return 3.0, 6.0, res.fun if hasattr(res, "fun") else np.nan, {"M": M, "q_init": q_init}
    return float(a), float(b), float(res.fun), {"M": M, "q_init": q_init}

def _fit_admitted_without_ratio(row, lo, hi, weights):
    """경쟁률 정보가 없는 연도: 합격자 분포로 가정하여 피팅(근사)"""
    N, cap = row["total_seats"], row["capacity"]
    q_terms: List[QuantileTerm] = []
    m_terms: List[MeanTerm] = []

    if pd.notnull(row.get("median")):
        q_terms.append(QuantileTerm("median", 0.5, row["median"]))
    if pd.notnull(row.get("p70")):
        q_terms.append(QuantileTerm("p70", 0.7, row["p70"]))
    if pd.notnull(row.get("final_cut")) and N:
        q_terms.append(QuantileTerm("final_cut", N/(N+1), row["final_cut"]))
    if pd.notnull(row.get("best")) and N:
        q_terms.append(QuantileTerm("best", 1/(N+1), row["best"]))
    if pd.notnull(row.get("mean")):
        m_terms.append(MeanTerm("mean", row["mean"], None))

    if cap and N and N > 0:
        phi = cap / float(N)
        if pd.notnull(row.get("init_best")):
            q_terms.append(QuantileTerm("init_best", (1/(cap+1))*phi, row["init_best"]))
        if pd.notnull(row.get("init_worst")):
            q_terms.append(QuantileTerm("init_worst", (cap/(cap+1))*phi, row["init_worst"]))
        if pd.notnull(row.get("init_mean")):
            m_terms.append(MeanTerm("init_mean", row["init_mean"], phi))

    if len(q_terms) < 2:
        return 3.0, 6.0, np.nan, {}

    q_terms.sort(key=lambda t: t.p)
    init_a, init_b = _solve_beta_two_quantiles(q_terms[0], q_terms[-1], lo, hi)

    res = optimize.minimize(
        _objective_function,
        np.log([init_a, init_b]),
        args=(q_terms, m_terms, lo, hi, weights),
        method="Nelder-Mead",
        options={"maxiter": 2500}
    )
    a, b = np.exp(res.x)
    if (not res.success) or _is_bad(a, b):
        return 3.0, 6.0, res.fun if hasattr(res, "fun") else np.nan, {}
    return float(a), float(b), float(res.fun), {}

def fit_per_year_models(df, lo, hi, weights, anchor_tol: float = 0.75):
    """연도별 (a,b) 피팅 + 주요 지표의 적합치 계산 + 로버스트 앵커 생성"""
    def choose_anchor(input_val, fitted_val, tol=anchor_tol):
        iv_ok = (input_val is not None) and pd.notnull(input_val)
        fv_ok = (fitted_val is not None) and pd.notnull(fitted_val)
        if iv_ok and fv_ok:
            return float(input_val) if abs(float(fitted_val) - float(input_val)) > tol else float(fitted_val)
        elif iv_ok:
            return float(input_val)
        elif fv_ok:
            return float(fitted_val)
        else:
            return np.nan

    cols = [
        "year","beta_a","beta_b","model_type","fit_loss","q_select","total_seats","capacity",
        "fitted_median","fitted_p70","fitted_final","fitted_best","fitted_final_mean",
        "fitted_init_best","fitted_init_worst","fitted_init_mean",
        "fitted_app_best","fitted_app_worst","fitted_app_mean",
        # robust anchors
        "anchor_median","anchor_p70","anchor_final","anchor_best",
        "anchor_init_best","anchor_init_worst","anchor_final_mean","anchor_init_mean","anchor_app_mean"
    ]
    outs = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if pd.notnull(row["q_select"]):
            a, b, loss, meta = _fit_underlying_from_competitive(row_dict, lo, hi, weights)
            q_sel = float(row["q_select"])
            total = int(row["total_seats"]) if row["total_seats"] else None
            cap = int(row["capacity"]) if row["capacity"] else None
            M = meta.get("M", None)
            q_init = meta.get("q_init", None)

            med = _admitted_quantile_from_underlying(0.5, a, b, lo, hi, q_sel)
            p70 = _admitted_quantile_from_underlying(0.7, a, b, lo, hi, q_sel)
            fin = _admitted_quantile_from_underlying(1.0, a, b, lo, hi, q_sel)
            bst = _admitted_quantile_from_underlying(1.0/(total+1.0), a, b, lo, hi, q_sel) if total else np.nan

            z = _beta_ppf_std(q_sel, a, b)
            fm = lo + (hi - lo) * _beta_trunc_mean_std(a, b, z)

            if q_init is not None and cap:
                init_b = _admitted_quantile_from_underlying(1.0/(cap+1.0), a, b, lo, hi, q_init)
                init_w = _admitted_quantile_from_underlying(cap/(cap+1.0), a, b, lo, hi, q_init)
                z0 = _beta_ppf_std(q_init, a, b)
                im = lo + (hi - lo) * _beta_trunc_mean_std(a, b, z0)
            else:
                init_b = init_w = im = np.nan

            if M:
                app_b = _beta_quantile(1.0/(M+1.0), a, b, lo, hi)
                app_w = _beta_quantile(M/(M+1.0), a, b, lo, hi)
                app_m = lo + (hi - lo) * (a/(a+b))
            else:
                app_b = app_w = app_m = np.nan

            # anchors (robust)
            anc_median = choose_anchor(row.get("median"), med)
            anc_p70    = choose_anchor(row.get("p70"), p70)
            anc_final  = choose_anchor(row.get("final_cut"), fin)
            anc_best   = choose_anchor(row.get("best"), bst)
            anc_ibest  = choose_anchor(row.get("init_best"), init_b)
            anc_iworst = choose_anchor(row.get("init_worst"), init_w)
            anc_fmean  = fm  # 입력치 없음 → fitted 사용
            anc_imean  = choose_anchor(row.get("init_mean"), im)
            anc_amean  = choose_anchor(row.get("app_mean"), app_m)

            outs.append([
                row["year"], a, b, "underlying+trunc", loss, q_sel,
                row["total_seats"], row["capacity"],
                med, p70, fin, bst, fm,
                init_b, init_w, im,
                app_b, app_w, app_m,
                anc_median, anc_p70, anc_final, anc_best,
                anc_ibest, anc_iworst, anc_fmean, anc_imean, anc_amean
            ])
        else:
            a, b, loss, _ = _fit_admitted_without_ratio(row_dict, lo, hi, weights)
            N = int(row["total_seats"]) if row["total_seats"] else None
            cap = int(row["capacity"]) if row["capacity"] else None

            med = _beta_quantile(0.5, a, b, lo, hi)
            p70 = _beta_quantile(0.7, a, b, lo, hi)
            fin = _beta_quantile(N/(N+1.0), a, b, lo, hi) if N else np.nan
            bst = _beta_quantile(1.0/(N+1.0), a, b, lo, hi) if N else np.nan
            fm  = lo + (hi - lo) * (a/(a+b))

            if cap and N and N > 0:
                phi = cap / float(N)
                init_b = _beta_quantile((1.0/(cap+1.0)) * phi, a, b, lo, hi)
                init_w = _beta_quantile((cap/(cap+1.0)) * phi, a, b, lo, hi)
                z_phi = _beta_ppf_std(phi, a, b)
                im = lo + (hi - lo) * _beta_trunc_mean_std(a, b, z_phi)
            else:
                init_b = init_w = im = np.nan

            # anchors (robust)
            anc_median = choose_anchor(row.get("median"), med)
            anc_p70    = choose_anchor(row.get("p70"), p70)
            anc_final  = choose_anchor(row.get("final_cut"), fin)
            anc_best   = choose_anchor(row.get("best"), bst)
            anc_ibest  = choose_anchor(row.get("init_best"), init_b)
            anc_iworst = choose_anchor(row.get("init_worst"), init_w)
            anc_fmean  = fm
            anc_imean  = choose_anchor(row.get("init_mean"), im)
            anc_amean  = choose_anchor(row.get("app_mean"), np.nan)

            outs.append([
                row["year"], a, b, "admitted-only", loss, np.nan,
                row["total_seats"], row["capacity"],
                med, p70, fin, bst, fm,
                init_b, init_w, im,
                np.nan, np.nan, np.nan,
                anc_median, anc_p70, anc_final, anc_best,
                anc_ibest, anc_iworst, anc_fmean, anc_imean, anc_amean
            ])

    return pd.DataFrame(outs, columns=cols).sort_values("year").reset_index(drop=True)

# ---- 최근연도 가중 회귀로 1년 후 지표 앵커 예측 ----
def _recency_weighted_forecast(years: np.ndarray, values: np.ndarray, x_new: float,
                               half_life: float = 1.0, mix: float = 0.65) -> float:
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values[-1])

    y_last = float(values[-1])
    t_last = float(years[-1])

    w = 0.5 ** ((t_last - years) / float(half_life))
    w = np.clip(w, 1e-6, 1.0)

    S_w  = np.sum(w)
    S_x  = np.sum(w * years)
    S_y  = np.sum(w * values)
    S_xx = np.sum(w * years * years)
    S_xy = np.sum(w * years * values)
    denom = (S_w * S_xx - S_x * S_x)
    if abs(denom) < 1e-12:
        y_hat = y_last
    else:
        a = (S_w * S_xy - S_x * S_y) / denom
        b = (S_y - a * S_x) / S_w
        y_hat = a * x_new + b

    return float(mix * y_hat + (1.0 - mix) * y_last)

def project_current_year(
    df, df_fit, lo, hi,
    capacity_this_year, extra_this_year, ratio_this_year, weights
):
    """올해 지원자 분포(Beta) 산출: ratio는 정원(capacity) 기준"""
    years = df_fit["year"].values.astype(float)
    current_year = years.max() + 1.0 if len(years) > 0 else 2026

    cap = int(capacity_this_year or 0)
    if cap <= 0:
        raise ValueError("올해 정원은 1 이상이어야 합니다.")
    total_seats_base = cap + (int(extra_this_year) if extra_this_year is not None else 0)

    # ratio = applicants / capacity (경쟁률만 사용)
    ratio = ratio_this_year if ratio_this_year and ratio_this_year > 0 else None

    ratio_for_projection = ratio
    if ratio_for_projection is None:
        historical = df["competition_ratio"].dropna().values
        ratio_for_projection = np.median(historical) if len(historical) > 0 else None

    M = int(round(ratio_for_projection * cap)) if (ratio_for_projection and cap > 0) else None
    q_sel = (min(1.0, total_seats_base / M) if (M and M > 0) else None)
    q_init = (cap / float(M)) if (cap and M and M > 0) else None

    # --- 앵커는 'robust anchor' 열을 사용 ---
    def _ts(name):
        s = df_fit[[name, "year"]].dropna()
        if s.empty:
            return None
        return _recency_weighted_forecast(
            s["year"].values.astype(float),
            s[name].values.astype(float),
            current_year,
            half_life=1.0,   # 최근 1년 반감
            mix=0.65         # 회귀예측 65% + 직전값 35%
        )

    q_terms: List[QuantileTerm] = []
    m_terms: List[MeanTerm] = []
    anchors_map = {
        "median": ("anchor_median", 0.5),
        "p70": ("anchor_p70", 0.7),
        "final_cut": ("anchor_final", 1.0),
        "best": ("anchor_best", 1.0/(total_seats_base+1)),
        "init_best": ("anchor_init_best", 1.0/(cap+1) if cap else 0),
        "init_worst": ("anchor_init_worst", cap/(cap+1) if cap else 1),
    }
    for kind, (col, p_cond) in anchors_map.items():
        v = _ts(col)
        if v is None or p_cond == 0:
            continue
        if kind.startswith("init") and q_init:
            q_terms.append(QuantileTerm(kind, p_cond * q_init, v))
        else:
            p = p_cond * q_sel if q_sel else p_cond
            q_terms.append(QuantileTerm(kind, p, v))

    if (v := _ts("anchor_final_mean")) is not None:
        m_terms.append(MeanTerm("mean", v, q_sel))
    if (v := _ts("anchor_app_mean")) is not None and M:
        m_terms.append(MeanTerm("app_mean", v, None))
    if (v := _ts("anchor_init_mean")) is not None and q_init:
        m_terms.append(MeanTerm("init_mean", v, q_init))

    if len(q_terms) < 2:
        a0, b0 = 3.0, 6.0
    else:
        q_terms.sort(key=lambda x: x.p)
        a0, b0 = _solve_beta_two_quantiles(q_terms[0], q_terms[-1], lo, hi)

    res = optimize.minimize(
        _objective_function,
        np.log([a0, b0]),
        args=(q_terms, m_terms, lo, hi, weights),
        method="Nelder-Mead",
        options={"maxiter": 2500}
    )
    a, b = np.exp(res.x)
    if (not res.success) or _is_bad(a, b):
        a, b = 3.0, 6.0

    return {
        "a": float(a), "b": float(b),
        "model_type": "underlying+trunc" if q_sel else "admitted-only",
        "q_select_this": q_sel,
        "total_seats_this": total_seats_base,  # N
        "capacity_this": cap,                   # 정원
        "ratio_this": ratio,                    # ratio = applicants/capacity
        "M_this": M,                            # 지원자수
        "q_init_this": q_init
    }

# ===================== 시뮬레이션 =====================
def simulate_acceptance(
    proj, df, applicant_grade, lo, hi,
    capacity_this_year: int,
    extra_this_year: Optional[int],
    ratio_this_year: Optional[float],
    n_scenarios=5000
):
    a, b = proj["a"], proj["b"]
    is_ratio_known = ratio_this_year is not None and ratio_this_year > 0.0
    is_extra_known = extra_this_year is not None

    years = df["year"].values.astype(float)
    current_year = years.max() + 1.0 if len(years) > 0 else 2026

    extra_ratios = np.array([])

    if is_ratio_known:
        ratios = np.full(n_scenarios, float(ratio_this_year))
    else:
        ratios = _get_stochastic_forecast(
            years, df["competition_ratio"].values, current_year, n_scenarios, min_val=0.5
        )

    if is_extra_known:
        extras = np.full(n_scenarios, int(extra_this_year))
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            cap_series = pd.to_numeric(df["capacity"], errors="coerce").astype(float)
            extra_series = pd.to_numeric(df["extra"], errors="coerce").astype(float)
            extra_ratio_series = np.where(
                (cap_series > 0) & np.isfinite(cap_series),
                np.nan_to_num(extra_series / cap_series, nan=0.0),
                np.nan
            )
        extra_ratios = _get_stochastic_forecast(
            years, extra_ratio_series, current_year, n_scenarios, min_val=0.0
        )
        extras = np.round(np.nan_to_num(extra_ratios, nan=0.0) * capacity_this_year).astype(int)

    N_scenarios = capacity_this_year + extras                        # N
    M_scenarios = np.round(ratios * capacity_this_year).astype(int)  # M = ratio * capacity

    valid_mask = (N_scenarios > 0) & (M_scenarios >= N_scenarios)
    if not np.any(valid_mask):
        Ts = _beta_quantile(0.999, a, b, lo, hi) * np.ones(n_scenarios)
        accepted = (Ts >= applicant_grade)
        k, n = int(np.sum(accepted)), int(len(accepted))
        from scipy.stats import beta as _beta
        p_accept = k / n if n > 0 else np.nan
        p_accept_ci = tuple(_beta.ppf([0.05, 0.95], k + 0.5, (n - k) + 0.5)) if n > 0 else (np.nan, np.nan)
        return {"mode":"underfilled","Ts":Ts,"p_accept":p_accept,"p_accept_ci":p_accept_ci,
                "p_pct":np.nan,"p_pct_ci":(np.nan,np.nan)}

    N_scenarios = N_scenarios[valid_mask]
    M_scenarios = M_scenarios[valid_mask]

    U_samples = stats.beta.rvs(N_scenarios, M_scenarios - N_scenarios + 1)
    Ts = _beta_quantile(U_samples, a, b, lo, hi)

    accepted_mask = (Ts >= applicant_grade)
    k, n = int(np.sum(accepted_mask)), int(len(accepted_mask))
    from scipy.stats import beta as _beta
    p_accept = k / n if n > 0 else np.nan
    p_accept_ci = tuple(_beta.ppf([0.05, 0.95], k + 0.5, (n - k) + 0.5)) if n > 0 else (np.nan, np.nan)

    Fg = _beta_cdf(applicant_grade, a, b, lo, hi)
    if k > 0:
        ranks = np.clip(Fg / np.maximum(U_samples[accepted_mask], 1e-9), 0.0, 1.0)
        p_pct = float(np.mean(ranks))
        p_pct_ci = tuple(np.quantile(ranks, [0.05, 0.95]))
    else:
        p_pct, p_pct_ci = np.nan, (np.nan, np.nan)

    out = {
        "mode": "probabilistic",
        "Ts": Ts[np.isfinite(Ts)],
        "p_accept": p_accept,
        "p_accept_ci": p_accept_ci,
        "p_pct": p_pct,
        "p_pct_ci": p_pct_ci,
    }
    if ratio_this_year is None and len(ratios) > 0:
        out["forecasted_ratio_median"] = float(np.median(ratios))
        out["forecasted_ratio_ci"] = tuple(np.quantile(ratios, [0.05, 0.95]))
    if extra_this_year is None and len(extra_ratios) > 0:
        out["forecasted_extra_rate_median"] = float(np.median(extra_ratios))
        out["forecasted_extra_rate_ci"] = tuple(np.quantile(extra_ratios, [0.05, 0.95]))
    return out

# ===================== 파이프라인 실행 & 시각화 =====================
def run_pipeline(
    years_data, department_name, applicant_grade,
    grade_bounds=(1.0, 9.0),
    capacity_this_year=None, extra_this_year=None,
    ratio_this_year=None,
    n_scenarios=5000, weights=BASE_WEIGHTS, debug=False
):
    lo, hi = grade_bounds

    df = prepare_years(years_data)
    df_fit = fit_per_year_models(df, lo, hi, weights)
    proj = project_current_year(
        df, df_fit, lo, hi,
        capacity_this_year, extra_this_year,
        ratio_this_year, weights
    )
    sim = simulate_acceptance(
        proj, df, applicant_grade, lo, hi,
        capacity_this_year=int(capacity_this_year),
        extra_this_year=int(extra_this_year) if extra_this_year is not None else None,
        ratio_this_year=proj.get("ratio_this"),
        n_scenarios=n_scenarios
    )

    # ---------- 디버그 패널 ----------
    if debug:
        st.markdown("### 🧩 디버그 패널")

        # 전처리 핵심 지표
        dbg_pre = pd.DataFrame({
            "year": df["year"],
            "capacity": df["capacity"],
            "extra": df["extra"],
            "ratio(raw)": df["competition_ratio"],
            "ratio(effective)": df["effective_competition_ratio"],
            "total_seats(N)": df["total_seats"],
            "q_select(N/M)": df["q_select"],
            "M(=ratio*capacity)": np.where(
                pd.notnull(df["effective_competition_ratio"]) & pd.notnull(df["capacity"]),
                df["effective_competition_ratio"] * df["capacity"], np.nan
            ),
        })
        st.code("🧩 전처리 결과 (q_select · 모집/지원 규모)", language="text")
        st.dataframe(dbg_pre, use_container_width=True)

        # 피팅 결과(입력 vs 적합치 vs 앵커)
        cols_show = [
            "year","beta_a","beta_b","model_type","fit_loss","q_select",
            "fitted_best","fitted_final","fitted_median","fitted_p70","fitted_final_mean",
            "anchor_best","anchor_final","anchor_median","anchor_p70","anchor_final_mean"
        ]
        st.code("🧪 연도별 피팅 결과 (입력 vs 적합치 · 앵커)", language="text")
        st.dataframe(df_fit[cols_show], use_container_width=True)

        # 올해 투영 파라미터
        st.code("🧮 올해 최종 파라미터/규모", language="text")
        st.code(json.dumps({
            "beta_a": proj["a"], "beta_b": proj["b"],
            "q_select_this": proj["q_select_this"],
            "N(total_seats_this)": proj["total_seats_this"],
            "M_this(지원자수)": proj["M_this"],
            "ratio_this(입력/자동)": proj["ratio_this"],
        }, indent=2, ensure_ascii=False), language="json")

        # 컷 분포 요약/샘플
        if len(sim["Ts"]) >= 10:
            g80, g50, g20 = np.quantile(sim["Ts"], [0.20, 0.50, 0.80])
        else:
            a, b = proj["a"], proj["b"]
            N = max(1, proj["total_seats_this"])
            samples = stats.beta.rvs(a, b, size=(2000, N)) * (hi - lo) + lo
            Ts_synth = samples.max(axis=1)
            g80, g50, g20 = np.quantile(Ts_synth, [0.20, 0.50, 0.80])

        st.code("📊 시뮬레이션 요약(샘플/분위수)", language="text")
        st.code(json.dumps({
            "g80": float(g80), "g50": float(g50), "g20": float(g20)
        }, indent=2, ensure_ascii=False), language="json")
        sample_list = [float(x) for x in np.round(sim["Ts"][:10], 3)]
        st.code("최종컷 샘플 10개:", language="text")
        st.code(json.dumps(sample_list, ensure_ascii=False), language="json")

    # 최종컷 분포 분위수 (시각화용)
    if len(sim["Ts"]) >= 10:
        g80, g50, g20 = np.quantile(sim["Ts"], [0.20, 0.50, 0.80])
    else:
        a, b = proj["a"], proj["b"]
        N = max(1, proj["total_seats_this"])
        samples = stats.beta.rvs(a, b, size=(2000, N)) * (hi - lo) + lo
        Ts_synth = samples.max(axis=1)
        g80, g50, g20 = np.quantile(Ts_synth, [0.20, 0.50, 0.80])

    # Risk 버킷(합격확률 p 기준)
    p = sim['p_accept']
    if p >= 0.80: risk_bucket = "안정권"
    elif p >= 0.50: risk_bucket = "적정권"
    elif p >= 0.20: risk_bucket = "소신권"
    else: risk_bucket = "위험권"

    # --------- 좌: 입시학년도별 추세 ---------
    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6.6, 4.6))
        plot_map = {
            "50% (중위수)": ("median", "fitted_median"),
            "70%": ("p70", "fitted_p70"),
            "최종컷": ("final_cut", "fitted_final"),
            "최고점": ("best", "fitted_best"),
            "최초합 최고": ("init_best", "fitted_init_best"),
            "최초합 최저": ("init_worst", "fitted_init_worst"),
        }
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        series_handles = []
        for i, (label, (input_col, fitted_col)) in enumerate(plot_map.items()):
            if fitted_col in df_fit and df_fit[fitted_col].notnull().any():
                color = colors[i % len(colors)]
                line, = ax2.plot(
                    df_fit["year"], df_fit[fitted_col],
                    color=color, marker='o', linestyle='-', label=label, zorder=3
                )
                series_handles.append(line)
                if input_col in df and df[input_col].notnull().any():
                    ax2.scatter(
                        df["year"], df[input_col],
                        color=color, marker='x', s=80, linewidths=2, alpha=0.9, zorder=5
                    )

        if series_handles:
            leg_series = ax2.legend(handles=series_handles, loc='upper left', title='지표', frameon=True)
            ax2.add_artist(leg_series)

        marker_legend = [
            Line2D([0], [0], color='gray', marker='o', linestyle='-', label='모델 적합치'),
            Line2D([0], [0], color='gray', marker='x', linestyle='None', markersize=9, label='원본 입력값'),
        ]
        ax2.legend(handles=marker_legend, loc='lower right', title='표기', frameon=True)

        ax2.set_title("입시학년도별 적합 지표 추세")
        ax2.set_xlabel("입시학년도")
        ax2.set_ylabel("등급")
        ax2.invert_yaxis()
        ax2.grid(True, linestyle=':', alpha=0.6)

        years_ticks = sorted(pd.Series(df_fit["year"]).dropna().astype(int).unique().tolist())
        if years_ticks:
            ax2.set_xticks(years_ticks)
            ax2.set_xticklabels([str(y) for y in years_ticks])

        st.pyplot(fig2, use_container_width=True)

    # --------- 우: 합격 가능성 분석 ---------
    with col2:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        lo, hi = grade_bounds

        # 배경 구간
        ax.axvspan(lo, g80,  alpha=0.12, color="#1f77b4", zorder=0)
        ax.axvspan(g80, g50, alpha=0.12, color="#2ca02c", zorder=0)
        ax.axvspan(g50, g20, alpha=0.12, color="#ffbf00", zorder=0)
        ax.axvspan(g20, hi,  alpha=0.12, color="#d62728", zorder=0)

        # 경계선
        ax.axvline(g80, ls=":", lw=1, color="gray")
        ax.axvline(g50, ls=":", lw=1, color="gray")
        ax.axvline(g20, ls=":", lw=1, color="gray")

        # 최종컷 시뮬레이션 히스토그램
        if len(sim["Ts"]) > 0:
            ax.hist(sim["Ts"], bins=30, density=True, alpha=0.40, label="최종컷 시뮬레이션 분포")

        # 지원자 점수선 & 리스크 라벨
        ax.axvline(applicant_grade, ls="--", lw=2, color="black", label=f"지원자 점수 = {applicant_grade:.2f}")
        ax.text(applicant_grade, ax.get_ylim()[1]*0.95, f"▶ {'안정권' if p>=0.8 else '적정권' if p>=0.5 else '소신권' if p>=0.2 else '위험권'}",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

        # 중앙값 가이드
        ax.axvline(g50, ls=":", color="blue", label=f"예상 최종컷 중앙값 ≒ {g50:.2f}")

        ax.set_title("합격 가능성 분석")
        ax.set_xlabel("내신 등급 (낮을수록 우수)")
        ax.set_ylabel("확률 밀도")
        ax.set_xlim(lo, hi)
        ax.invert_xaxis()

        # 범례(차트용)는 등급 경계값 표기 유지
        zone_handles = [
            Patch(color="#1f77b4", alpha=0.35, label=f"안정권 (≤ {g80:.2f})"),
            Patch(color="#2ca02c", alpha=0.35, label=f"적정권 ({g80:.2f} ~ {g50:.2f})"),
            Patch(color="#ffbf00", alpha=0.35, label=f"소신권 ({g50:.2f} ~ {g20:.2f})"),
            Patch(color="#d62728", alpha=0.35, label=f"위험권 (≥ {g20:.2f})"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + zone_handles, loc="best")
        st.pyplot(fig, use_container_width=True)

    # --------- 요약 메트릭 ---------
    st.markdown("### 결과 요약")
    def _fmt_ci(ci_tuple):
        a, b = ci_tuple if isinstance(ci_tuple, (list, tuple)) and len(ci_tuple) == 2 else (np.nan, np.nan)
        if not (np.isfinite(a) and np.isfinite(b)): return "—"
        return f"{a*100:.1f}% ~ {b*100:.1f}%"

    c1, c2, c3 = st.columns(3)
    risk_help = (
        "구간 정의(합격확률 기준):\n"
        "- 안정권: 합격확률 ≥ 80%\n"
        "- 적정권: 50% ≤ 합격확률 < 80%\n"
        "- 소신권: 20% ≤ 합격확률 < 50%\n"
        "- 위험권: 합격확률 < 20%"
    )
    st.metric(label="지원자 점수 위치", value=("안정권" if p>=0.8 else "적정권" if p>=0.5 else "소신권" if p>=0.2 else "위험권"),
              delta=f"{applicant_grade} 등급", help=risk_help)
    c1.metric("예상 합격 확률", f"{sim['p_accept']*100:.1f}%", help=f"90% 신뢰구간: {_fmt_ci(sim['p_accept_ci'])}")
    c2.metric("합격 시 예상 순위(상위)", (f"{sim['p_pct']*100:.1f}%" if np.isfinite(sim['p_pct']) else "—"),
              help=f"90% CI: {_fmt_ci(sim['p_pct_ci'])}")
    c3.metric("올해 모집정원", f"{proj['capacity_this']}명",
              help=f"정원 {proj['capacity_this']}명 + 추가충원 {(extra_this_year if extra_this_year is not None else '자동 추정')}")

    # 보충 정보
    is_ratio_known = proj.get("ratio_this") is not None
    is_extra_known = extra_this_year is not None
    if sim.get("mode") == "probabilistic":
        info_text = []
        if not is_ratio_known or not is_extra_known:
            info_text.append("💡 **올해 미입력 정보를 다음과 같이 추정하여 분석했습니다:**")
            if sim.get("forecasted_ratio_median") is not None:
                med, ci = sim["forecasted_ratio_median"], sim["forecasted_ratio_ci"]
                info_text.append(f"- **경쟁률:** 중앙값 **{med:.2f}:1** (90% CI: {ci[0]:.2f} ~ {ci[1]:.2f})")
            if sim.get("forecasted_extra_rate_median") is not None:
                med, ci = sim["forecasted_extra_rate_median"], sim["forecasted_extra_rate_ci"]
                info_text.append(f"- **충원율:** 중앙값 **{med:.1%}** (90% CI: {ci[0]:.1%} ~ {ci[1]:.1%})")
        else:
            info_text.append(f"올해 경쟁률 **{proj['ratio_this']:.2f}:1** (지원자 약 {proj['M_this']}명) 기준 분석입니다.")
        if info_text:
            st.info("\n".join(info_text))
    elif sim.get("mode") == "fallback":
        st.warning("경쟁률/충원율 과거 데이터가 부족하여 **폴백 모드**(합격자 분포만으로 추정)로 분석되었습니다. 불확실성이 큽니다.")
    elif sim.get("mode") == "underfilled":
        st.info("지원자가 정원에 못 미치는 **미달 시나리오(underfilled)**로 간주하여 컷이 상단으로 수렴하는 폴백을 적용했습니다.")

# ===================== 상단 타이틀/사이드바 =====================
st.title("🎓 수시 합격 가능성 예측기")

with st.sidebar:
    st.header("👤 지원자 정보")
    applicant_grade = st.number_input(
        "나의 내신 등급", min_value=1.0, max_value=9.0, value=4.44, step=0.01,
        help="분석하고 싶은 본인의 등급을 입력하세요."
    )

    st.header("⚙️ 시뮬레이션 설정")
    n_sims = st.slider(
        "시뮬레이션 횟수", min_value=1000, max_value=10000, value=5000, step=500,
        help="높을수록 안정적인 추정이 가능합니다."
    )

    use_app_dist_features = st.checkbox(
        "지원자 분포 특성반영", value=True,
        help="지원자 최고/평균/최저(app_*) 데이터를 활용해 지원자 풀의 분포 형태(왜도 등)를 더 정교하게 추정합니다. 관련 데이터가 신뢰할 수 있을 때 유효합니다."
    )

    # ✅ 디버그 모드 (복원)
    debug_mode = st.checkbox(
        "디버그 모드 (중간값 출력)", value=False,
        help="전처리/피팅/투영/시뮬레이션의 핵심 중간값을 화면에 표시합니다."
    )

# ---- 기본 데이터 ----
default_df = pd.DataFrame([
    {"year": 2023, "mean": 3.83, "median": None, "p70": None, "final_cut": 5.85, "best": 2.73,
     "capacity": 14, "extra": 8, "competition_ratio": 14.79,
     "app_best": 1.80, "app_mean": 4.25, "app_worst": 8.72,
     "init_best": 2.94, "init_mean": 3.67, "init_worst": 5.67},
    {"year": 2024, "mean": 4.12, "median": None, "p70": None, "final_cut": 5.29, "best": 3.05,
     "capacity": 6, "extra": 6, "competition_ratio": 22.17,
     "app_best": 1.90, "app_mean": 4.32, "app_worst": 7.52,
     "init_best": 1.90, "init_mean": 3.54, "init_worst": 4.86},
    {"year": 2025, "mean": 3.72, "median": None, "p70": None, "final_cut": 4.50, "best": 2.76,
     "capacity": 11, "extra": 5, "competition_ratio": 32.73,
     "app_best": 1.82, "app_mean": 3.96, "app_worst": 7.22,
     "init_best": 1.82, "init_mean": 3.49, "init_worst": 4.50},
], dtype=object)

# ---- 최초 세션 초기화 ----
if 'init' not in st.session_state:
    st.session_state.department_name = "가톨릭대 미디어기술콘텐츠학과 잠재능력우수자서류"
    st.session_state.capacity_this = 6
    st.session_state.extra_this = 0
    st.session_state.ratio_this = 0.0
    # 🆕 명시적 입력 상태 제어
    st.session_state.extra_inputted = False
    st.session_state.ratio_inputted = False
    st.session_state.df_editor = default_df.copy()
    st.session_state.init = True

# ===================== 프로필 Pre-apply (위젯 생성 전 1회 적용) =====================
def _apply_profile_payload(payload: dict, column_order):
    cur = payload.get("current_year_inputs", {}) if isinstance(payload, dict) else {}
    st.session_state.department_name = str(payload.get("department_name", "") or "")
    st.session_state.capacity_this   = int(cur.get("capacity", 1) or 1)
    st.session_state.extra_this      = int(cur.get("extra", 0) or 0)
    st.session_state.ratio_this      = float(cur.get("ratio", 0.0) or 0.0)

    # 🆕 입력 상태 복원 (기본값: 입력 안함)
    st.session_state.extra_inputted = cur.get("extra_inputted", False)
    st.session_state.ratio_inputted = cur.get("ratio_inputted", False)

    hist = pd.DataFrame(payload.get("historical_data", []))
    st.session_state.df_editor = hist.reindex(columns=column_order)

if st.session_state.get("_PROFILE_TO_APPLY", None) is not None:
    try:
        _apply_profile_payload(st.session_state["_PROFILE_TO_APPLY"], COLUMN_ORDER)
        st.session_state["_PROFILE_TO_APPLY"] = None
        st.session_state["_UPLOAD_REV"] = st.session_state.get("_UPLOAD_REV", 0) + 1
    except Exception as e:
        st.session_state["_PROFILE_TO_APPLY"] = None
        st.session_state["_UPLOAD_REV"] = st.session_state.get("_UPLOAD_REV", 0) + 1
        st.session_state["_PROFILE_APPLY_ERROR"] = f"프로필 적용 실패: {e}"

# ===================== 본문: 프로필/표/파일 IO/실행 =====================
with st.container(border=True):
    
    st.markdown("##### 올해 입시 정보")
   

    # 🆕 2줄 레이아웃: 1줄에 학과명과 정원, 2줄에 추가충원, 입력함, 경쟁률, 입력함
    # 1줄: 학과명 | 정원
    c1, c2 = st.columns([1, 1])
    department_name = c1.text_input("학과명", key="department_name")
    capacity_this = c2.number_input("정원", min_value=1, step=1, key="capacity_this")
    
    # 2줄: 추가충원 | 입력함 | 경쟁률 | 입력함
    c3, c4, c5, c6 = st.columns([1, 1, 1, 1])
    
    # 추가충원
    extra_this = c3.number_input(
        "추가충원", 
        min_value=0, 
        step=1, 
        key="extra_this",
        disabled=not st.session_state.get("extra_inputted", False)
    )
    
    # 추가충원 입력함 체크박스
    extra_inputted = c4.checkbox(
        "입력함", 
        value=st.session_state.get("extra_inputted", False),
        key="extra_inputted",
        help="체크하면 추가충원 값을 입력하고, 체크하지 않으면 자동 추정됩니다",
        on_change=lambda: _reset_value_on_uncheck("extra_inputted", "extra_this", 0)
    )
    
    # 경쟁률
    ratio_this = c5.number_input(
        "경쟁률",
        min_value=0.0,
        step=0.1,
        key="ratio_this",
        disabled=not st.session_state.get("ratio_inputted", False)
    )
    
    # 경쟁률 입력함 체크박스
    ratio_inputted = c6.checkbox(
        "입력함",
        value=st.session_state.get("ratio_inputted", False),
        key="ratio_inputted",
        help="체크하면 경쟁률을 입력하고, 체크하지 않으면 자동 추정됩니다",
        on_change=lambda: _reset_value_on_uncheck("ratio_inputted", "ratio_this", 0.0)
    )

    # ------ 과거 입시 결과 (폼으로 편집-적용 분리) ------
    st.markdown("##### 과거 입시 결과")
    st.caption("표를 편집한 뒤, 아래 ‘표 변경 적용’을 눌러 반영하세요.")

    column_config = {
        "year": st.column_config.NumberColumn("입시학년도", min_value=2000, max_value=2100, step=1, format="%d"),
        "best": st.column_config.NumberColumn("최종등록자 최고", step=0.01, format="%.2f"),
        "median": st.column_config.NumberColumn("최종등록자 50%", step=0.01, format="%.2f"),
        "p70": st.column_config.NumberColumn("최종등록자 70%", step=0.01, format="%.2f"),
        "final_cut": st.column_config.NumberColumn("최종등록자 최저(컷)", step=0.01, format="%.2f"),
        "mean": st.column_config.NumberColumn("최종등록자 평균", step=0.01, format="%.2f"),
        "init_best": st.column_config.NumberColumn("최초합격자 최고", step=0.01, format="%.2f"),
        "init_worst": st.column_config.NumberColumn("최초합격자 최저", step=0.01, format="%.2f"),
        "init_mean": st.column_config.NumberColumn("최초합격자 평균", step=0.01, format="%.2f"),
        "app_best": st.column_config.NumberColumn("지원자 최고", step=0.01, format="%.2f"),
        "app_worst": st.column_config.NumberColumn("지원자 최저", step=0.01, format="%.2f"),
        "app_mean": st.column_config.NumberColumn("지원자 평균", step=0.01, format="%.2f"),
        "capacity": st.column_config.NumberColumn("정원", min_value=0, step=1, format="%d"),
        "extra": st.column_config.NumberColumn("추가충원", min_value=0, step=1, format="%d"),
        "competition_ratio": st.column_config.NumberColumn("경쟁률", min_value=0.0, step=0.01, format="%.2f"),
    }

    df_display = st.session_state.df_editor.reindex(columns=COLUMN_ORDER).rename(columns=COLUMN_CONFIG)

    with st.form("history_form", border=False):
        edited_df_display = st.data_editor(
            df_display,
            key="hist_editor",
            column_config=column_config,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )
        submitted = st.form_submit_button("✅ 표 변경 적용", use_container_width=True)
        if submitted:
            try:
                st.session_state.df_editor = edited_df_display.rename(columns=REVERSE_COLUMN_MAP).reindex(columns=COLUMN_ORDER)
                st.success("표 변경 사항을 반영했습니다.")
            except Exception as e:
                st.error(f"표 변경 반영 중 오류: {e}")

    # ------ 프로필 파일 저장하기 / 불러오기 ------
    st.markdown("##### 프로필 파일  저장하기 / 불러오기")
    lcol, rcol = st.columns([1,1], vertical_alignment="center")

    with lcol:
        data_to_save = {
            "department_name": st.session_state.get("department_name", ""),
            "current_year_inputs": {
                "capacity":     int(st.session_state.get("capacity_this", 0) or 0),
                "extra":        int(st.session_state.get("extra_this", 0) or 0),
                "ratio":        float(st.session_state.get("ratio_this", 0.0) or 0.0),
                # 🆕 입력 상태 정보 저장
                "extra_inputted": st.session_state.get("extra_inputted", False),
                "ratio_inputted": st.session_state.get("ratio_inputted", False),
            },
            "historical_data": st.session_state.df_editor.reindex(columns=COLUMN_ORDER).to_dict(orient="records"),
        }
        json_data = json.dumps(data_to_save, indent=2, ensure_ascii=False)
        safe_department_name = "".join([c for c in st.session_state.get("department_name","") if c.isalnum() or c in (' ', '-')]).rstrip()
        st.download_button(
            "💾 저장하기 (JSON)", data=json_data,
            file_name=f'{safe_department_name or "profile"}_프로필.json',
            mime='application/json', use_container_width=True
        )

        if "_UPLOAD_REV" not in st.session_state:
            st.session_state["_UPLOAD_REV"] = 0

    with rcol:
        uc1, uc2 = st.columns([3,1], vertical_alignment="center")
        uploaded_file = uc1.file_uploader(
            "불러오기 (JSON)", type="json",
            key=f"prof_uploader_{st.session_state['_UPLOAD_REV']}",
            help="파일 선택 후 ‘적용' 버튼을 누르세요."
        )
        apply_disabled = uploaded_file is None
        if uc2.button("적용", use_container_width=True, disabled=apply_disabled):
            if uploaded_file is None:
                st.warning("먼저 JSON 파일을 선택하세요.")
            else:
                try:
                    payload = json.load(uploaded_file)
                    st.session_state["_PROFILE_TO_APPLY"] = payload
                    st.session_state["_UPLOAD_REV"] += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

        if st.session_state.get("_PROFILE_APPLY_ERROR"):
            st.error(st.session_state.pop("_PROFILE_APPLY_ERROR"))

# ------ 실행 버튼 ------
if st.button("🚀 예측 실행", type="primary", use_container_width=True):
    with st.spinner("데이터 분석 및 몬테카를로 시뮬레이션을 실행 중입니다..."):
        try:
            np.random.seed(None)  # 항상 비고정

            current_weights = BASE_WEIGHTS.copy()
            if use_app_dist_features:
                st.info("ℹ️ 옵션 활성: 지원자 분포 특성(app_*)을 반영합니다.")
                current_weights.update({"app_mean": 0.3, "app_best": 0.3, "app_worst": 0.3})

            def to_py_type(v):
                if v is None: return None
                if isinstance(v, str) and v.strip() == "": return None
                if isinstance(v, (float, np.floating)) and pd.isna(v): return None
                try: return float(v)
                except (ValueError, TypeError): return None

            years_data = []
            for row in st.session_state.df_editor.to_dict(orient="records"):
                rec = {k: to_py_type(row.get(k)) for k in COLUMN_ORDER}
                if rec.get("year") is not None:
                    years_data.append(rec)

            run_pipeline(
                years_data=years_data,
                department_name=st.session_state.department_name,
                applicant_grade=float(applicant_grade),
                grade_bounds=GRADE_BOUNDS,
                capacity_this_year=int(st.session_state.capacity_this),
                # 🆕 입력 상태에 따른 값 전달
                extra_this_year=(int(st.session_state.extra_this) if st.session_state.get("extra_inputted", False) else None),
                ratio_this_year=(float(st.session_state.ratio_this) if st.session_state.get("ratio_inputted", False) else None),
                n_scenarios=int(n_sims),
                weights=current_weights,
                debug=bool(debug_mode)
            )
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.exception(e)
