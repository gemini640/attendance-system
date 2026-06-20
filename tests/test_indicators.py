"""indicators モジュールの単体テスト。"""

import numpy as np
import pandas as pd

from stock_detector import indicators as ind


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[1])          # window 未満は NaN
    assert out.iloc[2] == 2.0             # (1+2+3)/3
    assert out.iloc[4] == 4.0             # (3+4+5)/3


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40), dtype=float)  # 単調増加
    out = ind.rsi(s, 14)
    assert out.dropna().iloc[-1] == 100.0


def test_rsi_range():
    rng = np.random.default_rng(0)
    s = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    out = ind.rsi(s, 14).dropna()
    assert out.between(0, 100).all()


def test_linreg_detects_smooth_uptrend():
    # きれいな指数的上昇 → 傾き>0 かつ R²≈1
    close = pd.Series(1000 * np.exp(np.linspace(0, 0.4, 60)))
    reg = ind.linreg_slope_r2(close, 40)
    last = reg.iloc[-1]
    assert last["slope_pct"] > 0
    assert last["r2"] > 0.99


def test_linreg_noise_lowers_r2():
    rng = np.random.default_rng(1)
    # 傾きは同じでもノイズを足すと R² は下がる
    base = np.linspace(0, 0.4, 60)
    clean = pd.Series(1000 * np.exp(base))
    noisy = pd.Series(1000 * np.exp(base + rng.normal(0, 0.05, 60)))
    r2_clean = ind.linreg_slope_r2(clean, 40).iloc[-1]["r2"]
    r2_noisy = ind.linreg_slope_r2(noisy, 40).iloc[-1]["r2"]
    assert r2_clean > r2_noisy


def test_macd_columns():
    s = pd.Series(np.arange(1, 100), dtype=float)
    out = ind.macd(s)
    assert list(out.columns) == ["macd", "signal", "hist"]


def test_above_ratio():
    close = pd.Series([10, 11, 12, 9, 13], dtype=float)
    ref = pd.Series([10, 10, 10, 10, 10], dtype=float)
    out = ind.above_ratio(close, ref, 5)
    # 直近5日で 11,12,13 が上回り、10は等しく、9は下回る → 3/5
    assert out.iloc[-1] == 0.6
