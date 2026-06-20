"""合成データ生成：ネットワーク不要でロジックを検証/デモするためのダミー株価。

実データの代わりに、性質の異なる複数の銘柄パターンを生成する。
検知ロジックが「徐々に上昇」を高評価し、「横ばい/急騰過熱/下降」を
適切に区別できるかの確認に使う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ohlcv_from_close(
    close: np.ndarray, start: str = "2022-01-03", base_vol: float = 1_000_000.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """終値配列から、整合的な OHLCV の日足 DataFrame を作る。"""
    rng = rng or np.random.default_rng(0)
    n = len(close)
    idx = pd.bdate_range(start=start, periods=n)
    close = np.maximum(close, 1.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    intraday = np.abs(rng.normal(0, 0.01, n)) * close
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    # 上昇日は出来高多めにして「出来高の裏付け」を表現
    up = np.concatenate([[0.0], np.diff(close)]) > 0
    vol = base_vol * (1.0 + 0.4 * up) * rng.uniform(0.7, 1.3, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def make_steady_uptrend(n: int = 400, daily: float = 0.0025, noise: float = 0.008,
                        seed: int = 1) -> pd.DataFrame:
    """徐々に・着実に上昇（理想ケース）。日次 +0.25%、低ノイズ。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(daily, noise, n)
    close = 1000.0 * np.exp(np.cumsum(rets))
    return _ohlcv_from_close(close, rng=rng)


def make_choppy_flat(n: int = 400, noise: float = 0.015, seed: int = 2) -> pd.DataFrame:
    """横ばい・ノイズ多め（検知されてはいけないケース）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, noise, n)
    close = 1000.0 * np.exp(np.cumsum(rets))
    return _ohlcv_from_close(close, rng=rng)


def make_spike_overbought(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """前半横ばい→直近に急騰し過熱（フィルタで除外されるべきケース）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    rets[-15:] += 0.03  # 直近15日で急騰
    close = 1000.0 * np.exp(np.cumsum(rets))
    return _ohlcv_from_close(close, rng=rng)


def make_downtrend(n: int = 400, daily: float = -0.002, seed: int = 4) -> pd.DataFrame:
    """緩やかな下降（検知されてはいけないケース）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(daily, 0.01, n)
    close = 1000.0 * np.exp(np.cumsum(rets))
    return _ohlcv_from_close(close, rng=rng)


def make_low_liquidity(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """上昇しているが売買代金が小さい（流動性フィルタで除外）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0025, 0.008, n)
    close = 1000.0 * np.exp(np.cumsum(rets))
    df = _ohlcv_from_close(close, rng=rng)
    df["Volume"] = df["Volume"] * 0.005  # 出来高を極端に小さく
    return df


def demo_universe() -> dict[str, pd.DataFrame]:
    """デモ/テスト用の銘柄群。キーは説明的な擬似コード。"""
    return {
        "STEADY_UP": make_steady_uptrend(),
        "CHOPPY_FLAT": make_choppy_flat(),
        "SPIKE_OB": make_spike_overbought(),
        "DOWNTREND": make_downtrend(),
        "LOW_LIQ": make_low_liquidity(),
    }
