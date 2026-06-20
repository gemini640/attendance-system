"""テクニカル指標の計算（純粋関数）。

すべての関数は pandas の Series / DataFrame を受け取り、新しい Series を返す。
データ取得層（yfinance など）からは独立しており、合成データでも単体テスト可能。

価格系列はすべて「終値（Close）」を想定し、十分な長さが無い場合は
先頭が NaN の系列を返す（呼び出し側で dropna / 末尾判定すること）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均（Simple Moving Average）。"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移動平均（Exponential Moving Average）。"""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def slope(series: pd.Series, window: int) -> pd.Series:
    """移動平均線などの「傾き」を、直近 window 日の1日あたり変化率(%)で返す。

    末尾値と window 日前の値から年率ではなく素朴な平均変化率を出す。
    線の向き判定（上向き/下向き）に使う軽量な指標。
    """
    prev = series.shift(window)
    return (series - prev) / prev / window * 100.0


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI（Wilder の平滑化）。0〜100。

    50 超で上昇優勢、70〜80 超で買われすぎ（過熱）とみなすのが一般的。
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder の平滑化 = alpha = 1/window の EMA
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss が 0（連続上昇）の場合は RSI=100 とする
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD を返す。列: macd, signal, hist。

    macd  = EMA(fast) - EMA(slow)
    signal= EMA(macd, signal)
    hist  = macd - signal （ヒストグラム。プラスかつ拡大で上昇加速）
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def roc(series: pd.Series, window: int = 20) -> pd.Series:
    """変化率（Rate of Change, %）。window 日前比の騰落率。"""
    return series.pct_change(periods=window) * 100.0


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR（Average True Range）。ボラティリティ（値動きの荒さ）の指標。

    df は High, Low, Close 列を持つこと。
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """OBV（On Balance Volume）。上昇日は出来高を加算、下落日は減算した累積値。

    df は Close, Volume 列を持つこと。上昇トレンドで OBV も切り上がると
    「上昇に出来高が伴っている」裏付けになる。
    """
    direction = np.sign(df["Close"].diff().fillna(0.0))
    return (direction * df["Volume"]).cumsum()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    """直近 window 日の高値（当日含む）。高値ブレイク判定に使う。"""
    return series.rolling(window=window, min_periods=1).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    """直近 window 日の安値（当日含む）。"""
    return series.rolling(window=window, min_periods=1).min()


def linreg_slope_r2(series: pd.Series, window: int) -> pd.DataFrame:
    """log(価格) に対する線形回帰の「傾き」と「決定係数 R²」を返す。

    列:
      - slope_pct: 1日あたりの平均上昇率(%)。log 回帰なので複利的な傾き。
      - r2:        0〜1。1 に近いほど「ノイズの少ない、きれいな直線的上昇/下降」。

    ★このロジックの核。slope_pct > 0 かつ r2 が高い ＝「徐々に・着実に上昇」。
      急騰スパイクは slope は大きくても r2 が下がりやすく、区別できる。
    """
    log_series = np.log(series)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev**2).sum()

    # rolling.apply は単一の float しか返せないため、slope と r2 を
    # それぞれ別の apply で計算する（下の2クロージャ）。
    def _slope(window_vals: np.ndarray) -> float:
        y = window_vals
        if np.any(~np.isfinite(y)):
            return np.nan
        b = (x_dev * (y - y.mean())).sum() / x_var
        return b * 100.0  # 1日あたり log 変化率 ≒ %/日

    def _r2(window_vals: np.ndarray) -> float:
        y = window_vals
        if np.any(~np.isfinite(y)):
            return np.nan
        y_mean = y.mean()
        b = (x_dev * (y - y_mean)).sum() / x_var
        a = y_mean - b * x_mean
        y_hat = a + b * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    slope_pct = log_series.rolling(window=window, min_periods=window).apply(
        _slope, raw=True
    )
    r2 = log_series.rolling(window=window, min_periods=window).apply(_r2, raw=True)
    return pd.DataFrame({"slope_pct": slope_pct, "r2": r2})


def above_ratio(series: pd.Series, reference: pd.Series, window: int) -> pd.Series:
    """直近 window 日で series が reference を上回っていた割合（0〜1）。

    例: 終値が 25日線の上にあった日数の割合 → トレンドの一貫性。
    """
    above = (series > reference).astype(float)
    return above.rolling(window=window, min_periods=window).mean()
