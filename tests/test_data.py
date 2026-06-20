"""data モジュールのテスト（ネットワーク不要）。

yfinance 1.4.x は単一銘柄でも列が MultiIndex（例: ('Close','7203.T')）で
返るため、その正規化と、正規化後データが検知ロジックに通ることを保証する。
"""

import numpy as np
import pandas as pd

from stock_detector.data import _normalize, to_yahoo_symbol
from stock_detector.signals import score_symbol


def test_to_yahoo_symbol():
    assert to_yahoo_symbol("7203") == "7203.T"
    assert to_yahoo_symbol("7203.T") == "7203.T"
    assert to_yahoo_symbol(" 6758 ") == "6758.T"
    assert to_yahoo_symbol("9984.T") == "9984.T"  # 付与済みはそのまま


def _yfinance_like_frame(n: int = 60) -> pd.DataFrame:
    """yfinance.download(multi_level_index=True) 相当の MultiIndex 列フレーム。"""
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(1000, 1200, n)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["7203.T"]]
    )
    data = np.column_stack(
        [close, close * 1.01, close * 0.99, close, np.full(n, 1_000_000.0)]
    )
    return pd.DataFrame(data, index=idx, columns=cols)


def test_normalize_flattens_multiindex():
    df = _normalize(_yfinance_like_frame())
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert not df.empty


def test_normalize_passthrough_plain_columns():
    idx = pd.bdate_range("2024-01-01", periods=10)
    plain = pd.DataFrame(
        {c: np.arange(10, dtype=float) for c in
         ["Open", "High", "Low", "Close", "Volume"]},
        index=idx,
    )
    out = _normalize(plain)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalized_frame_scores_without_error():
    # yfinance 形式 → 正規化 → 検知ロジックが通ることのEnd-to-End確認
    df = _normalize(_yfinance_like_frame(n=120))
    res = score_symbol("7203", df)
    assert 0.0 <= res.score <= 100.0
