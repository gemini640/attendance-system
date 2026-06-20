"""データ取得層：Yahoo ファイナンスから日本株の日足を取得する。

検知ロジック（signals.py）からは独立。取得失敗やネットワーク制限に備え、
例外を握りつぶさず呼び出し側に伝える。

日本株の銘柄コードは Yahoo では「証券コード + .T」（東証）で表す。
  例: トヨタ自動車 = 7203.T、ソニーG = 6758.T
"""

from __future__ import annotations

import time

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - 実行時に明示エラー
    yf = None

# OHLCV の標準列名。signals.py はこの列名を前提にする。
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def to_yahoo_symbol(code: str) -> str:
    """証券コードを Yahoo 形式に正規化する。

    "7203" -> "7203.T"、"7203.T" -> "7203.T"（既に付いていればそのまま）。
    """
    code = str(code).strip().upper()
    if "." in code:
        return code
    return f"{code}.T"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance の戻り（MultiIndex 列になる場合あり）を OHLCV に整える。"""
    if isinstance(df.columns, pd.MultiIndex):
        # 単一銘柄取得時でも ('Close','7203.T') のようになることがある
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    df = df[cols].dropna(how="all")
    return df


def fetch_history(
    code: str,
    period: str = "1y",
    interval: str = "1d",
    retries: int = 3,
    pause: float = 1.0,
) -> pd.DataFrame:
    """1 銘柄の価格履歴を取得する。

    Args:
        code: 証券コード（"7203" など）。.T は自動付与。
        period: 取得期間（"6mo", "1y", "2y" など）。
        interval: 足の種類（"1d" 日足）。
        retries: ネットワーク失敗時の再試行回数。
        pause: 再試行間隔の基準秒（指数バックオフ）。

    Returns:
        Open/High/Low/Close/Volume を列に持つ DataFrame（DatetimeIndex）。
    """
    if yf is None:
        raise ImportError("yfinance が未インストールです。`pip install yfinance`")

    symbol = to_yahoo_symbol(code)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            # まず Ticker.history（単一銘柄で安定）、失敗時は download にフォールバック。
            df = (
                yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            )
            df = _normalize(df)
            if df.empty:
                df = _normalize(
                    yf.download(
                        symbol,
                        period=period,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                    )
                )
            if not df.empty:
                return df
            last_err = ValueError(f"空のデータ: {symbol}")
        except Exception as e:  # noqa: BLE001 - 再試行のため一旦捕捉
            last_err = e
        time.sleep(pause * (2**attempt))
    raise RuntimeError(f"{symbol} の取得に失敗しました: {last_err}")


def fetch_many(
    codes: list[str], period: str = "1y", interval: str = "1d", pause: float = 0.5
) -> dict[str, pd.DataFrame]:
    """複数銘柄をまとめて取得。失敗した銘柄はスキップしログ的に欠落させる。

    返り値は {証券コード: DataFrame}。Yahoo への負荷を避けるため銘柄間で待機。
    """
    out: dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            out[code] = fetch_history(code, period=period, interval=interval)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {code}: 取得スキップ ({e})")
        time.sleep(pause)
    return out
