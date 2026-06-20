"""バックテスト：検知ロジックが「実際に上昇前に点灯するか」を過去データで検証。

各営業日について、その日までの情報だけでスコアを計算（先読みなし）し、
その後 horizon 営業日の将来リターンを突き合わせる。

出力する代表的な指標：
  - 検知日（score>=閾値）の平均将来リターン vs 全日の平均
  - 勝率（将来リターン > 0 の割合）
  - スコア帯別の平均将来リターン（スコアが高いほどリターンも高ければ妥当）

これにより重み・閾値を「勘」ではなくデータで詰められる。
※ 過去の有効性は将来を保証しない点に留意。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .signals import DEFAULT_CONFIG, compute_features, score_row


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    horizon: int = 20,
    warmup: int | None = None,
) -> pd.DataFrame:
    """1 銘柄を日次でスコアリングし、将来リターンを付けた DataFrame を返す。

    Args:
        horizon: 何営業日先のリターンを評価するか（既定20≒1か月）。
        warmup: 指標が安定するまでスキップする先頭日数。既定は長期線+回帰窓。
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    feat = compute_features(df, cfg)

    if warmup is None:
        warmup = cfg["sma_long"] + cfg["regression_window"]

    close = feat["Close"]
    # horizon 日先の終値リターン(%)。末尾 horizon 日は将来が無いため NaN。
    fwd_return = (close.shift(-horizon) / close - 1.0) * 100.0

    records: list[dict[str, Any]] = []
    index = feat.index
    for i in range(warmup, len(feat)):
        row = feat.iloc[i]
        res = score_row(symbol, row, cfg)
        records.append(
            {
                "date": index[i],
                "symbol": symbol,
                "score": res.score,
                "detected": res.is_detected,
                "fwd_return": float(fwd_return.iloc[i]),
            }
        )
    return pd.DataFrame(records)


def summarize(bt: pd.DataFrame, score_threshold: float | None = None) -> dict[str, Any]:
    """バックテスト結果（複数銘柄を縦に連結可）を集計する。"""
    data = bt.dropna(subset=["fwd_return"])
    if data.empty:
        return {"n": 0}

    if score_threshold is None:
        detected = data[data["detected"]]
    else:
        detected = data[data["score"] >= score_threshold]

    def _stats(d: pd.DataFrame) -> dict[str, float]:
        if d.empty:
            return {"n": 0, "mean_fwd": float("nan"), "win_rate": float("nan")}
        return {
            "n": int(len(d)),
            "mean_fwd": round(float(d["fwd_return"].mean()), 2),
            "median_fwd": round(float(d["fwd_return"].median()), 2),
            "win_rate": round(float((d["fwd_return"] > 0).mean()), 3),
        }

    # スコア帯別（単調性の確認用）
    bins = [0, 40, 50, 60, 70, 80, 100]
    data = data.copy()
    data["bucket"] = pd.cut(data["score"], bins=bins, include_lowest=True)
    by_bucket = (
        data.groupby("bucket", observed=True)["fwd_return"]
        .agg(["count", "mean"])
        .round(2)
        .to_dict("index")
    )

    return {
        "n": int(len(data)),
        "all": _stats(data),
        "detected": _stats(detected),
        "edge": round(
            _stats(detected)["mean_fwd"] - _stats(data)["mean_fwd"], 2
        )
        if not detected.empty
        else float("nan"),
        "by_score_bucket": {str(k): v for k, v in by_bucket.items()},
    }


def backtest_many(
    price_data: dict[str, pd.DataFrame],
    cfg: dict[str, Any] | None = None,
    horizon: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """複数銘柄をまとめてバックテストし、(明細, 集計) を返す。"""
    frames: list[pd.DataFrame] = []
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            frames.append(backtest_symbol(symbol, df, cfg, horizon=horizon))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {symbol}: バックテストスキップ ({e})")
    if not frames:
        return pd.DataFrame(), {"n": 0}
    allbt = pd.concat(frames, ignore_index=True)
    return allbt, summarize(allbt)
