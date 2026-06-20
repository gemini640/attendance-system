"""スクリーナー：複数銘柄を評価し、スコア順に並べた一覧を返す。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .signals import SignalResult, score_symbol


def screen(
    price_data: dict[str, pd.DataFrame],
    cfg: dict[str, Any] | None = None,
    detected_only: bool = False,
) -> tuple[pd.DataFrame, list[SignalResult]]:
    """銘柄→価格DataFrame の辞書を評価し、(一覧DataFrame, 結果リスト) を返す。

    一覧 DataFrame はスコア降順。detected_only=True で検知銘柄のみに絞る。
    """
    cfg = cfg or {}
    results: list[SignalResult] = []
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            results.append(score_symbol(symbol, df, cfg))
        except Exception as e:  # noqa: BLE001 - 1銘柄の失敗で全体を止めない
            print(f"[warn] {symbol}: 評価スキップ ({e})")

    results.sort(key=lambda r: r.score, reverse=True)
    rows = [r.to_row() for r in results]
    table = pd.DataFrame(rows)
    if detected_only and not table.empty:
        table = table[table["detected"]].reset_index(drop=True)
    return table, results
