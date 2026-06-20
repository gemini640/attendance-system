"""stock_detector：日本株の「徐々に上昇／これから上がりそう」を検知するパッケージ。

主要モジュール:
  - data        : Yahoo ファイナンスからの価格取得
  - indicators  : テクニカル指標（純粋関数）
  - signals     : 検知ロジックの中核（スコアリング）
  - screener    : 複数銘柄の一括評価・ランキング
  - backtest    : ロジックの過去データ検証
"""

from .signals import DEFAULT_CONFIG, SignalResult, score_symbol
from .screener import screen

__all__ = ["DEFAULT_CONFIG", "SignalResult", "score_symbol", "screen"]
