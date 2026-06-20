"""検知ロジックの中核：銘柄のスコアリング。

価格 DataFrame（Open/High/Low/Close/Volume, 日足）を受け取り、
「徐々に上昇 ＋ これから上がりそう」を 0〜100 点で評価する。

考え方：
  5つの軸ごとに 0〜1 のサブスコアを出し、重み付き平均して 0〜100 点に。
  さらに「除外フィルタ（過熱・乖離・流動性）」に該当する銘柄は
  is_detected=False とし、ノイズを落とす。

  A. トレンド方向   : 今、上昇基調にあるか（移動平均の並びと向き）
  B. 上昇の滑らかさ : *徐々に*＝log回帰の傾き+R²（このロジックの核）
  C. 勢い           : MACD / RSI / ROC
  D. 出来高の裏付け : 出来高増加・OBV 上昇
  E. これから上がりそう: 高値ブレイク / 押し目反発 / 過熱していないか

スコアは将来の上昇を保証するものではなく、候補抽出（スクリーニング）の
補助指標である点に注意。重み・閾値は config から差し替え可能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ind


# ---- 設定（config.yaml で上書きされる既定値） ---------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sma_short": 25,        # 短期移動平均（日本株の標準的な月足相当）
    "sma_mid": 75,          # 中期
    "sma_long": 200,        # 長期
    "trend_window": 60,     # トレンド一貫性を見る期間
    "regression_window": 40,  # 回帰（滑らかさ）を測る期間 ★Bの核
    "rsi_window": 14,
    "roc_window": 20,
    "breakout_window": 60,  # 高値ブレイク判定の参照期間
    "vol_window": 25,       # 出来高平均の期間
    # 重み（合計で正規化されるので比率でよい）
    "weights": {
        "trend": 0.25,      # A
        "smoothness": 0.30,  # B ★最重視
        "momentum": 0.20,   # C
        "volume": 0.10,     # D
        "timing": 0.15,     # E
    },
    # 除外フィルタ
    "filters": {
        "rsi_overbought": 78.0,        # これ超は買われすぎで除外
        "max_ext_from_sma_pct": 25.0,  # 25日線からの上方乖離がこれ超は過熱で除外
        "min_avg_turnover_jpy": 5.0e7,  # 平均売買代金(円)。流動性下限（既定5千万円）
    },
    "detect_threshold": 60.0,  # 検知とみなす合計スコアの下限
}


@dataclass
class SignalResult:
    """1 銘柄・最新時点の評価結果。"""

    symbol: str
    score: float                       # 0〜100 合計スコア
    is_detected: bool                  # 閾値超 かつ 除外フィルタ非該当
    sub_scores: dict[str, float] = field(default_factory=dict)  # 軸別 0〜1
    metrics: dict[str, float] = field(default_factory=dict)     # 主要指標の生値
    reasons: list[str] = field(default_factory=list)            # 検知/除外の理由
    rejected_by: list[str] = field(default_factory=list)        # 該当した除外フィルタ

    def to_row(self) -> dict[str, Any]:
        """一覧表示（DataFrame 化）用の平坦な dict。"""
        row: dict[str, Any] = {
            "symbol": self.symbol,
            "score": round(self.score, 1),
            "detected": self.is_detected,
        }
        for k, v in self.sub_scores.items():
            row[f"s_{k}"] = round(v, 3)
        for k, v in self.metrics.items():
            row[k] = round(v, 3) if isinstance(v, float) else v
        row["reasons"] = "; ".join(self.reasons)
        row["rejected_by"] = "; ".join(self.rejected_by)
        return row


def _clip01(x: float) -> float:
    """NaN を 0 に潰しつつ 0〜1 にクリップ。"""
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _ramp(x: float, lo: float, hi: float) -> float:
    """x を [lo, hi] で 0→1 に線形変換（範囲外は 0/1 に飽和）。"""
    if not np.isfinite(x):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return _clip01((x - lo) / (hi - lo))


def compute_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """価格 DataFrame に各種指標列を付与して返す（時系列のまま）。

    backtest でも使い回せるよう、最新行だけでなく全行ぶん計算する。
    """
    out = df.copy()
    close = out["Close"]

    out["sma_s"] = ind.sma(close, cfg["sma_short"])
    out["sma_m"] = ind.sma(close, cfg["sma_mid"])
    out["sma_l"] = ind.sma(close, cfg["sma_long"])
    out["sma_s_slope"] = ind.slope(out["sma_s"], 10)

    reg = ind.linreg_slope_r2(close, cfg["regression_window"])
    out["reg_slope"] = reg["slope_pct"]
    out["reg_r2"] = reg["r2"]

    macd_df = ind.macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["hist"]

    out["rsi"] = ind.rsi(close, cfg["rsi_window"])
    out["roc"] = ind.roc(close, cfg["roc_window"])

    out["obv"] = ind.obv(out)
    out["obv_slope"] = ind.slope(out["obv"], 20)
    out["vol_avg"] = ind.sma(out["Volume"], cfg["vol_window"])

    out["hh"] = ind.rolling_high(close, cfg["breakout_window"])
    out["trend_consistency"] = ind.above_ratio(
        close, out["sma_s"], cfg["trend_window"]
    )
    # 25日線からの上方乖離率(%)。過熱フィルタ＆タイミングに使用。
    out["ext_from_sma_pct"] = (close - out["sma_s"]) / out["sma_s"] * 100.0
    out["turnover"] = close * out["Volume"]
    out["turnover_avg"] = ind.sma(out["turnover"], cfg["vol_window"])
    return out


def _score_trend(row: pd.Series) -> tuple[float, list[str]]:
    """A. トレンド方向（移動平均の並びと向き、一貫性）。"""
    reasons: list[str] = []
    parts: list[float] = []

    close = row["Close"]
    # パーフェクトオーダー：株価 > 短期 > 中期 > 長期 を満たすほど高得点
    levels = [close, row["sma_s"], row["sma_m"], row["sma_l"]]
    if all(np.isfinite(v) for v in levels):
        order_ok = sum(
            1 for a, b in zip(levels, levels[1:]) if a > b
        ) / (len(levels) - 1)
        parts.append(order_ok)
        if order_ok == 1.0:
            reasons.append("パーフェクトオーダー（株価>25>75>200日線）")
    elif np.isfinite(row["sma_s"]):
        # 長期線が無い（データ短い）場合は短期線との位置で代替
        parts.append(1.0 if close > row["sma_s"] else 0.0)

    # 短期線の傾き（上向きか）
    parts.append(_ramp(row.get("sma_s_slope", np.nan), 0.0, 0.3))
    # トレンドの一貫性（直近で株価が短期線の上にあった割合）
    tc = row.get("trend_consistency", np.nan)
    parts.append(_clip01(tc))
    if np.isfinite(tc) and tc >= 0.8:
        reasons.append(f"直近の{int(tc * 100)}%の期間で25日線の上を維持")

    return (float(np.mean(parts)) if parts else 0.0), reasons


def _score_smoothness(row: pd.Series) -> tuple[float, list[str]]:
    """B. 上昇の滑らかさ ★核：log回帰の傾き×R²。

    傾きがプラスで R² が高いほど「徐々に・着実な上昇」。
    傾きがマイナスなら 0（下降を滑らかと評価しない）。
    """
    slope_pct = row.get("reg_slope", np.nan)
    r2 = row.get("reg_r2", np.nan)
    if not np.isfinite(slope_pct) or not np.isfinite(r2):
        return 0.0, []
    if slope_pct <= 0:
        return 0.0, []

    # 傾き：0.05%/日〜0.5%/日 を 0→1（年率換算で概ね +13%〜+200%）
    slope_score = _ramp(slope_pct, 0.05, 0.5)
    # R²：0.5〜0.9 を 0→1（直線的なほど高評価）
    r2_score = _ramp(r2, 0.5, 0.9)
    score = slope_score * r2_score  # 両方そろって初めて高得点（積）

    reasons: list[str] = []
    if score >= 0.5:
        reasons.append(
            f"着実な右肩上がり（傾き{slope_pct:.2f}%/日, R²={r2:.2f}）"
        )
    return score, reasons


def _score_momentum(row: pd.Series) -> tuple[float, list[str]]:
    """C. 勢い：MACD・RSI・ROC。"""
    reasons: list[str] = []
    parts: list[float] = []

    # MACD：シグナル上抜け & ヒストグラム陽転
    macd_v, sig_v, hist = row.get("macd"), row.get("macd_signal"), row.get("macd_hist")
    if all(np.isfinite(v) for v in (macd_v, sig_v, hist)):
        macd_score = 0.0
        if macd_v > sig_v:
            macd_score += 0.6
        if hist > 0:
            macd_score += 0.4
        parts.append(macd_score)
        if macd_v > sig_v and hist > 0:
            reasons.append("MACDがシグナルを上抜け（上昇の勢い）")

    # RSI：50〜70 を理想帯として山形に評価（過熱は別途フィルタ）
    rsi_v = row.get("rsi", np.nan)
    if np.isfinite(rsi_v):
        if rsi_v <= 50:
            parts.append(_ramp(rsi_v, 40, 50) * 0.6)
        elif rsi_v <= 70:
            parts.append(1.0)
        else:
            parts.append(_clip01(1.0 - (rsi_v - 70) / 15.0))  # 過熱で減点

    # ROC：直近の騰落率がプラス
    roc_v = row.get("roc", np.nan)
    parts.append(_ramp(roc_v, 0.0, 15.0))

    return (float(np.mean(parts)) if parts else 0.0), reasons


def _score_volume(row: pd.Series) -> tuple[float, list[str]]:
    """D. 出来高の裏付け：直近出来高 vs 平均、OBV の向き。"""
    reasons: list[str] = []
    parts: list[float] = []

    vol, vol_avg = row.get("Volume"), row.get("vol_avg")
    if np.isfinite(vol) and np.isfinite(vol_avg) and vol_avg > 0:
        ratio = vol / vol_avg
        parts.append(_ramp(ratio, 0.8, 1.8))  # 平均比 0.8→1.8 倍で 0→1
        if ratio >= 1.5:
            reasons.append(f"出来高が平均の{ratio:.1f}倍（注目度上昇）")

    # OBV の傾き（資金が流入方向か）
    parts.append(_ramp(row.get("obv_slope", np.nan), 0.0, 1.0))

    return (float(np.mean(parts)) if parts else 0.0), reasons


def _score_timing(row: pd.Series) -> tuple[float, list[str]]:
    """E. これから上がりそう：高値ブレイク／押し目／過熱していないか。"""
    reasons: list[str] = []
    parts: list[float] = []

    close, hh = row.get("Close"), row.get("hh")
    ext = row.get("ext_from_sma_pct", np.nan)

    # 高値圏ブレイク：直近高値に近い/更新（0.97〜1.0 倍で加点）
    if np.isfinite(close) and np.isfinite(hh) and hh > 0:
        prox = close / hh
        parts.append(_ramp(prox, 0.97, 1.0))
        if prox >= 1.0:
            reasons.append("直近高値を更新（ブレイク）")

    # 乖離が適度（押し目〜順張り）であるほど良い。乖離大は過熱で減点。
    # 0%付近〜+8% を理想、+8%超で徐々に減点。
    if np.isfinite(ext):
        if ext < 0:
            parts.append(_ramp(ext, -8.0, 0.0))   # 下に離れすぎは弱い
        elif ext <= 8.0:
            parts.append(1.0)
            if 0 <= ext <= 4:
                reasons.append("25日線に近く過熱が少ない（押し目圏）")
        else:
            parts.append(_clip01(1.0 - (ext - 8.0) / 17.0))

    return (float(np.mean(parts)) if parts else 0.0), reasons


def _apply_filters(row: pd.Series, cfg: dict[str, Any]) -> list[str]:
    """除外フィルタ。該当した理由名のリストを返す（空なら通過）。"""
    f = cfg["filters"]
    rejected: list[str] = []

    rsi_v = row.get("rsi", np.nan)
    if np.isfinite(rsi_v) and rsi_v > f["rsi_overbought"]:
        rejected.append(f"RSI過熱({rsi_v:.0f}>{f['rsi_overbought']:.0f})")

    ext = row.get("ext_from_sma_pct", np.nan)
    if np.isfinite(ext) and ext > f["max_ext_from_sma_pct"]:
        rejected.append(
            f"25日線から上方乖離しすぎ({ext:.0f}%>{f['max_ext_from_sma_pct']:.0f}%)"
        )

    turnover = row.get("turnover_avg", np.nan)
    if np.isfinite(turnover) and turnover < f["min_avg_turnover_jpy"]:
        rejected.append("流動性不足（平均売買代金が下限未満）")

    return rejected


def score_row(symbol: str, row: pd.Series, cfg: dict[str, Any]) -> SignalResult:
    """指標付与済みの1行（ある時点）をスコアリングする。"""
    axes = {
        "trend": _score_trend,
        "smoothness": _score_smoothness,
        "momentum": _score_momentum,
        "volume": _score_volume,
        "timing": _score_timing,
    }
    sub_scores: dict[str, float] = {}
    reasons: list[str] = []
    for name, fn in axes.items():
        s, rs = fn(row)
        sub_scores[name] = _clip01(s)
        reasons.extend(rs)

    weights = cfg["weights"]
    total_w = sum(weights.values())
    weighted = sum(sub_scores[k] * weights[k] for k in sub_scores) / total_w
    score = round(weighted * 100.0, 2)

    rejected = _apply_filters(row, cfg)
    is_detected = (score >= cfg["detect_threshold"]) and not rejected

    metrics = {
        k: float(row[k])
        for k in ("Close", "reg_slope", "reg_r2", "rsi", "ext_from_sma_pct")
        if k in row and np.isfinite(row[k])
    }
    return SignalResult(
        symbol=symbol,
        score=score,
        is_detected=is_detected,
        sub_scores=sub_scores,
        metrics=metrics,
        reasons=reasons,
        rejected_by=rejected,
    )


def score_symbol(
    symbol: str, df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> SignalResult:
    """銘柄の価格 DataFrame から最新時点のスコアを算出する高水準 API。"""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    feat = compute_features(df, cfg)
    last = feat.iloc[-1]
    return score_row(symbol, last, cfg)
