"""検知ロジック（signals / screener）の単体テスト。

合成データで「徐々に上昇は検知、横ばい/下降/過熱/薄商いは非検知」を保証する。
"""

import pandas as pd

from stock_detector import synthetic as syn
from stock_detector.signals import score_symbol
from stock_detector.screener import screen


def test_steady_uptrend_is_detected():
    df = syn.make_steady_uptrend()
    res = score_symbol("STEADY_UP", df)
    assert res.is_detected
    assert res.score >= 60
    # 滑らかさ軸が高いこと（このロジックの核）
    assert res.sub_scores["smoothness"] > 0.4


def test_flat_is_not_detected():
    res = score_symbol("CHOPPY_FLAT", syn.make_choppy_flat())
    assert not res.is_detected
    assert res.sub_scores["smoothness"] == 0.0


def test_downtrend_is_not_detected():
    res = score_symbol("DOWNTREND", syn.make_downtrend())
    assert not res.is_detected
    assert res.score < 30


def test_overbought_spike_is_filtered():
    res = score_symbol("SPIKE_OB", syn.make_spike_overbought())
    assert not res.is_detected
    # スコアではなく除外フィルタで弾かれること
    assert res.rejected_by, "過熱銘柄は除外フィルタに該当するはず"


def test_low_liquidity_is_filtered():
    res = score_symbol("LOW_LIQ", syn.make_low_liquidity())
    assert not res.is_detected
    assert any("流動性" in r for r in res.rejected_by)


def test_score_in_range():
    for name, df in syn.demo_universe().items():
        res = score_symbol(name, df)
        assert 0.0 <= res.score <= 100.0, name


def test_screen_ranks_steady_first():
    table, results = screen(syn.demo_universe())
    assert not table.empty
    # 最上位は徐々に上昇の銘柄
    assert results[0].symbol == "STEADY_UP"
    # 検知されるのは STEADY_UP のみ（他はフィルタ/低スコア）
    detected = [r.symbol for r in results if r.is_detected]
    assert detected == ["STEADY_UP"]


def test_handles_short_history():
    # 200日線が計算できない短い系列でも例外を出さない
    df = syn.make_steady_uptrend(n=30)
    res = score_symbol("SHORT", df)
    assert 0.0 <= res.score <= 100.0
