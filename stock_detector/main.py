"""コマンドラインエントリポイント。

使い方:
  # 監視リストをスクリーニング（実データ取得が必要）
  python -m stock_detector.main screen

  # ロジックを過去データで検証
  python -m stock_detector.main backtest

  # ネットワーク不要のデモ（合成データで検知ロジックの挙動を確認）
  python -m stock_detector.main demo
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | None) -> dict[str, Any]:
    """config.yaml を読み込む。未指定なら同梱の既定ファイル。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_table(table: pd.DataFrame, cols: list[str] | None = None) -> None:
    if table.empty:
        print("（該当なし）")
        return
    cols = cols or [
        c
        for c in ["symbol", "score", "detected", "s_smoothness", "reg_slope",
                  "reg_r2", "rsi", "reasons", "rejected_by"]
        if c in table.columns
    ]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(table[cols].to_string(index=False))


def cmd_screen(args: argparse.Namespace) -> None:
    from .data import fetch_many
    from .screener import screen

    cfg = load_config(args.config)
    codes = [str(c) for c in cfg.get("watchlist", [])]
    period = cfg.get("data", {}).get("period", "1y")
    print(f"取得中: {len(codes)} 銘柄 (period={period}) ...")
    price = fetch_many(codes, period=period)
    table, _ = screen(price, cfg, detected_only=args.detected_only)
    print("\n=== スクリーニング結果（スコア降順） ===")
    _print_table(table)


def cmd_backtest(args: argparse.Namespace) -> None:
    from .backtest import backtest_many
    from .data import fetch_many

    cfg = load_config(args.config)
    codes = [str(c) for c in cfg.get("watchlist", [])]
    print(f"取得中: {len(codes)} 銘柄 ...")
    price = fetch_many(codes, period=cfg.get("data", {}).get("period", "2y"))
    _, summary = backtest_many(price, cfg, horizon=args.horizon)
    _print_summary(summary, args.horizon)


def cmd_demo(args: argparse.Namespace) -> None:
    """ネットワーク不要の合成データで検知ロジックの挙動を見る。"""
    from .synthetic import demo_universe
    from .backtest import backtest_many
    from .screener import screen

    cfg = load_config(args.config)
    price = demo_universe()
    print("=== デモ：合成データのスクリーニング ===")
    table, _ = screen(price, cfg)
    _print_table(table)

    print("\n=== デモ：バックテスト（horizon=20日） ===")
    _, summary = backtest_many(price, cfg, horizon=20)
    _print_summary(summary, 20)


def _print_summary(summary: dict[str, Any], horizon: int) -> None:
    if summary.get("n", 0) == 0:
        print("データ不足で集計できませんでした。")
        return
    print(f"\n[{horizon}営業日先リターンの集計] サンプル数={summary['n']}")
    print(f"  全 日 平均: {summary['all']['mean_fwd']}%  勝率 {summary['all']['win_rate']}")
    print(
        f"  検知日 平均: {summary['detected']['mean_fwd']}%  "
        f"勝率 {summary['detected']['win_rate']}  (n={summary['detected']['n']})"
    )
    print(f"  エッジ（検知 - 全体）: {summary['edge']}%")
    print("  スコア帯別の平均将来リターン:")
    for bucket, v in summary["by_score_bucket"].items():
        print(f"    {bucket}: 平均 {v['mean']}%  (n={v['count']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="日本株 検知システム")
    parser.add_argument("--config", default=None, help="config.yaml のパス")
    sub = parser.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="監視リストをスクリーニング")
    p_screen.add_argument("--detected-only", action="store_true")
    p_screen.set_defaults(func=cmd_screen)

    p_bt = sub.add_parser("backtest", help="ロジックを過去データで検証")
    p_bt.add_argument("--horizon", type=int, default=20)
    p_bt.set_defaults(func=cmd_backtest)

    p_demo = sub.add_parser("demo", help="合成データでロジックを確認（通信不要）")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
