#!/usr/bin/env python3
"""
Phase 2/3 验收脚本

用途：用落盘历史判断「规则新闻 / CFTC / 4h共振」是否值得继续占 ensemble 权重。

用法：
  python scripts/phase_validation.py              # 先记今日快照，再出验收报告
  python scripts/phase_validation.py --log-only   # 只记快照
  python scripts/phase_validation.py --eval-only  # 只评估
  python scripts/phase_validation.py --json       # 机器可读 JSON

验收标准（来自规划）：
  Phase2 新闻：窗口≥14天，sentiment 与次日金价涨跌相关 > 0.30 → KEEP
  Phase3 CFTC：至少 4 个周报点；MM 净持仓周变化方向与随后 5 日金价同向率 ≥ 55% → KEEP
  Phase3 4h ：至少 20 条日样本；AGREE_* 时 1d 方向命中率 高于 MIXED/DIVERGE 至少 5pt → KEEP
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase_validation")


def _paths() -> Dict[str, Path]:
    from config import OUTPUT_DIR, BASE_DIR, NEWS_SENTIMENT
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "output": out,
        "base": Path(BASE_DIR),
        "sentiment_history": out / "sentiment_history.jsonl",
        "sentiment_daily": out / "sentiment_daily.json",
        "cftc_daily": out / "cftc_daily.json",
        "cot_history": Path(BASE_DIR) / "data" / "cot_history.csv",
        "timeframe_daily": out / "timeframe_daily.json",
        "validation_log": out / "validation_daily.jsonl",
        "report_json": out / "phase_validation_report.json",
        "report_md": out / "phase_validation_report.md",
        "news_cfg": NEWS_SENTIMENT,
    }


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_gold_daily(days: int = 120) -> pd.DataFrame:
    """返回索引为日期、列 close 的日线。"""
    import yfinance as yf
    hist = yf.Ticker("GC=F").history(period=f"{max(days, 60)}d", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError("无法获取黄金日线")
    df = hist.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    out = df[["close"]].dropna().sort_index()
    out["ret_1d"] = out["close"].pct_change().shift(-1)  # 次日收益（今日行）
    out["ret_3d"] = out["close"].pct_change(3).shift(-3)
    out["ret_5d"] = out["close"].pct_change(5).shift(-5)
    out["dir_1d"] = np.sign(out["ret_1d"]).replace(0, np.nan)
    out["dir_5d"] = np.sign(out["ret_5d"]).replace(0, np.nan)
    return out


def log_today_snapshot(paths: Dict[str, Path]) -> Dict:
    """写入今日验收快照（幂等：同日覆盖重写文件尾部逻辑用去重）。"""
    today = datetime.now().strftime("%Y-%m-%d")

    # 尽量刷新当日信号（失败则用缓存）
    sentiment = _read_json(paths["sentiment_daily"]) or {}
    try:
        from data.news_sentiment import get_sentiment
        sentiment = get_sentiment(force_refresh=False) or sentiment
    except Exception as e:
        logger.warning(f"sentiment refresh skip: {e}")

    cftc = _read_json(paths["cftc_daily"]) or {}
    try:
        from data.cftc_collector import get_cftc_signal
        cftc = get_cftc_signal(force_refresh=False) or cftc
    except Exception as e:
        logger.warning(f"cftc refresh skip: {e}")

    timeframe = _read_json(paths["timeframe_daily"]) or {}
    try:
        from data.timeframe_analyzer import analyze_timeframes
        # 用缓存日线信号；没有则 NEUTRAL
        daily_sig = timeframe.get("daily_signal") or "NEUTRAL"
        timeframe = analyze_timeframes(daily_sig) or timeframe
    except Exception as e:
        logger.warning(f"timeframe refresh skip: {e}")

    gold_close = None
    try:
        g = fetch_gold_daily(days=10)
        gold_close = float(g["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"gold price skip: {e}")

    row = {
        "date": today,
        "logged_at": datetime.now().isoformat(),
        "gold_close": gold_close,
        "sentiment_score": sentiment.get("sentiment_score"),
        "sentiment_label": sentiment.get("sentiment_label"),
        "n_headlines": sentiment.get("n_headlines"),
        "cftc_score": cftc.get("cftc_score"),
        "cftc_label": cftc.get("cftc_label"),
        "cftc_report_date": cftc.get("report_date"),
        "mm_net": cftc.get("mm_net"),
        "wow_mm_net": cftc.get("wow_mm_net"),
        "timeframe_agreement": timeframe.get("timeframe_agreement"),
        "agreement_score": timeframe.get("agreement_score"),
        "daily_signal": timeframe.get("daily_signal"),
        "h4_score": (timeframe.get("h4") or {}).get("score"),
        "h4_label": (timeframe.get("h4") or {}).get("label"),
    }

    from data.snapshot_store import upsert_row
    backend = upsert_row(row, paths["validation_log"])
    row["_store"] = backend

    # 同步确保 sentiment_history 有今日行
    sh = [r for r in _read_jsonl(paths["sentiment_history"]) if r.get("date") != today]
    sh.append({
        "date": today,
        "sentiment_score": row["sentiment_score"],
        "n_headlines": row["n_headlines"],
        "label": row["sentiment_label"],
    })
    with paths["sentiment_history"].open("w", encoding="utf-8") as f:
        for r in sh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"logged snapshot {today}: sent={row['sentiment_score']} cftc={row['cftc_score']} tf={row['timeframe_agreement']}")
    return row


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_phase2(paths: Dict[str, Path], gold: pd.DataFrame) -> Dict:
    cfg = paths["news_cfg"]
    window = int(cfg.get("correlation_window_days", 14))
    thr = float(cfg.get("keep_threshold_corr", 0.30))

    from data.snapshot_store import load_rows
    # 合并 sentiment_history + 持久化验收快照
    rows = {}
    for r in _read_jsonl(paths["sentiment_history"]):
        if r.get("date") is not None and r.get("sentiment_score") is not None:
            rows[r["date"]] = float(r["sentiment_score"])
    for r in load_rows(paths["validation_log"]):
        if r.get("date") is not None and r.get("sentiment_score") is not None:
            rows[r["date"]] = float(r["sentiment_score"])

    pairs_x, pairs_y, used_dates = [], [], []
    for d, score in sorted(rows.items()):
        dt = pd.Timestamp(d)
        if dt not in gold.index:
            # 找最近交易日
            idx = gold.index.searchsorted(dt)
            if idx >= len(gold.index):
                continue
            dt = gold.index[idx]
        ret = gold.loc[dt, "ret_1d"]
        if pd.isna(ret):
            continue
        pairs_x.append(score)
        pairs_y.append(float(ret))
        used_dates.append(str(dt.date()))

    # 只用最近 window 个有效点
    if len(pairs_x) > window:
        pairs_x = pairs_x[-window:]
        pairs_y = pairs_y[-window:]
        used_dates = used_dates[-window:]

    n = len(pairs_x)
    corr = _pearson(np.array(pairs_x), np.array(pairs_y)) if n else None

    if n < window:
        verdict = "WAIT"
        reason = f"有效样本 {n}/{window}，继续积累"
    elif corr is None:
        verdict = "WAIT"
        reason = "方差过小，无法计算相关"
    elif abs(corr) > thr:
        verdict = "KEEP"
        reason = f"|corr|={abs(corr):.3f} > {thr}"
    else:
        verdict = "DROP"
        reason = f"|corr|={abs(corr):.3f} ≤ {thr}，建议降权/关闭 NEWS_SENTIMENT"

    return {
        "phase": "phase2_news_sentiment",
        "verdict": verdict,
        "reason": reason,
        "n_samples": n,
        "required_samples": window,
        "corr_vs_next_day_return": None if corr is None else round(corr, 4),
        "threshold_abs_corr": thr,
        "sample_dates": used_dates[-5:],
    }


def evaluate_phase3_cftc(paths: Dict[str, Path], gold: pd.DataFrame) -> Dict:
    try:
        from config import CFTC
        min_points = int(CFTC.get("validation_min_weeks", 4))
        hit_thr = float(CFTC.get("validation_hit_rate", 0.55))
    except Exception:
        min_points, hit_thr = 4, 0.55

    cot_path = paths["cot_history"]
    if not cot_path.exists():
        return {
            "phase": "phase3_cftc",
            "verdict": "WAIT",
            "reason": "尚无 cot_history.csv",
            "n_samples": 0,
            "required_samples": min_points,
        }

    cot = pd.read_csv(cot_path)
    if cot.empty or "report_date" not in cot.columns:
        return {
            "phase": "phase3_cftc",
            "verdict": "WAIT",
            "reason": "cot_history 为空",
            "n_samples": 0,
            "required_samples": min_points,
        }

    cot = cot.sort_values("report_date").reset_index(drop=True)
    cot["mm_net"] = pd.to_numeric(cot["mm_net"], errors="coerce")
    cot["wow"] = cot["mm_net"].diff()

    hits, total, details = 0, 0, []
    for i in range(1, len(cot)):
        row = cot.iloc[i]
        wow = row["wow"]
        if pd.isna(wow) or wow == 0:
            continue
        rd = pd.Timestamp(row["report_date"])
        # 报告日后约 5 个交易日收益方向
        # 找报告日及之后的交易日
        idx = gold.index.searchsorted(rd)
        if idx >= len(gold.index) - 5:
            continue
        # 用报告日对应交易日的 ret_5d（若缺失则用 close[t+5]/close[t]-1）
        t0 = gold.index[idx]
        if idx + 5 < len(gold.index):
            fwd = float(gold["close"].iloc[idx + 5] / gold["close"].iloc[idx] - 1)
        else:
            continue
        agree = (wow > 0 and fwd > 0) or (wow < 0 and fwd < 0)
        total += 1
        hits += int(agree)
        details.append({
            "report_date": str(row["report_date"]),
            "wow_mm_net": int(wow),
            "fwd_5d_return": round(fwd, 5),
            "agree": agree,
        })

    if total < min_points:
        verdict = "WAIT"
        reason = f"有效周报样本 {total}/{min_points}（需跨周积累）"
        hit_rate = None
    else:
        hit_rate = hits / total
        if hit_rate >= hit_thr:
            verdict = "KEEP"
            reason = f"同向率 {hit_rate:.1%} ≥ {hit_thr:.0%}"
        else:
            verdict = "DROP"
            reason = f"同向率 {hit_rate:.1%} < {hit_thr:.0%}，建议降权 CFTC"

    return {
        "phase": "phase3_cftc",
        "verdict": verdict,
        "reason": reason,
        "n_samples": total,
        "required_samples": min_points,
        "hit_rate_vs_fwd_5d": None if hit_rate is None else round(hit_rate, 4),
        "threshold_hit_rate": hit_thr,
        "recent": details[-5:],
    }


def evaluate_phase3_timeframe(paths: Dict[str, Path], gold: pd.DataFrame) -> Dict:
    try:
        from config import TIMEFRAME
        min_n = int(TIMEFRAME.get("validation_min_days", 20))
        edge_thr = float(TIMEFRAME.get("validation_edge", 0.05))
    except Exception:
        min_n, edge_thr = 20, 0.05

    from data.snapshot_store import load_rows
    rows = load_rows(paths["validation_log"])
    # 也允许只有 timeframe_daily 的单点（不够评估）
    samples = []
    for r in rows:
        d = r.get("date")
        agree = r.get("timeframe_agreement")
        daily = (r.get("daily_signal") or "").upper()
        if not d or not agree or agree in ("UNKNOWN", "NO_DATA", "ERROR", "DISABLED"):
            continue
        dt = pd.Timestamp(d)
        if dt not in gold.index:
            idx = gold.index.searchsorted(dt)
            if idx >= len(gold.index):
                continue
            dt = gold.index[idx]
        ret = gold.loc[dt, "ret_1d"]
        if pd.isna(ret):
            continue
        # 日线信号方向
        if daily in ("BUY", "STRONG_BUY"):
            pred_dir = 1
        elif daily in ("SELL", "STRONG_SELL"):
            pred_dir = -1
        else:
            continue
        hit = int(np.sign(ret) == pred_dir)
        samples.append({"date": str(dt.date()), "agreement": agree, "hit": hit, "daily": daily})

    if len(samples) < min_n:
        return {
            "phase": "phase3_timeframe",
            "verdict": "WAIT",
            "reason": f"有效日样本 {len(samples)}/{min_n}",
            "n_samples": len(samples),
            "required_samples": min_n,
        }

    def rate(subset):
        if not subset:
            return None
        return sum(s["hit"] for s in subset) / len(subset)

    agree = [s for s in samples if s["agreement"] in ("AGREE_BULL", "AGREE_BEAR")]
    other = [s for s in samples if s["agreement"] not in ("AGREE_BULL", "AGREE_BEAR")]
    r_agree = rate(agree)
    r_other = rate(other)

    if r_agree is None or r_other is None:
        verdict = "WAIT"
        reason = "AGREE 或 非AGREE 分组为空"
        edge = None
    else:
        edge = r_agree - r_other
        if edge >= edge_thr:
            verdict = "KEEP"
            reason = f"共振命中率 {r_agree:.1%} 比非共振高 {edge:.1%} (≥{edge_thr:.0%})"
        else:
            verdict = "DROP"
            reason = f"共振优势仅 {edge:.1%} < {edge_thr:.0%}，建议降权 TIMEFRAME"

    return {
        "phase": "phase3_timeframe",
        "verdict": verdict,
        "reason": reason,
        "n_samples": len(samples),
        "required_samples": min_n,
        "hit_rate_agree": None if r_agree is None else round(r_agree, 4),
        "hit_rate_other": None if r_other is None else round(r_other, 4),
        "edge": None if edge is None else round(edge, 4),
        "n_agree": len(agree),
        "n_other": len(other),
        "threshold_edge": edge_thr,
    }


def build_report(paths: Dict[str, Path]) -> Dict:
    gold = fetch_gold_daily(days=180)
    p2 = evaluate_phase2(paths, gold)
    p3c = evaluate_phase3_cftc(paths, gold)
    p3t = evaluate_phase3_timeframe(paths, gold)

    actions = []
    for block in (p2, p3c, p3t):
        if block["verdict"] == "DROP":
            if block["phase"].startswith("phase2"):
                actions.append("建议：config.NEWS_SENTIMENT.enabled=False 或 weight→0")
            elif "cftc" in block["phase"]:
                actions.append("建议：config.CFTC.enabled=False 或 weight→0")
            elif "timeframe" in block["phase"]:
                actions.append("建议：config.TIMEFRAME.enabled=False 或 weight→0")
        elif block["verdict"] == "KEEP":
            actions.append(f"保留 {block['phase']}")
        else:
            actions.append(f"继续观察 {block['phase']}：{block['reason']}")

    from data.snapshot_store import load_rows, persistence_backend
    report = {
        "generated_at": datetime.now().isoformat(),
        "gold_last_close": float(gold["close"].iloc[-1]),
        "snapshot_days": len(load_rows(paths["validation_log"])),
        "persistence": persistence_backend(),
        "phases": [p2, p3c, p3t],
        "actions": actions,
        "summary": {
            "keep": [b["phase"] for b in (p2, p3c, p3t) if b["verdict"] == "KEEP"],
            "drop": [b["phase"] for b in (p2, p3c, p3t) if b["verdict"] == "DROP"],
            "wait": [b["phase"] for b in (p2, p3c, p3t) if b["verdict"] == "WAIT"],
        },
    }
    return report


def render_markdown(report: Dict) -> str:
    lines = [
        f"# Phase 2/3 验收报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 金价参考：${report['gold_last_close']:.2f}",
        "",
        "## 结论摘要",
        "",
        f"- KEEP：{', '.join(report['summary']['keep']) or '无'}",
        f"- DROP：{', '.join(report['summary']['drop']) or '无'}",
        f"- WAIT：{', '.join(report['summary']['wait']) or '无'}",
        "",
        "## 分项",
        "",
    ]
    for p in report["phases"]:
        lines += [
            f"### {p['phase']}",
            f"- 判定：**{p['verdict']}**",
            f"- 原因：{p['reason']}",
            f"- 样本：{p.get('n_samples')}/{p.get('required_samples')}",
            "",
        ]
        for k, v in p.items():
            if k in ("phase", "verdict", "reason", "n_samples", "required_samples", "sample_dates", "recent"):
                continue
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines += ["## 建议动作", ""]
    for a in report["actions"]:
        lines.append(f"- {a}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 2/3 验收")
    parser.add_argument("--log-only", action="store_true", help="只写今日快照")
    parser.add_argument("--eval-only", action="store_true", help="只评估不写快照")
    parser.add_argument("--json", action="store_true", help="stdout 输出 JSON")
    args = parser.parse_args()

    paths = _paths()
    snapshot = None
    if not args.eval_only:
        snapshot = log_today_snapshot(paths)
        if args.log_only:
            out = {"snapshot": snapshot}
            print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else f"logged {snapshot['date']}")
            return

    report = build_report(paths)
    paths["report_json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(report)
    paths["report_md"].write_text(md, encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(md)
        print(f"\n报告已写入:\n  {paths['report_md']}\n  {paths['report_json']}")


if __name__ == "__main__":
    main()
