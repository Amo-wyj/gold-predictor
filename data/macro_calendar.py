"""
财经日历：拉取本周高影响事件（默认 ForexFactory 镜像 JSON）
过滤美国 + High impact，供预警与 /api/calendar 使用。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

FF_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# 标题关键词 → 内部事件类型
EVENT_KEYWORDS = [
    ("FOMC", "fomc_meeting", "美联储FOMC会议"),
    ("Fed Interest Rate", "fomc_meeting", "美联储利率决议"),
    ("Federal Funds Rate", "fomc_meeting", "美联储利率决议"),
    ("Non-Farm", "nonfarm_payroll", "非农就业报告"),
    ("Nonfarm", "nonfarm_payroll", "非农就业报告"),
    ("NFP", "nonfarm_payroll", "非农就业报告"),
    ("CPI", "cpi_release", "美国CPI数据"),
    ("Core CPI", "cpi_release", "美国核心CPI"),
    ("PCE", "pce_release", "美国PCE物价指数"),
    ("Core PCE", "pce_release", "美国核心PCE"),
    ("GDP", "gdp_release", "美国GDP数据"),
    ("Powell", "speech_powell", "鲍威尔讲话"),
    ("FOMC Member", "speech_powell", "联储官员讲话"),
]


def _http_get_json(url: str, timeout: int = 15):
    req = Request(url, headers={"User-Agent": "gold-predictor/1.0 (+macro-calendar)"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        # 2026-09-13T04:15:00-04:00
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _classify(title: str) -> Optional[tuple]:
    t = title or ""
    for key, etype, cname in EVENT_KEYWORDS:
        if key.lower() in t.lower():
            return etype, cname
    return None


def fetch_us_high_impact(days_ahead: int = 7) -> List[Dict]:
    """返回未来 days_ahead 天内美国高影响事件。"""
    try:
        raw = _http_get_json(FF_THIS_WEEK)
    except Exception as e:
        logger.warning(f"[Calendar] 拉取失败: {e}")
        return []

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    out: List[Dict] = []

    for item in raw if isinstance(raw, list) else []:
        country = (item.get("country") or "").strip().upper()
        if country not in ("USD", "US"):
            continue

        impact = (item.get("impact") or "").strip().lower()
        title = item.get("title") or ""
        classified = _classify(title)

        # High 全收；Medium 仅收我们关心的关键词事件
        if impact == "high":
            pass
        elif impact == "medium" and classified:
            pass
        else:
            continue

        dt = _parse_dt(item.get("date") or "")
        if dt is None:
            continue
        dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if dt_utc < now - timedelta(hours=6) or dt_utc > end:
            continue

        etype, cname = classified if classified else ("other_high", title)
        tips = {
            "fomc_meeting": "会议前后波动加大；降息/鸽派通常利多黄金",
            "nonfarm_payroll": "强于预期→美元涨、金价承压；弱于预期则相反",
            "cpi_release": "通胀超预期可能强化加息预期→短期压金价",
            "pce_release": "联储核心通胀指标，影响降息路径",
            "gdp_release": "增长过强可能延后降息→金价承压",
            "speech_powell": "关注政策指引，鸽派偏多金、鹰派偏空金",
        }.get(etype, "重大数据公布前后注意点差与波动")

        out.append({
            "event_type": etype,
            "event_name": cname if classified else title,
            "title_raw": title,
            "event_date": dt_utc.isoformat(),
            "event_date_local": dt.isoformat(),
            "impact": item.get("impact"),
            "forecast": item.get("forecast") or "",
            "previous": item.get("previous") or "",
            "country": item.get("country"),
            "preparation": tips,
            "days_until": round((dt_utc - now).total_seconds() / 86400, 2),
        })

    out.sort(key=lambda x: x["event_date"])
    return out


def get_calendar(days_ahead: int = 7, force_refresh: bool = False) -> Dict:
    from config import OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, "macro_calendar.json")

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            ts = cached.get("updated_at")
            if ts:
                age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                if age < 6 * 3600:
                    return cached
        except Exception:
            pass

    events = fetch_us_high_impact(days_ahead=days_ahead)
    result = {
        "events": events,
        "n_events": len(events),
        "days_ahead": days_ahead,
        "source": "forexfactory_mirror",
        "updated_at": datetime.now().isoformat(),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(get_calendar(force_refresh=True), ensure_ascii=False, indent=2))
