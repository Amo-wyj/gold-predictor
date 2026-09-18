"""
Phase 3：CFTC 持仓（Disaggregated Futures）
- 源：https://www.cftc.gov/dea/newcot/c_disagg.txt
- 标的：GOLD - COMMODITY EXCHANGE INC.
- 用 Managed Money 净持仓及周环比变化 → score ∈ [-1,1]
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

CFTC_DISAGG_URL = "https://www.cftc.gov/dea/newcot/c_disagg.txt"
GOLD_NAME_PREFIX = "GOLD - COMMODITY EXCHANGE INC."


def _http_get(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "gold-predictor/1.0 (+phase3-cftc)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _to_int(x: str) -> int:
    return int(str(x).strip().replace(",", "") or 0)


def parse_gold_row(text: str) -> Optional[Dict]:
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        name = row[0].strip().strip('"')
        if name != GOLD_NAME_PREFIX:
            continue
        # Disagg Futures-Only 固定列（CFTC 文档）
        # 7 OI | 8-9 ProdMerc L/S | 10-12 Swap L/S/Spr | 13-15 MM L/S/Spr
        oi = _to_int(row[7])
        mm_long = _to_int(row[13])
        mm_short = _to_int(row[14])
        mm_spread = _to_int(row[15])
        pm_long = _to_int(row[8])
        pm_short = _to_int(row[9])
        mm_net = mm_long - mm_short
        pm_net = pm_long - pm_short
        return {
            "market": name,
            "report_date": row[2].strip(),
            "as_of_yymmdd": row[1].strip(),
            "open_interest": oi,
            "mm_long": mm_long,
            "mm_short": mm_short,
            "mm_spread": mm_spread,
            "mm_net": mm_net,
            "pm_net": pm_net,
            "mm_net_pct_oi": round(mm_net / oi, 4) if oi else 0.0,
        }
    return None


class CFTCCollector:
    def __init__(self, data_dir: Optional[str] = None):
        from config import BASE_DIR, OUTPUT_DIR
        self.data_dir = data_dir or os.path.join(BASE_DIR, "data")
        self.output_dir = OUTPUT_DIR
        self.history_path = os.path.join(self.data_dir, "cot_history.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_latest(self) -> Optional[Dict]:
        try:
            text = _http_get(CFTC_DISAGG_URL)
            row = parse_gold_row(text)
            if not row:
                logger.warning("[CFTC] 未找到 GOLD COMEX 行")
                return None
            logger.info(
                f"[CFTC] {row['report_date']} MM_net={row['mm_net']} "
                f"({row['mm_net_pct_oi']*100:.1f}% OI)"
            )
            return row
        except Exception as e:
            logger.warning(f"[CFTC] 拉取失败: {e}")
            return None

    def _load_history(self) -> list:
        if not os.path.exists(self.history_path):
            return []
        rows = []
        with open(self.history_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows

    def _append_history(self, row: Dict) -> None:
        fieldnames = [
            "report_date", "open_interest", "mm_long", "mm_short", "mm_net",
            "mm_net_pct_oi", "pm_net", "fetched_at",
        ]
        exists = os.path.exists(self.history_path)
        # 同日去重：重写文件
        hist = [h for h in self._load_history() if h.get("report_date") != row["report_date"]]
        hist.append({
            "report_date": row["report_date"],
            "open_interest": row["open_interest"],
            "mm_long": row["mm_long"],
            "mm_short": row["mm_short"],
            "mm_net": row["mm_net"],
            "mm_net_pct_oi": row["mm_net_pct_oi"],
            "pm_net": row["pm_net"],
            "fetched_at": datetime.now().isoformat(),
        })
        hist.sort(key=lambda x: x["report_date"])
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(hist)

    def compute_signal(self, force_refresh: bool = True) -> Dict:
        """返回 cftc_score ∈ [-1,1] 与明细。"""
        latest = self.fetch_latest() if force_refresh else None
        if latest is None:
            # 尝试读缓存
            cached = self._load_json_cache()
            if cached:
                return cached
            return {
                "cftc_score": 0.0,
                "cftc_label": "NO_DATA",
                "report_date": None,
                "mm_net": None,
                "updated_at": datetime.now().isoformat(),
            }

        self._append_history(latest)
        hist = self._load_history()
        prev = hist[-2] if len(hist) >= 2 else None

        oi = float(latest["open_interest"] or 1)
        mm_net = float(latest["mm_net"])
        level = mm_net / oi  # 净多占比

        if prev:
            prev_net = float(prev["mm_net"])
            wow = (mm_net - prev_net) / oi
            # 周变化主导，水平为辅
            score = max(-1.0, min(1.0, wow * 25.0 + level * 0.4))
            wow_contracts = int(mm_net - prev_net)
        else:
            wow = None
            wow_contracts = None
            score = max(-1.0, min(1.0, level * 1.2))

        if score >= 0.25:
            label = "MM_NET_BULLISH"
        elif score <= -0.25:
            label = "MM_NET_BEARISH"
        else:
            label = "MM_NET_NEUTRAL"

        result = {
            "cftc_score": round(float(score), 4),
            "cftc_label": label,
            "report_date": latest["report_date"],
            "open_interest": latest["open_interest"],
            "mm_long": latest["mm_long"],
            "mm_short": latest["mm_short"],
            "mm_net": latest["mm_net"],
            "mm_net_pct_oi": latest["mm_net_pct_oi"],
            "pm_net": latest["pm_net"],
            "wow_mm_net": wow_contracts,
            "wow_mm_net_pct_oi": round(wow, 4) if wow is not None else None,
            "history_points": len(hist),
            "note": "CFTC 延迟约 3 天，定位周度大方向",
            "updated_at": datetime.now().isoformat(),
            "source": "cftc_disagg",
        }
        self._save_json_cache(result)
        return result

    def _cache_path(self) -> str:
        return os.path.join(self.output_dir, "cftc_daily.json")

    def _save_json_cache(self, result: Dict) -> None:
        try:
            with open(self._cache_path(), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[CFTC] 写缓存失败: {e}")

    def _load_json_cache(self) -> Optional[Dict]:
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def get_cftc_signal(force_refresh: bool = False) -> Dict:
    c = CFTCCollector()
    if not force_refresh:
        cached = c._load_json_cache()
        if cached and cached.get("cftc_label") != "NO_DATA":
            return cached
    return c.compute_signal(force_refresh=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(get_cftc_signal(force_refresh=True), ensure_ascii=False, indent=2))
