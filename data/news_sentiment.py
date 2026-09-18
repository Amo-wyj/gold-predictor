"""
Phase 2：规则版新闻情绪
- 拉取财经 RSS（Google News / 可选备用源）
- 关键词极性打分（零 LLM）
- 输出日度 sentiment_score ∈ [-1, 1]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# 默认 RSS（Google News 稳定可用；财联社无稳定公开 RSS，用关键词检索覆盖）
DEFAULT_FEEDS = [
    {
        "name": "google_zh_gold",
        "url": "https://news.google.com/rss/search?"
        + urlencode({"q": "黄金 OR 金价 OR 美联储 OR 降息", "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-CN"}),
    },
    {
        "name": "google_en_gold",
        "url": "https://news.google.com/rss/search?"
        + urlencode({"q": "gold price OR XAU OR Fed OR inflation", "hl": "en-US", "gl": "US", "ceid": "US:en"}),
    },
]

# 极性词典（中英混合，覆盖金价/宏观/风险偏好；约 200 词）
BULLISH_TERMS = {
    # 中文看涨
    "降息": 1.2, "宽松": 1.1, "鸽派": 1.2, "避险": 1.0, "避险需求": 1.2,
    "买盘": 0.8, "抢购": 0.9, "上涨": 0.7, "大涨": 1.0, "飙升": 1.1,
    "突破": 0.8, "新高": 1.0, "创纪录": 1.0, "走强": 0.8, "反弹": 0.7,
    "看涨": 1.0, "利多": 1.0, "支撑": 0.5, "金价上涨": 1.2, "黄金上涨": 1.2,
    "地缘紧张": 0.9, "战争风险": 1.0, "冲突升级": 0.9, "制裁": 0.6,
    "美元走弱": 1.1, "美元下跌": 1.0, "实际利率下降": 1.2, "负利率": 0.9,
    "央行购金": 1.3, "增持黄金": 1.2, "储备多元化": 0.8, "去美元化": 0.9,
    "通胀升温": 0.7, "滞胀": 0.6, "避险资产": 1.0, "配置黄金": 1.0,
    "实物需求": 0.8, "ETF流入": 1.1, "资金流入": 0.8, "净买入": 0.9,
    # English bullish
    "rate cut": 1.2, "easing": 1.0, "dovish": 1.2, "safe haven": 1.1,
    "rally": 0.9, "surge": 1.0, "soar": 1.1, "record high": 1.0,
    "breakout": 0.8, "bullish": 1.0, "gold rises": 1.1, "gold up": 0.9,
    "weaker dollar": 1.1, "dollar falls": 1.0, "negative real yields": 1.2,
    "central bank buying": 1.3, "gold buying": 1.1, "etf inflows": 1.1,
    "geopolitical risk": 0.9, "war fears": 1.0, "sanctions": 0.5,
    "inflation fears": 0.7, "flight to safety": 1.1, "risk-off": 0.8,
}

BEARISH_TERMS = {
    # 中文看跌
    "加息": 1.2, "紧缩": 1.1, "鹰派": 1.2, "美元走强": 1.1, "美元上涨": 1.0,
    "抛售": 1.0, "下跌": 0.7, "大跌": 1.0, "暴跌": 1.2, "回落": 0.6,
    "走弱": 0.8, "看跌": 1.0, "利空": 1.0, "承压": 0.7, "跌破": 0.9,
    "金价下跌": 1.2, "黄金下跌": 1.2, "获利了结": 0.7, "风险偏好回升": 0.8,
    "实际利率上升": 1.2, "债市收益率上升": 0.9, "美债收益率上行": 1.0,
    "ETF流出": 1.1, "资金流出": 0.8, "净卖出": 0.9, "减持黄金": 1.1,
    "避险情绪降温": 0.9, "和平谈判": 0.6, "停火": 0.5, "风险资产": 0.5,
    "强势美元": 1.1, "加息预期": 1.0, "推迟降息": 1.0, "更高更久": 1.1,
    # English bearish
    "rate hike": 1.2, "tightening": 1.1, "hawkish": 1.2, "stronger dollar": 1.1,
    "selloff": 1.0, "sell-off": 1.0, "plunge": 1.2, "slump": 1.0, "drop": 0.6,
    "bearish": 1.0, "gold falls": 1.1, "gold down": 0.9, "profit taking": 0.7,
    "higher for longer": 1.1, "delayed cuts": 1.0, "real yields rise": 1.2,
    "etf outflows": 1.1, "risk-on": 0.8, "dollar rally": 1.0, "treasury yields rise": 0.9,
    "ceasefire": 0.5, "peace talks": 0.5, "gold selling": 1.0,
}

NEGATORS = ("不", "未", "无", "否认", "并非", "没有", "难言", "not", "no ", "never", "without")


def _http_get(url: str, timeout: int = 12) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "gold-predictor/1.0 (+phase2-news-sentiment)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _local_text(el: Optional[ET.Element]) -> str:
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def parse_rss(xml_text: str, source: str) -> List[Dict]:
    items: List[Dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"[News] RSS 解析失败 {source}: {e}")
        return items

    # RSS 2.0
    for item in root.findall(".//item"):
        title = _local_text(item.find("title"))
        link = _local_text(item.find("link"))
        pub = _local_text(item.find("pubDate"))
        desc = _local_text(item.find("description"))
        if title:
            items.append({
                "title": title,
                "link": link,
                "published": pub,
                "summary": re.sub(r"<[^>]+>", " ", desc)[:300],
                "source": source,
            })

    # Atom
    if not items:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//a:entry", ns) or root.findall(".//entry"):
            title = _local_text(entry.find("a:title", ns)) or _local_text(entry.find("title"))
            link_el = entry.find("a:link", ns) or entry.find("link")
            link = ""
            if link_el is not None:
                link = link_el.get("href") or _local_text(link_el)
            pub = (
                _local_text(entry.find("a:updated", ns))
                or _local_text(entry.find("updated"))
                or _local_text(entry.find("a:published", ns))
            )
            if title:
                items.append({
                    "title": title,
                    "link": link,
                    "published": pub,
                    "summary": "",
                    "source": source,
                })
    return items


def score_text(text: str) -> Tuple[float, List[str]]:
    """对单条标题/摘要打分，返回 (score[-1,1], matched_terms)。"""
    if not text:
        return 0.0, []
    lower = text.lower()
    bull = 0.0
    bear = 0.0
    matched: List[str] = []
    claimed: List[str] = []

    def _already_covered(term: str) -> bool:
        t = term.lower()
        return any((t in c or c in t) and c != t for c in claimed)

    # 合并后按长度优先，避免短词误伤长词
    pooled = (
        [(t, w, "bull") for t, w in BULLISH_TERMS.items()]
        + [(t, w, "bear") for t, w in BEARISH_TERMS.items()]
    )
    pooled.sort(key=lambda x: len(x[0]), reverse=True)

    for term, w, polar in pooled:
        key = term.lower()
        hay = lower if re.search(r"[a-z]", term) else text
        if key not in hay and term not in text:
            continue
        if _already_covered(term):
            continue
        idx = hay.find(key) if key in hay else text.find(term)
        window = (hay[max(0, idx - 4):idx] if key in hay else text[max(0, idx - 4):idx]).lower()
        negated = any(n.strip() in window for n in NEGATORS)
        if polar == "bull":
            if negated:
                bear += w
                matched.append(f"-{term}")
            else:
                bull += w
                matched.append(f"+{term}")
        else:
            if negated:
                bull += w
                matched.append(f"+!{term}")
            else:
                bear += w
                matched.append(f"-{term}")
        claimed.append(key)

    raw = bull - bear
    score = float(max(-1.0, min(1.0, raw / 4.0)))
    return score, matched[:12]


class NewsSentimentAnalyzer:
    """规则版新闻情绪分析器。"""

    def __init__(self, feeds: Optional[List[Dict]] = None, data_dir: Optional[str] = None):
        from config import BASE_DIR, OUTPUT_DIR

        self.feeds = feeds or DEFAULT_FEEDS
        self.data_dir = data_dir or os.path.join(BASE_DIR, "data", "news")
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_headlines(self, max_per_feed: int = 30) -> List[Dict]:
        all_items: List[Dict] = []
        for feed in self.feeds:
            name = feed.get("name", "feed")
            url = feed.get("url")
            if not url:
                continue
            try:
                xml_text = _http_get(url)
                items = parse_rss(xml_text, name)[:max_per_feed]
                all_items.extend(items)
                logger.info(f"[News] {name}: {len(items)} 条")
            except (URLError, HTTPError, TimeoutError, OSError) as e:
                logger.warning(f"[News] 拉取失败 {name}: {e}")
            except Exception as e:
                logger.warning(f"[News] 异常 {name}: {e}")
            time.sleep(0.2)
        # 去重
        seen = set()
        unique = []
        for it in all_items:
            key = hashlib.md5((it.get("title") or "").encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)
        return unique

    def save_raw(self, items: List[Dict]) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(self.data_dir, f"news_{day}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for it in items:
                row = dict(it)
                row["fetched_at"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def compute_daily_score(self, items: Optional[List[Dict]] = None) -> Dict:
        if items is None:
            items = self.fetch_headlines()
        if items:
            try:
                self.save_raw(items)
            except Exception as e:
                logger.warning(f"[News] 落盘失败: {e}")

        if not items:
            result = {
                "sentiment_score": 0.0,
                "sentiment_label": "NO_DATA",
                "n_headlines": 0,
                "bullish_share": 0.0,
                "bearish_share": 0.0,
                "top_matches": [],
                "sample_titles": [],
                "updated_at": datetime.now().isoformat(),
                "source": "rules_rss",
            }
            self._persist(result)
            return result

        scores = []
        all_matches: List[str] = []
        for it in items:
            text = f"{it.get('title', '')} {it.get('summary', '')}"
            s, matched = score_text(text)
            it["score"] = s
            it["matches"] = matched
            scores.append(s)
            all_matches.extend(matched)

        avg = float(sum(scores) / len(scores))
        bull_n = sum(1 for s in scores if s > 0.15)
        bear_n = sum(1 for s in scores if s < -0.15)
        n = len(scores)

        if avg >= 0.25:
            label = "BULLISH"
        elif avg <= -0.25:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # 高频命中词
        from collections import Counter
        top = [w for w, _ in Counter(all_matches).most_common(8)]

        result = {
            "sentiment_score": round(max(-1.0, min(1.0, avg)), 4),
            "sentiment_label": label,
            "n_headlines": n,
            "bullish_share": round(bull_n / n, 3),
            "bearish_share": round(bear_n / n, 3),
            "top_matches": top,
            "sample_titles": [it.get("title", "")[:80] for it in items[:5]],
            "updated_at": datetime.now().isoformat(),
            "source": "rules_rss",
        }
        self._persist(result)
        self._append_history(result)
        return result

    def _persist(self, result: Dict) -> None:
        path = os.path.join(self.output_dir, "sentiment_daily.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[News] 写 sentiment_daily.json 失败: {e}")

    def _append_history(self, result: Dict) -> None:
        """供 Phase2 验收：2 周相关性观察。"""
        path = os.path.join(self.output_dir, "sentiment_history.jsonl")
        try:
            row = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sentiment_score": result.get("sentiment_score"),
                "n_headlines": result.get("n_headlines"),
                "label": result.get("sentiment_label"),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def load_cached(output_dir: Optional[str] = None) -> Optional[Dict]:
        from config import OUTPUT_DIR
        path = os.path.join(output_dir or OUTPUT_DIR, "sentiment_daily.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def get_sentiment(force_refresh: bool = False, max_age_sec: int = 3600) -> Dict:
    """供 ensemble / dashboard 调用的统一入口。"""
    cached = NewsSentimentAnalyzer.load_cached()
    if cached and not force_refresh:
        try:
            ts = cached.get("updated_at")
            if ts:
                # 简单解析 ISO
                age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                if age < max_age_sec and cached.get("n_headlines", 0) > 0:
                    return cached
        except Exception:
            pass
    try:
        return NewsSentimentAnalyzer().compute_daily_score()
    except Exception as e:
        logger.warning(f"[News] get_sentiment 失败: {e}")
        return cached or {
            "sentiment_score": 0.0,
            "sentiment_label": "ERROR",
            "n_headlines": 0,
            "error": str(e)[:120],
            "updated_at": datetime.now().isoformat(),
            "source": "rules_rss",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = get_sentiment(force_refresh=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
