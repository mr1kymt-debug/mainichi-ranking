# -*- coding: utf-8 -*-
"""
毎朝ランキング：毎朝の「数字集め」の部品

やること
  1. 日本株（日経平均の225社）とアメリカ株（S&P500の約500社）の一覧を読む
  2. 株情報サービス（Yahoo Finance）から、次の3つの数字を集める
       ・業績の伸び      … 直近の四半期の売上が、前の年の同じ時期より何％伸びたか
       ・アナリスト評価  … アナリストの「買い」の強さを5点満点にしたもの（5に近いほど強い買い）
       ・株価の勢い      … 直近およそ3か月（63営業日）で株価が何％動いたか
  3. 3つの数字それぞれの「全体の中での位置」を同じ重さで平均して、100点満点の点数にする
  4. 日本株・アメリカ株それぞれの順位を作り、app.html の「株の数字のかたまり」に入れて書き出す

うまく集められなかった日（集められた株が少なすぎる日）は、何も書き出さずに「失敗」で終わります。
（前の日の画面がそのまま残り、画面の上に赤い文字で「今朝の更新に失敗しました」と出ます）
"""
import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
JST = timezone(timedelta(hours=9))

KEYS = ("growth", "rating", "momentum")
US_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
MIN_ANALYSTS = 3          # アナリストが3人未満の株は、アナリスト評価を「データなし」にする
MOMENTUM_DAYS = 63        # 株価の勢い＝およそ3か月（63営業日）の値動き
MIN_SUCCESS_RATE = 0.6    # 日本株・アメリカ株それぞれ、6割以上の株で数字がそろわなければ「失敗」


def log(msg):
    print(msg, flush=True)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def clean_name(s):
    """全角の英数字を半角にして、名前の中の空白を取る（「Ｊ Ｔ」→「JT」）"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


# ---------------------------------------------------------------- 一覧を読む
def load_jp(path=None):
    path = path or os.path.join(HERE, "nikkei225.csv")
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            code = r["code"].strip()
            out.append({"ticker": code + ".T", "code": code, "name": clean_name(r["name"])})
    return out


def load_us(fallback=None):
    fallback = fallback or os.path.join(HERE, "sp500_fallback.csv")
    text = None
    try:
        with urllib.request.urlopen(US_LIST_URL, timeout=30) as res:
            text = res.read().decode("utf-8")
        log("アメリカ株の一覧：最新の一覧を読みました")
    except Exception as e:  # 読めなければ、手元に保存してある一覧を使う
        log("アメリカ株の一覧：最新の一覧が読めなかったので、保存してある一覧を使います（%s）" % e)
        with open(fallback, encoding="utf-8", newline="") as f:
            text = f.read()
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        sym = r["Symbol"].strip()
        out.append({"ticker": sym.replace(".", "-"), "code": sym, "name": r["Security"].strip()})
    return out


# ---------------------------------------------------------------- 数字を集める
def fetch_metrics_yahoo(tickers):
    """Yahoo Finance から3つの数字を集める。読めなかった数字は None のままにする。"""
    import yfinance as yf

    out = {t: {"growth": None, "rating": None, "momentum": None} for t in tickers}

    # (1) 株価の勢い：まとめて株価の履歴を読む
    got = 0
    bad_chunks = 0
    for i in range(0, len(tickers), 100):
        if bad_chunks >= 2:
            log("株価が続けて読めないため、株価の読み込みをやめます")
            break
        chunk = tickers[i:i + 100]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period="6mo", interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                log("株価の読み込みに失敗（%d回目）：%s" % (attempt + 1, e))
                time.sleep(3 * (attempt + 1))
        if df is None or getattr(df, "empty", True):
            bad_chunks += 1
            continue
        bad_chunks = 0
        for t in chunk:
            try:
                try:
                    closes = df[t]["Close"].dropna()
                except Exception:
                    closes = df["Close"].dropna()
                closes = [float(x) for x in closes.tolist()]
                if len(closes) > MOMENTUM_DAYS and closes[-1 - MOMENTUM_DAYS] > 0:
                    mom = (closes[-1] / closes[-1 - MOMENTUM_DAYS] - 1) * 100
                    if is_num(mom):
                        out[t]["momentum"] = round(max(-999.9, min(999.9, mom)), 1)
                        got += 1
            except Exception:
                pass
        time.sleep(1)
    log("株価の勢い：%d / %d 社で読めました" % (got, len(tickers)))

    # (2) 業績の伸び・アナリスト評価：1社ずつ読む
    import threading
    lock = threading.Lock()
    tally = {"ok": 0, "ng": 0}
    stop = threading.Event()

    def fetch_info(t):
        if stop.is_set():
            return {}
        for attempt in range(3):
            try:
                info = yf.Ticker(t).info or {}
                with lock:
                    tally["ok"] += 1
                return info
            except Exception:
                time.sleep(2 * (attempt + 1))
        with lock:
            tally["ng"] += 1
            if tally["ng"] >= 30 and tally["ok"] == 0:
                stop.set()          # 30社続けて1つも読めなければ、つながっていないと判断してやめる
                log("株情報サービスにつながらないため、業績・アナリスト評価の読み込みをやめます")
        return {}

    with ThreadPoolExecutor(max_workers=6) as ex:
        infos = list(ex.map(fetch_info, tickers))

    g = r = 0
    for t, info in zip(tickers, infos):
        rg = info.get("revenueGrowth")
        if is_num(rg):
            out[t]["growth"] = round(max(-999.9, min(999.9, rg * 100)), 1)
            g += 1
        rm = info.get("recommendationMean")
        n = info.get("numberOfAnalystOpinions")
        if is_num(rm) and is_num(n) and n >= MIN_ANALYSTS:
            out[t]["rating"] = round(max(1.0, min(5.0, 6 - rm)), 1)   # 1(強い買い)〜5(売り) → 5(強い買い)〜1
            r += 1
    log("業績の伸び：%d 社 / アナリスト評価：%d 社 で読めました" % (g, r))
    return out


# ---------------------------------------------------------------- 点数をつける
def score_market(items, metrics):
    """3つのうち2つ以上そろう株だけを対象に、100点満点の点数をつけて、点数の大きい順に返す。"""
    rows = []
    for it in items:
        m = metrics.get(it["ticker"]) or {}
        row = {"name": it["name"], "code": it["code"]}
        for k in KEYS:
            row[k] = m.get(k) if is_num(m.get(k)) else None
        if sum(row[k] is not None for k in KEYS) >= 2:
            rows.append(row)

    for k in KEYS:
        have = sorted([r for r in rows if r[k] is not None], key=lambda r: -r[k])
        for idx, r in enumerate(have):
            r["_p_" + k] = 1 - idx / len(have)

    for i, r in enumerate(rows):
        ps = [r["_p_" + k] for k in KEYS if r[k] is not None]
        r["score"] = int(round(sum(ps) / len(ps) * 100))
        r["_i"] = i
    rows.sort(key=lambda r: (-r["score"], r["_i"]))
    return [{"name": r["name"], "code": r["code"], "score": r["score"],
             "growth": r["growth"], "rating": r["rating"], "momentum": r["momentum"]} for r in rows]


# ---------------------------------------------------------------- app.html に入れる
DATA_BLOCK = re.compile(r'(<script id="ranking-data" type="application/json">)(.*?)(</script>)', re.S)


def embed(template_html, data):
    body = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    new, n = DATA_BLOCK.subn(lambda m: m.group(1) + "\n" + body + "\n" + m.group(3), template_html)
    if n != 1:
        raise RuntimeError("app.html の中に「株の数字のかたまり」が1つだけ見つかりませんでした（%d個）" % n)
    return new


def build(fetcher=fetch_metrics_yahoo, now=None, jp=None, us=None):
    now = now or datetime.now(JST)
    jp = jp if jp is not None else load_jp()
    us = us if us is not None else load_us()
    metrics = fetcher([x["ticker"] for x in jp + us])
    jp_stocks = score_market(jp, metrics)
    us_stocks = score_market(us, metrics)
    log("順位をつけられた株：日本株 %d / %d 社、アメリカ株 %d / %d 社" % (len(jp_stocks), len(jp), len(us_stocks), len(us)))
    if len(jp_stocks) < MIN_SUCCESS_RATE * len(jp) or len(us_stocks) < MIN_SUCCESS_RATE * len(us):
        raise RuntimeError("数字がそろった株が少なすぎるため、今朝の更新は行いません")
    return {
        "isSample": False,
        "updatedAt": now.astimezone(JST).strftime("%Y-%m-%dT%H:%M:00+09:00"),
        "markets": {
            "jp": {"label": "日本株", "codeLabel": "コード", "stocks": jp_stocks},
            "us": {"label": "アメリカ株", "codeLabel": "ティッカー", "stocks": us_stocks},
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="毎朝ランキングの数字を集めて、app.html を作る")
    ap.add_argument("--template", default=os.path.join(HERE, "app.html"), help="元になる app.html")
    ap.add_argument("--out", nargs="+", default=[os.path.join(HERE, "site", "index.html"),
                                                 os.path.join(HERE, "site", "app.html")],
                    help="書き出す先（複数可）")
    args = ap.parse_args(argv)
    try:
        data = build()
        html = embed(open(args.template, encoding="utf-8").read(), data)
    except Exception as e:
        log("【更新に失敗しました】%s" % e)
        return 1
    for p in args.out:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        log("書き出しました：%s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
