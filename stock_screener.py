"""株スクリーナー（自分用）

使い方:
    pip install yfinance pandas
    python stock_screener.py

3つの見方で銘柄を評価し、日本株・米国株それぞれの市場内でランキングします。
  1. 上昇の勢い … 株価の上がり方と、売上・利益の伸び
  2. 頭打ちリスク … 上がってきた株が、この先足踏みしそうかどうか
  3. 1か月後の見立て … 上の2つに、直近のニュース（news_view.json）と市場全体の状況を加えた判断
     あわせて、過去1年の値動きの大きさから「1か月後の株価の目安の範囲」も出します。

総合スコア = 上昇の勢い75% + 頭打ちしにくさ25%

あくまで候補を絞り込む補助ツールで、将来の値上がりを保証するものではありません。
"""
import datetime
import json
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf

# 対象銘柄（自由に追加・変更してください）。日本株は末尾に ".T"
UNIVERSE = {
    "US": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "JP": ["7203.T", "6758.T", "9984.T", "8035.T", "6861.T"],
}

# 上昇の勢い（高いほど良い指標）。合計1。仮の値なので、使いながら調整してください
WEIGHTS = {
    "ret_3m": 0.25,           # 3か月の値上がり率
    "ret_1m": 0.10,           # 1か月の値上がり率
    "above_ma200": 0.10,      # 約10か月の平均株価との差（長い目で見た上昇傾向）
    "volume_ratio": 0.10,      # 直近5日÷60日の取引量（注目度の変化）
    "revenue_growth": 0.20,   # 売上の伸び
    "earnings_growth": 0.20,  # 利益の伸び
    "pe_improve": 0.05,       # 実績PER÷予想PER（1より大きいと増益予想）
}

# 頭打ちリスク（高いほど頭打ちしやすい指標）。合計1
PLATEAU_WEIGHTS = {
    "rsi14": 0.30,         # 買われすぎ度（0〜100。70超は過熱ぎみ）
    "slowdown": 0.30,      # 上昇ペースの鈍り（3か月の月平均ペース − 直近1か月）
    "forward_pe": 0.25,    # 株価は今後の利益の何倍か（高いほど割高）
    "volume_ratio": 0.15,  # 取引の細り（取引量が少ないほどリスク大）
}
LOW_IS_RISKY = {"volume_ratio"}  # 値が「小さい」ほどリスクが高い指標

PLATEAU_SHARE = 0.25   # 総合スコアのうち「頭打ちしにくさ」が占める割合
HIGH_RISK = 67         # これ以上なら「頭打ち注意」
LOW_RISK = 33          # これ以下なら「上昇継続の可能性」

# 1か月後の見立て（news_view.json のニュースと市場全体の状況を加味）
NEWS_FILE = "news_view.json"
NEWS_MAX_AGE_DAYS = 10                              # これより古いニュースは見立てに使わない
TONE_VALUE = {"追い風": 1, "中立": 0, "逆風": -1}
TICKER_NEWS_WEIGHT = 0.75                           # 個別銘柄のニュースの効き方
MARKET_NEWS_WEIGHT = 0.50                           # 市場全体の状況の効き方
PLATEAU_PENALTY = 0.5                               # 「頭打ち注意」のときの減点
TRADING_DAYS_1M = 21                                # 1か月の営業日数


def compute_rsi(close: pd.Series, n: int = 14) -> float:
    """買われすぎ度（RSI）。上げ幅と下げ幅の平均の比から0〜100で表す。"""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up.iloc[-1] / down.iloc[-1]
        return float(100 - 100 / (1 + rs))


def compute_price_features(hist: pd.DataFrame) -> dict:
    """株価履歴（Close, Volume列）から価格系の指標を計算する。"""
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    feats = {}
    if len(close) >= 1:
        feats["price"] = float(close.iloc[-1])
    if len(close) >= 22:
        feats["ret_1m"] = close.iloc[-1] / close.iloc[-22] - 1
    if len(close) >= 64:
        feats["ret_3m"] = close.iloc[-1] / close.iloc[-64] - 1
    if len(close) >= 200:
        feats["above_ma200"] = close.iloc[-1] / close.rolling(200).mean().iloc[-1] - 1
    if len(volume) >= 60 and volume.tail(60).mean() > 0:
        feats["volume_ratio"] = volume.tail(5).mean() / volume.tail(60).mean()
    if len(close) >= 30:
        rsi = compute_rsi(close)
        if np.isfinite(rsi):
            feats["rsi14"] = rsi
    if "ret_1m" in feats and "ret_3m" in feats:
        feats["slowdown"] = feats["ret_3m"] / 3 - feats["ret_1m"]
    if len(close) >= 60:
        # 過去約1年の日々の値動きの大きさ → 1か月分に換算
        daily = np.log(close).diff().dropna().tail(252).std()
        if np.isfinite(daily):
            feats["vol_1m"] = float(daily * np.sqrt(TRADING_DAYS_1M))
    return feats


def fetch_features(ticker: str) -> dict:
    """1銘柄分のデータを取得して指標にまとめる。取得失敗時は空の指標を返す。"""
    feats = {}
    try:
        t = yf.Ticker(ticker)
        feats.update(compute_price_features(t.history(period="1y")))
        info = t.info
        feats["revenue_growth"] = info.get("revenueGrowth")
        feats["earnings_growth"] = info.get("earningsGrowth")
        trailing, forward = info.get("trailingPE"), info.get("forwardPE")
        if trailing and forward and trailing > 0 and forward > 0:
            feats["pe_improve"] = trailing / forward
        if forward and forward > 0:
            feats["forward_pe"] = forward
    except Exception as e:  # ネットワークエラーや銘柄コード違いなど
        print(f"[警告] {ticker} の取得に失敗: {e}")
    return feats


def load_news(path=NEWS_FILE, today=None):
    """news_view.json を読み込む。無い・古い・壊れているときは None（ニュースなしで計算）。"""
    p = pathlib.Path(path)
    if not p.exists():
        print("[情報] news_view.json がないため、ニュースは見立てに使いません。")
        return None
    try:
        news = json.loads(p.read_text(encoding="utf-8"))
        as_of = datetime.date.fromisoformat(news["asOf"])
    except Exception as e:
        print(f"[警告] news_view.json を読み取れませんでした（{e}）。ニュースは使いません。")
        return None
    age = ((today or datetime.date.today()) - as_of).days
    if age > NEWS_MAX_AGE_DAYS:
        print(f"[警告] ニュースが{age}日前の情報なので、見立てには使いません。")
        return None
    return news


def _rank(df: pd.DataFrame, col: str, high_is_top: bool = True) -> pd.Series:
    """同じ市場の中での順位（0〜1）。データがない銘柄は真ん中（0.5）として扱う。"""
    return df.groupby("market")[col].rank(pct=True, ascending=high_is_top).fillna(0.5)


def direction_label(points: float) -> str:
    if points >= 1.0:
        return "上向き"
    if points >= 0.4:
        return "やや上向き"
    if points > -0.4:
        return "横ばい"
    if points > -1.0:
        return "やや下向き"
    return "下向き"


def build_scores(df: pd.DataFrame, news=None) -> pd.DataFrame:
    """上昇の勢い・頭打ちリスク・総合スコア・1か月後の見立てを計算する。"""
    df = df.copy()
    for col in set(WEIGHTS) | set(PLATEAU_WEIGHTS) | {"ret_3m", "above_ma200", "price", "vol_1m"}:
        if col not in df.columns:
            df[col] = np.nan

    rising = sum(w * _rank(df, c) for c, w in WEIGHTS.items())
    risk = sum(w * _rank(df, c, high_is_top=c not in LOW_IS_RISKY) for c, w in PLATEAU_WEIGHTS.items())
    df["rising_score"] = (rising * 100).round(1)
    df["plateau_risk"] = (risk * 100).round(1)

    known = df["ret_3m"].notna() & df["above_ma200"].notna()
    is_rising = (df["ret_3m"] > 0) & (df["above_ma200"] > 0)
    df["outlook"] = np.select(
        [~known, ~is_rising, df["plateau_risk"] >= HIGH_RISK, df["plateau_risk"] <= LOW_RISK],
        ["判定できない", "上昇していない", "頭打ち注意", "上昇継続の可能性"],
        default="様子見",
    )
    df["score"] = ((1 - PLATEAU_SHARE) * df["rising_score"] + PLATEAU_SHARE * (100 - df["plateau_risk"])).round(1)

    # --- 1か月後の見立て（スコア + 頭打ち + ニュース + 市場全体） ---
    news = news or {}
    ticker_tone = df.index.to_series().map(lambda t: (news.get("tickers", {}).get(t) or {}).get("tone"))
    market_tone = df["market"].map(lambda m: (news.get("market", {}).get(m) or {}).get("tone"))
    df["news_tone"] = ticker_tone.fillna("")
    df["market_tone"] = market_tone.fillna("")
    base = ((df["score"] - 50) / 25).clip(-1, 1)
    points = (
        base
        - np.where(df["outlook"] == "頭打ち注意", PLATEAU_PENALTY, 0.0)
        + TICKER_NEWS_WEIGHT * ticker_tone.map(TONE_VALUE).fillna(0)
        + MARKET_NEWS_WEIGHT * market_tone.map(TONE_VALUE).fillna(0)
    )
    df["direction"] = points.map(direction_label)

    # --- 1か月後の株価の目安（過去1年の値動きの大きさから計算。方向は決めつけない） ---
    for name, z in (("range", 1.0), ("wide", 1.96)):   # 約7割 / 約95%
        df[f"{name}_low"] = (df["price"] * np.exp(-z * df["vol_1m"])).round(2)
        df[f"{name}_high"] = (df["price"] * np.exp(z * df["vol_1m"])).round(2)
    df["price"] = df["price"].round(2)
    df["vol_1m"] = df["vol_1m"].round(4)
    return df.sort_values(["market", "score"], ascending=[True, False])


def main():
    news = load_news()
    rows = []
    for market, tickers in UNIVERSE.items():
        for ticker in tickers:
            print(f"取得中: {ticker}")
            rows.append({"ticker": ticker, "market": market, **fetch_features(ticker)})
    result = build_scores(pd.DataFrame(rows).set_index("ticker"), news)

    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print("\n=== ランキング（市場別） ===")
    print(result[["market", "score", "outlook", "direction", "price", "range_low", "range_high"]].to_string())
    result.to_csv("screener_result.csv", encoding="utf-8-sig")
    print("\nscreener_result.csv に保存しました")


if __name__ == "__main__":
    main()
