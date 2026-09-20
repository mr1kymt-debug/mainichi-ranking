"""screener_result.csv（と、あれば news_view.json）を stock_board.html に埋め込み、
公開用の docs/index.html を作る。

株価を取得できた銘柄が少ないときは、エラーで止まります。
（止まると変更は保存されず、公開中のページは前日の内容のまま残ります）
"""
import csv
import datetime
import io
import json
import pathlib
import sys

CSV_PATH = pathlib.Path("screener_result.csv")
NEWS_PATH = pathlib.Path("news_view.json")
TEMPLATE = pathlib.Path("stock_board.html")
OUT = pathlib.Path("docs/index.html")

text = CSV_PATH.read_text(encoding="utf-8-sig")
rows = list(csv.DictReader(io.StringIO(text)))
with_prices = [r for r in rows if r.get("ret_3m")]
if not rows or len(with_prices) < len(rows) * 0.5:
    sys.exit(f"株価を取得できた銘柄が少なすぎます（{len(with_prices)}/{len(rows)}）。更新を中止しました。")

payload = {"csv": text, "updatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()}
if NEWS_PATH.exists():
    try:
        payload["news"] = json.loads(NEWS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[警告] news_view.json を読み込めなかったため、ニュースなしで作成します: {e}")
data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
block = f'<script id="embedded" type="application/json">{data_json}</script>'

html = TEMPLATE.read_text(encoding="utf-8")
if "<!--EMBEDDED_DATA-->" not in html:
    sys.exit("stock_board.html に <!--EMBEDDED_DATA--> が見つかりません。")
html = html.replace("<!--EMBEDDED_DATA-->", block)
html = html.replace("<head>", '<head>\n<meta name="robots" content="noindex">', 1)  # 検索に出にくくする

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"{OUT} を作成しました（{len(rows)}銘柄、ニュース{'あり' if 'news' in payload else 'なし'}）")
