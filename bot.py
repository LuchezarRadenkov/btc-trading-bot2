"""
BTC Signal Bot — автоматизирана система за алгоритмично търгуване
Дипломна работа, Технически университет – София

Системата анализира пазара чрез седем независими източника, обединява
резултатите в претеглен сигнал и изпълнява сделки на демонстрационна
сметка в Bybit.

ВАЖНО: идентификационните данни по-долу са заместващи стойности.
Реалните ключове се задават чрез променливи на средата.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import feedparser
import xgboost as xgb
import gspread
from google.oauth2.service_account import Credentials
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tradingview_ta import TA_Handler, Interval
from pybit.unified_trading import HTTP

BG_TZ = ZoneInfo("Europe/Sofia")

# ══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# Стойностите се четат от променливи на средата; посочените
# по подразбиране са заместващи и служат само за документация.
# ══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "YOUR_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
SHEET_ID         = os.environ.get("SHEET_ID",         "YOUR_GOOGLE_SHEET_ID")
BYBIT_API_KEY    = os.environ.get("BYBIT_API_KEY",    "YOUR_BYBIT_API_KEY")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "YOUR_BYBIT_API_SECRET")

# Съдържанието на JSON файла на служебния акаунт в Google Cloud
GOOGLE_CREDENTIALS = json.loads(os.environ.get("GOOGLE_CREDENTIALS_JSON", "{}"))

# ══════════════════════════════════════════════════════════════
# ПАРАМЕТРИ НА СТРАТЕГИЯТА
# Оптимизирани чрез мрежово търсене (108 комбинации, 5 години данни)
# ══════════════════════════════════════════════════════════════
SYMBOL             = "BTCUSDT"
TRADE_SIZE_PCT     = 50      # дял от баланса на сделка
STOP_LOSS_PCT      = 5       # стоп-загуба
TAKE_PROFIT_PCT    = 6       # фиксиране на печалба
TRAILING_START_PCT = 4       # праг за активиране на плъзгащата защита
TRAILING_DIST_PCT  = 3       # дистанция на следване
CHECK_INTERVAL     = 14400   # интервал между циклите (4 часа)
WHALE_THRESHOLD    = 10_000_000   # праг за едра транзакция в USD
ML_RETRAIN_HOURS   = 24      # честота на преобучение на модела

session = HTTP(demo=True, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

STATE = {
    "ml_last_train": None,
    "ml_prediction": None,
    "ml_signal": "NEUTRAL",
    "error_count": 0,
    "cycle_count": 0,
    "last_daily_report": None,
}


# ══════════════════════════════════════════════════════════════
# СЛОЙ 1 — ДОСТЪП ДО ДАННИ
# ══════════════════════════════════════════════════════════════
def http_get(url, params=None, retries=3, timeout=8):
    """GET заявка с повторни опити и нарастващо изчакване."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def get_btc_price():
    """Текуща цена с верига от резервни източници."""
    try:
        r = session.get_tickers(category="linear", symbol=SYMBOL)
        return float(r["result"]["list"][0]["lastPrice"])
    except Exception:
        d = http_get("https://api.coingecko.com/api/v3/simple/price",
                     {"ids": "bitcoin", "vs_currencies": "usd"})
        if d:
            return float(d["bitcoin"]["usd"])
        d = http_get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        return float(d["data"]["amount"])


def get_historical_prices(days=365):
    """Дневни цени за обучение на прогнозиращия модел."""
    d = http_get("https://api.kraken.com/0/public/OHLC",
                 {"pair": "XBTUSD", "interval": 1440})
    if not d:
        return None
    result = d.get("result", {})
    keys = [k for k in result if k != "last"]
    if not keys:
        return None
    df = pd.DataFrame(result[keys[0]],
                      columns=["ts", "open", "high", "low",
                               "close", "vwap", "volume", "count"])
    df["y"] = df["close"].astype(float)
    return df[["y"]].tail(days).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# СЛОЙ 2 — АНАЛИТИЧНИ МОДУЛИ
# ══════════════════════════════════════════════════════════════
def get_tf_score(interval):
    """Технически резултат за конкретен времеви хоризонт."""
    try:
        analysis = TA_Handler(symbol="BTCUSDT", screener="crypto",
                              exchange="BINANCE", interval=interval)
        s = analysis.get_analysis().summary
        return s.get("BUY", 0) - s.get("SELL", 0)
    except Exception:
        return 0


def get_mtf_signal():
    """Мултитаймфрейм анализ — сигнал при съгласие на поне два хоризонта."""
    s1h = get_tf_score(Interval.INTERVAL_1_HOUR)
    s4h = get_tf_score(Interval.INTERVAL_4_HOURS)
    s1d = get_tf_score(Interval.INTERVAL_1_DAY)

    weighted = s1h * 0.2 + s4h * 0.35 + s1d * 0.45
    bullish = sum(1 for s in (s1h, s4h, s1d) if s > 2)
    bearish = sum(1 for s in (s1h, s4h, s1d) if s < -2)

    if bullish >= 2 and weighted >= 5:
        signal = "STRONG BUY"
    elif bullish >= 2 and weighted > 0:
        signal = "BUY"
    elif bearish >= 2 and weighted <= -5:
        signal = "STRONG SELL"
    elif bearish >= 2 and weighted < 0:
        signal = "SELL"
    else:
        signal = "HOLD"
    return s1h, s4h, s1d, round(weighted, 1), signal


def get_fear_greed():
    """Индекс на страха и алчността (контрариански сигнал)."""
    d = http_get("https://api.alternative.me/fng/?limit=1")
    if d:
        try:
            return int(d["data"][0]["value"]), d["data"][0]["value_classification"]
        except Exception:
            pass
    return 50, "Neutral"


def get_news_sentiment():
    """Анализ на настроенията в новинарския поток по ключови думи."""
    bullish_words = ["surge", "rally", "bull", "gain", "rise", "buy",
                     "growth", "pump", "record", "profit", "strong"]
    bearish_words = ["crash", "drop", "fall", "bear", "loss", "sell",
                     "decline", "dump", "fear", "ban", "hack", "risk"]

    positive = negative = neutral = 0
    feeds = ["https://cointelegraph.com/rss",
             "https://coindesk.com/arc/outboundfeeds/rss/"]

    for url in feeds:
        try:
            for entry in feedparser.parse(url).entries[:10]:
                title = entry.title.lower()
                bull = sum(1 for w in bullish_words if w in title)
                bear = sum(1 for w in bearish_words if w in title)
                if bull > bear:
                    positive += 1
                elif bear > bull:
                    negative += 1
                else:
                    neutral += 1
        except Exception:
            continue

    total = positive + negative + neutral
    if total == 0:
        return "NEUTRAL"

    bull_pct = positive / total * 100
    bear_pct = negative / total * 100
    if bull_pct >= 60:
        return "STRONG BULLISH"
    if bull_pct >= 45:
        return "BULLISH"
    if bear_pct >= 60:
        return "STRONG BEARISH"
    if bear_pct >= 45:
        return "BEARISH"
    return "NEUTRAL"


def get_onchain_signal():
    """Активност в блокчейн мрежата по брой транзакции."""
    d = http_get("https://blockchain.info/stats?format=json")
    if d:
        tx = d.get("n_tx", 0)
        if tx > 400_000:
            return "BULLISH"
        if tx > 300_000:
            return "NEUTRAL"
        return "BEARISH"
    return "NEUTRAL"


def get_orderbook_signal():
    """Дисбаланс между поръчките за покупка и продажба."""
    try:
        r = session.get_orderbook(category="linear", symbol=SYMBOL, limit=50)
        book = r["result"]
        bid_vol = sum(float(b[1]) for b in book["b"])
        ask_vol = sum(float(a[1]) for a in book["a"])
        total = bid_vol + ask_vol
        if total == 0:
            return "NEUTRAL", 0.5
        ratio = bid_vol / total
        if ratio >= 0.62:
            return "BULLISH", round(ratio, 2)
        if ratio <= 0.38:
            return "BEARISH", round(ratio, 2)
        return "NEUTRAL", round(ratio, 2)
    except Exception:
        return "NEUTRAL", 0.5


def get_whale_signal():
    """Активност на едри участници — предпазен филтър при несигурност."""
    try:
        params = {
            "q": "value_usd(" + str(WHALE_THRESHOLD) + ",)",
            "limit": 20,
            "s": "time(desc)",
        }
        d = http_get("https://api.blockchair.com/bitcoin/transactions", params)
        if not d or not isinstance(d.get("data"), list):
            return "QUIET", 0, 0

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=4)
        recent = []
        for t in d["data"]:
            try:
                if datetime.strptime(t["time"], "%Y-%m-%d %H:%M:%S") > cutoff:
                    recent.append(t)
            except Exception:
                continue

        count = len(recent)
        total_usd = sum(t.get("value_usd", 0) for t in recent)
        if count >= 8:
            return "HIGH ACTIVITY", count, total_usd
        if count >= 3:
            return "MODERATE", count, total_usd
        return "QUIET", count, total_usd
    except Exception as e:
        print(f"[WHALE ГРЕШКА] {e}")
        return "QUIET", 0, 0


def get_trend():
    """Основна пазарна тенденция по разположение спрямо EMA50 и EMA200."""
    try:
        analysis = TA_Handler(symbol="BTCUSDT", screener="crypto",
                              exchange="BINANCE", interval=Interval.INTERVAL_1_DAY)
        ind = analysis.get_analysis().indicators
        ema50 = ind.get("EMA50", 0)
        ema200 = ind.get("EMA200", 0)
        price = ind.get("close", 0)

        if price > ema50 > ema200:
            return "STRONG UPTREND"
        if price > ema50:
            return "UPTREND"
        if price < ema50 < ema200:
            return "STRONG DOWNTREND"
        if price < ema50:
            return "DOWNTREND"
        return "SIDEWAYS"
    except Exception:
        return "SIDEWAYS"


# ══════════════════════════════════════════════════════════════
# МОДУЛ ЗА МАШИННО ОБУЧЕНИЕ
# Ансамбъл от градиентен бустинг и невронна мрежа
# ══════════════════════════════════════════════════════════════
def make_features(series, lags=14):
    """Изгражда матрица от закъснения, пълзящи средни и волатилност."""
    feat = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        feat[f"lag{lag}"] = feat["y"].shift(lag)
    feat["ma7"] = feat["y"].rolling(7).mean()
    feat["ma30"] = feat["y"].rolling(30).mean()
    feat["ret1"] = feat["y"].pct_change()
    feat["vol7"] = feat["y"].pct_change().rolling(7).std()
    return feat.dropna()


def train_ml_model():
    """Обучава ансамбъла и кешира тридневна прогноза."""
    try:
        df = get_historical_prices(365)
        if df is None or len(df) < 100:
            print("[ML] Недостатъчно данни за обучение")
            return

        feats = make_features(df["y"].values)
        X, y = feats.drop("y", axis=1), feats["y"]

        # Модел 1 — градиентен бустинг
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                     learning_rate=0.05, random_state=42,
                                     n_jobs=1)
        xgb_model.fit(X, y)

        # Модел 2 — многослоен персептрон (невронна мрежа)
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.values.reshape(-1, 1))
        nn_model = MLPRegressor(hidden_layer_sizes=(64, 32),
                                activation="relu", solver="adam",
                                max_iter=500, random_state=42,
                                early_stopping=True)
        nn_model.fit(scaler_X.transform(X),
                     scaler_y.transform(y.values.reshape(-1, 1)).ravel())

        # Ансамблова прогноза с равни тегла
        series = list(df["y"].values)
        preds = []
        for _ in range(3):
            f = make_features(pd.Series(series)).iloc[[-1]].drop("y", axis=1)
            p_xgb = float(xgb_model.predict(f)[0])
            p_nn = float(scaler_y.inverse_transform(
                nn_model.predict(scaler_X.transform(f)).reshape(-1, 1))[0][0])
            p = 0.5 * p_xgb + 0.5 * p_nn
            preds.append(p)
            series.append(p)

        current = df["y"].iloc[-1]
        avg_pred = float(np.mean(preds))
        change_pct = (avg_pred - current) / current * 100

        if change_pct >= 2:
            signal = "STRONG BULLISH"
        elif change_pct >= 0.5:
            signal = "BULLISH"
        elif change_pct <= -2:
            signal = "STRONG BEARISH"
        elif change_pct <= -0.5:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        STATE["ml_last_train"] = datetime.now(BG_TZ)
        STATE["ml_prediction"] = round(avg_pred, 2)
        STATE["ml_signal"] = signal
        print(f"[ML] Ансамбъл XGB+НМ. Прогноза 3д: ${avg_pred:,.0f} "
              f"({change_pct:+.2f}%) → {signal}")
    except Exception as e:
        print(f"[ML ГРЕШКА] {e}")
        STATE["error_count"] += 1


def get_ml_signal():
    """Връща кешираната прогноза, като преобучава при необходимост."""
    needs_training = False
    if STATE["ml_last_train"] is None:
        needs_training = True
    else:
        elapsed = (datetime.now(BG_TZ) - STATE["ml_last_train"]).total_seconds()
        needs_training = elapsed > ML_RETRAIN_HOURS * 3600
    if needs_training:
        train_ml_model()
    return STATE["ml_signal"], STATE["ml_prediction"]


# ══════════════════════════════════════════════════════════════
# СЛОЙ 3 — ВЗЕМАНЕ НА РЕШЕНИЕ
# ══════════════════════════════════════════════════════════════
def get_final_signal(mtf_signal, fear_greed, news, onchain,
                     orderbook, whale, ml_signal, trend):
    """Претеглена агрегация с тренд филтър и праг."""
    score = 0
    score += {"STRONG BUY": 4, "BUY": 2, "HOLD": 0,
              "SELL": -2, "STRONG SELL": -4}.get(mtf_signal, 0)

    if fear_greed <= 20:
        score += 3
    elif fear_greed <= 40:
        score += 1
    elif fear_greed >= 80:
        score -= 3
    elif fear_greed >= 60:
        score -= 1

    score += {"STRONG BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
              "BEARISH": -1, "STRONG BEARISH": -2}.get(ml_signal, 0)
    score += {"STRONG BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
              "BEARISH": -1, "STRONG BEARISH": -2}.get(news, 0)
    score += {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(onchain, 0)
    score += {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(orderbook, 0)

    # Тренд филтър — не се търгува срещу основната посока
    if trend in ("STRONG UPTREND", "UPTREND") and score < 0:
        return "HOLD"
    if trend in ("STRONG DOWNTREND", "DOWNTREND") and score > 0:
        return "HOLD"
    if trend == "SIDEWAYS" and abs(score) < 5:
        return "HOLD"

    # Повишен праг при висока активност на едри участници
    if whale == "HIGH ACTIVITY" and abs(score) < 6:
        return "HOLD"

    if score >= 5:
        return "STRONG BUY"
    if score >= 3:
        return "BUY"
    if score <= -5:
        return "STRONG SELL"
    if score <= -3:
        return "SELL"
    return "HOLD"


# ══════════════════════════════════════════════════════════════
# СЛОЙ 4 — ИЗПЪЛНЕНИЕ И УПРАВЛЕНИЕ НА РИСКА
# ══════════════════════════════════════════════════════════════
def get_balance():
    try:
        r = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        for c in r["result"]["list"][0]["coin"]:
            if c["coin"] == "USDT":
                return float(c["walletBalance"])
        return 0
    except Exception:
        return 0


def get_position():
    try:
        r = session.get_positions(category="linear", symbol=SYMBOL)
        for p in r["result"]["list"]:
            size = p.get("size", "")
            if size and float(size) > 0:
                return {
                    "side": p["side"],
                    "size": float(size),
                    "entry": float(p.get("avgPrice", 0) or 0),
                    "pnl": float(p.get("unrealisedPnl", 0) or 0),
                }
        return None
    except Exception as e:
        print(f"[POSITION ГРЕШКА] {e}")
        return None


def open_position(side, qty, entry_price):
    """Пазарна поръчка с прикачени нива на защита."""
    try:
        if side == "Buy":
            sl = entry_price * (1 - STOP_LOSS_PCT / 100)
            tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
        else:
            sl = entry_price * (1 + STOP_LOSS_PCT / 100)
            tp = entry_price * (1 - TAKE_PROFIT_PCT / 100)

        r = session.place_order(category="linear", symbol=SYMBOL, side=side,
                                orderType="Market", qty=str(qty),
                                stopLoss=str(round(sl, 1)),
                                takeProfit=str(round(tp, 1)))
        return r["result"]["orderId"], sl, tp
    except Exception as e:
        print(f"[OPEN ГРЕШКА] {e}")
        STATE["error_count"] += 1
        return None, None, None


def close_position(side, qty):
    try:
        opposite = "Sell" if side == "Buy" else "Buy"
        session.place_order(category="linear", symbol=SYMBOL, side=opposite,
                            orderType="Market", qty=str(qty), reduceOnly=True)
        return True
    except Exception as e:
        print(f"[CLOSE ГРЕШКА] {e}")
        return False


def update_trailing_stop(position, current_price):
    """Активира плъзгаща се защита след достигане на прага на печалба."""
    try:
        entry, side = position["entry"], position["side"]
        if side == "Buy":
            profit_pct = (current_price - entry) / entry * 100
        else:
            profit_pct = (entry - current_price) / entry * 100

        if profit_pct >= TRAILING_START_PCT:
            trail = current_price * (TRAILING_DIST_PCT / 100)
            session.set_trading_stop(category="linear", symbol=SYMBOL,
                                     trailingStop=str(round(trail, 1)),
                                     positionIdx=0)
            print(f"[TRAILING] Активирана защита при +{profit_pct:.1f}%")
            return True
        return False
    except Exception as e:
        if "not modified" not in str(e).lower():
            print(f"[TRAILING ГРЕШКА] {e}")
        return False


# ══════════════════════════════════════════════════════════════
# СЛОЙ 5 — СЪХРАНЕНИЕ И КОМУНИКАЦИЯ
# ══════════════════════════════════════════════════════════════
def get_sheet(name="Combined"):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(GOOGLE_CREDENTIALS, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).worksheet(name)


def save_to_sheet(data):
    try:
        sheet = get_sheet()
        if sheet.cell(1, 1).value is None:
            sheet.append_row([
                "time", "price", "mtf_1h", "mtf_4h", "mtf_1d", "mtf_signal",
                "fear_greed", "news_signal", "onchain_signal",
                "orderbook_signal", "whale_signal", "ml_signal", "ml_pred",
                "trend", "final_signal", "balance",
            ])
        sheet.append_row([
            data["time"], data["price"], data["mtf_1h"], data["mtf_4h"],
            data["mtf_1d"], data["mtf_signal"], data["fear_greed"],
            data["news_signal"], data["onchain_signal"],
            data["orderbook_signal"], data["whale_signal"],
            data["ml_signal"], data["ml_pred"], data["trend"],
            data["final_signal"], data["balance"],
        ])
    except Exception as e:
        print(f"[SHEETS ГРЕШКА] {e}")
        STATE["error_count"] += 1


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message,
               "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"[TELEGRAM ГРЕШКА] {e}")


def maybe_daily_report(balance, price):
    """Изпраща обобщен отчет веднъж дневно след 8:00."""
    today = datetime.now(BG_TZ).date()
    if STATE["last_daily_report"] == today:
        return
    if datetime.now(BG_TZ).hour < 8:
        return

    STATE["last_daily_report"] = today
    ml_info = f"${STATE['ml_prediction']:,.0f}" if STATE["ml_prediction"] else "—"
    send_telegram(
        f"📋 <b>Дневен отчет</b>\n"
        f"💰 BTC: ${price:,.2f}\n"
        f"💵 Баланс: ${balance:,.2f}\n"
        f"🧠 Прогноза (3 дни): {ml_info} → {STATE['ml_signal']}\n"
        f"🔁 Изпълнени цикли: {STATE['cycle_count']}\n"
        f"⚠️ Регистрирани грешки: {STATE['error_count']}"
    )
    STATE["error_count"] = 0


# ══════════════════════════════════════════════════════════════
# ОРКЕСТРАТОР
# ══════════════════════════════════════════════════════════════
def run_bot():
    """Един пълен работен цикъл."""
    try:
        now = datetime.now(BG_TZ).strftime("%Y-%m-%d %H:%M:%S")
        STATE["cycle_count"] += 1
        print(f"\n[{now}] Цикъл #{STATE['cycle_count']}")

        balance = get_balance()
        price = get_btc_price()
        position = get_position()

        s1h, s4h, s1d, _, mtf_signal = get_mtf_signal()
        fear_greed, fg_label = get_fear_greed()
        news_signal = get_news_sentiment()
        onchain_signal = get_onchain_signal()
        orderbook_signal, ob_ratio = get_orderbook_signal()
        whale_signal, whale_count, whale_usd = get_whale_signal()
        ml_signal, ml_pred = get_ml_signal()
        trend = get_trend()

        signal = get_final_signal(mtf_signal, fear_greed, news_signal,
                                  onchain_signal, orderbook_signal,
                                  whale_signal, ml_signal, trend)

        print(f"Цена: ${price:,.2f} | Баланс: ${balance:,.2f}")
        print(f"Мултитаймфрейм: 1ч={s1h} 4ч={s4h} 1д={s1d} → {mtf_signal}")
        print(f"Прогноза: {ml_signal} | Книга с поръчки: {orderbook_signal}")
        print(f"Тенденция: {trend} | Решение: {signal}")

        save_to_sheet({
            "time": now, "price": round(price, 2),
            "mtf_1h": s1h, "mtf_4h": s4h, "mtf_1d": s1d,
            "mtf_signal": mtf_signal, "fear_greed": fear_greed,
            "news_signal": news_signal, "onchain_signal": onchain_signal,
            "orderbook_signal": f"{orderbook_signal} ({ob_ratio})",
            "whale_signal": f"{whale_signal} ({whale_count})",
            "ml_signal": ml_signal, "ml_pred": ml_pred or "",
            "trend": trend, "final_signal": signal,
            "balance": round(balance, 2),
        })

        maybe_daily_report(balance, price)

        if whale_signal == "HIGH ACTIVITY":
            send_telegram(f"🐋 <b>Висока активност на едри участници</b>\n"
                          f"{whale_count} транзакции за 4 часа\n"
                          f"Общ обем: ${whale_usd:,.0f}")

        # Управление на съществуваща позиция
        if position:
            update_trailing_stop(position, price)
            side = position["side"]
            if side == "Buy" and signal == "STRONG SELL":
                close_position(side, position["size"])
                send_telegram(f"🔄 <b>Затворена дълга позиция</b>\n"
                              f"Резултат: ${position['pnl']:+,.2f}")
            elif side == "Sell" and signal == "STRONG BUY":
                close_position(side, position["size"])
                send_telegram(f"🔄 <b>Затворена къса позиция</b>\n"
                              f"Резултат: ${position['pnl']:+,.2f}")
            return

        # Отваряне на нова позиция само при силен сигнал
        if signal in ("STRONG BUY", "STRONG SELL"):
            side = "Buy" if signal == "STRONG BUY" else "Sell"
            qty = round(balance * (TRADE_SIZE_PCT / 100) / price, 3)
            order_id, sl, tp = open_position(side, qty, price)
            if order_id:
                label = "🟢 Дълга" if side == "Buy" else "🔴 Къса"
                send_telegram(
                    f"{label} <b>позиция отворена</b>\n"
                    f"💰 Цена: ${price:,.2f} | {qty} BTC\n"
                    f"🛑 Стоп-загуба: ${sl:,.2f}\n"
                    f"🎯 Целева печалба: ${tp:,.2f}\n"
                    f"📊 Технически: {mtf_signal} | Прогноза: {ml_signal}\n"
                    f"📈 Тенденция: {trend}"
                )

    except Exception as e:
        print(f"[КРИТИЧНА ГРЕШКА] {e}")
        STATE["error_count"] += 1


if __name__ == "__main__":
    print("BTC Signal Bot — стартиране")
    train_ml_model()
    send_telegram(
        "✅ <b>Системата е стартирана</b>\n\n"
        "Аналитични слоеве:\n"
        "📊 Мултитаймфрейм анализ (1ч, 4ч, дневен)\n"
        "🧠 Ансамбъл XGBoost + невронна мрежа\n"
        "📖 Книга с поръчки | 🐋 Едри участници\n"
        "😱 Индекс на настроението | 📰 Новини | ⛓️ Блокчейн\n\n"
        f"Параметри: стоп-загуба {STOP_LOSS_PCT}%, "
        f"целева печалба {TAKE_PROFIT_PCT}%, "
        f"размер {TRADE_SIZE_PCT}%"
    )

    while True:
        run_bot()
        time.sleep(CHECK_INTERVAL)
