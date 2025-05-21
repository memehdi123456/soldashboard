import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np


st.set_page_config(page_title="Solana Market Signals", layout="wide")
st.title("📊 Solana Market Signals")

# ✅ Utilise une image locale à placer dans le même dossier que ce script
st.image("solana_banner.png", use_container_width=True)

# === Fonctions données ===
@st.cache_data
def get_fear_greed_index():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1").json()
        return int(response['data'][0]['value'])
    except:
        return None

@st.cache_data
def get_sol_data(days=60):
    return yf.download("SOL-USD", period=f"{days}d", interval="1d")

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def detect_signals(data, fg_index):
    signals = []

    # Assure-toi que les lignes nécessaires existent
    if len(data) < 30 or data['RSI'].isnull().all():
        return ["⚠️ Données insuffisantes ou RSI manquant"], 0.0, 0.0, 0.0

    try:
        rsi = float(data['RSI'].iloc[-1])
        price = float(data['Close'].iloc[-1])
        price_30d = float(data['Close'].iloc[-30])
        change = float(((price - price_30d) / price_30d) * 100)
    except Exception as e:
        return [f"⚠️ Erreur de calcul : {e}"], 0.0, 0.0, 0.0

    # Analyse Bull/Bear
    if fg_index is not None and fg_index < 30:
        signals.append("⚠️ Fear Index < 30")
    if fg_index is not None and fg_index > 60:
        signals.append("✅ Fear Index > 60")
    if rsi < 40:
        signals.append("⚠️ RSI < 40")
    if rsi > 60:
        signals.append("✅ RSI > 60")
    if change < -15:
        signals.append("⚠️ -15% in 30 days")
    if change > 20:
        signals.append("✅ +20% in 30 days")

    return signals, round(change, 2), round(rsi, 2), round(price, 2)


def forecast_price(data, future_days=7):
    data = data.dropna().reset_index()
    data['Timestamp'] = pd.to_datetime(data['Date']).astype(np.int64) // 10**9

    X = data['Timestamp'].values.reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression().fit(X, y)

    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    future_ts = future_dates.astype(np.int64) // 10**9
    future_ts = np.array(future_ts).reshape(-1, 1)  # ✅ conversion explicite

    future_preds = model.predict(future_ts)

    return future_dates, future_preds

    


# === Analyse ===
fg_index = get_fear_greed_index()
sol_data = get_sol_data()
sol_data = calculate_rsi(sol_data)
signals, change_30d, rsi_now, last_price = detect_signals(sol_data, fg_index)

# === Enregistrement historique CSV ===
today = datetime.today().strftime("%Y-%m-%d")
history_file = "signal_history.csv"

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(columns=["Date", "Price", "RSI", "Change30d", "FearGreed", "Signals"])

if today not in history_df["Date"].values:
    new_row = {
        "Date": today,
        "Price": last_price,
        "RSI": rsi_now,
        "Change30d": change_30d,
        "FearGreed": fg_index,
        "Signals": "; ".join(signals)
    }
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(history_file, index=False)

# === Affichage des métriques ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix SOL", f"${last_price}")
col2.metric("RSI", rsi_now)
col3.metric("Fear & Greed", fg_index)
col4.metric("Variation 30j", f"{change_30d}%")

# === Signaux ===
st.subheader("🧠 Signaux détectés")
if signals:
    for sig in signals:
        st.success(sig) if "✅" in sig else st.warning(sig)
else:
    st.info("Aucun signal fort détecté pour le moment.")

# === Graphiques ===
st.subheader("📈 Prix + Prévision (7j)")
future_dates, future_preds = forecast_price(sol_data)
forecast_df = pd.concat([
    sol_data[['Close']].rename(columns={"Close": "Prix réel"}),
    pd.DataFrame({"Prix prédit": future_preds.flatten()}, index=future_dates)
], axis=1)
st.line_chart(forecast_df)

st.subheader("📉 RSI sur 60 jours")
st.line_chart(sol_data['RSI'])

# === Historique des signaux ===
st.subheader("📋 Historique des signaux (30 derniers jours)")
history_df["Date"] = pd.to_datetime(history_df["Date"])
last_30_days = history_df[history_df["Date"] > (datetime.now() - pd.Timedelta(days=30))]
st.dataframe(last_30_days.sort_values("Date", ascending=False), use_container_width=True)

st.download_button("📥 Télécharger l'historique CSV", data=history_df.to_csv(index=False), file_name="signal_history.csv")


