# Combined Solana Dashboard Script

# --- Imports and Setup ---


# === Module: main_dashboard ===
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Solana Dashboard", layout="wide")

# Rafraîchissement automatique toutes les 10 secondes
st_autorefresh(interval=10000, key="refresh_live")

st.title("📊 Solana Market Dashboard")

@st.cache_data
def get_sol_data():
    return yf.download("SOL-USD", period="60d", interval="1d").reset_index()

# Données historiques
sol_data = get_sol_data()
sol_data = ajouter_indicateurs(sol_data)

# Prix en direct
live_price = round(yf.Ticker("SOL-USD").info["regularMarketPrice"], 2)

# Header metrics
col1, col2, col3 = st.columns(3)
col1.metric("💰 Prix SOL (live)", f"${live_price}")
col2.metric("🔁 Rafraîchissement", "Toutes les 10s")
col3.metric("📈 Source", "Yahoo Finance")

# Modules d'affichage
st.markdown("## 1️⃣ Indicateurs techniques")
afficher_indicateurs(sol_data, st)

st.markdown("## 2️⃣ Heatmap volumes & volatilité")
afficher_heatmap_volumes(sol_data)

st.markdown("## 3️⃣ Sentiment marché")
afficher_sentiment()

st.markdown("## 4️⃣ Événements Solana")
afficher_evenements()

st.markdown("## 5️⃣ Portefeuille fictif")
afficher_portefeuille(live_price)


# === Module: indicateurs ===
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import ta

def ajouter_indicateurs(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().reset_index(drop=True)

    try:
        macd_ind = ta.trend.MACD(close=data['Close'])
        data['MACD'] = pd.Series(macd_ind.macd().values, index=data.index)
        data['MACD_signal'] = pd.Series(macd_ind.macd_signal().values, index=data.index)
    except:
        data['MACD'] = data['MACD_signal'] = None

    try:
        bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['bb_high'] = pd.Series(bb.bollinger_hband().values, index=data.index)
        data['bb_low'] = pd.Series(bb.bollinger_lband().values, index=data.index)
        data['bb_mavg'] = pd.Series(bb.bollinger_mavg().values, index=data.index)
    except:
        data['bb_high'] = data['bb_low'] = data['bb_mavg'] = None

    try:
        stoch = ta.momentum.StochRSIIndicator(close=data['Close'])
        data['stoch_rsi_k'] = pd.Series(stoch.stochrsi_k().values, index=data.index)
        data['stoch_rsi_d'] = pd.Series(stoch.stochrsi_d().values, index=data.index)
    except:
        data['stoch_rsi_k'] = data['stoch_rsi_d'] = None

    return data

def afficher_indicateurs(data: pd.DataFrame, st):
    st.subheader("📉 Indicateurs Techniques")

    # === FIGURE 1 : Bandes de Bollinger ===
    fig1, ax1 = plt.subplots()
    valid_rows = data[['Date', 'bb_low', 'bb_high', 'Close']].copy()
    valid_rows = valid_rows.dropna()
    valid_rows = valid_rows[
        valid_rows[['bb_low', 'bb_high', 'Close']].applymap(np.isfinite).all(axis=1)
    ]
    valid_rows = valid_rows.astype({'bb_low': 'float64', 'bb_high': 'float64', 'Close': 'float64'})
    
    dates = pd.to_datetime(valid_rows["Date"])
    x_dates = mdates.date2num(dates)

    ax1.plot(dates, valid_rows['Close'], label="Prix", color='blue')
    ax1.plot(dates, valid_rows['bb_high'], label="Bollinger Haut", linestyle='--', color='orange')
    ax1.plot(dates, valid_rows['bb_low'], label="Bollinger Bas", linestyle='--', color='green')
    ax1.fill_between(x_dates, valid_rows['bb_low'], valid_rows['bb_high'], alpha=0.1)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig1.autofmt_xdate()
    ax1.set_title("Bandes de Bollinger")
    ax1.legend()
    st.pyplot(fig1)

    # === FIGURE 2 : MACD ===
    fig2, ax2 = plt.subplots()
    macd_rows = data[['Date', 'MACD', 'MACD_signal']].dropna()
    ax2.plot(macd_rows['Date'], macd_rows['MACD'], label="MACD", color='purple')
    ax2.plot(macd_rows['Date'], macd_rows['MACD_signal'], label="Signal", color='red')
    ax2.set_title("MACD")
    ax2.legend()
    st.pyplot(fig2)

    # === FIGURE 3 : RSI Stochastique ===
    fig3, ax3 = plt.subplots()
    rsi_rows = data[['Date', 'stoch_rsi_k', 'stoch_rsi_d']].dropna()
    ax3.plot(rsi_rows['Date'], rsi_rows['stoch_rsi_k'], label="%K", color='blue')
    ax3.plot(rsi_rows['Date'], rsi_rows['stoch_rsi_d'], label="%D", color='orange')
    ax3.axhline(80, linestyle='--', color='red', alpha=0.3)
    ax3.axhline(20, linestyle='--', color='green', alpha=0.3)
    ax3.set_title("RSI Stochastique")
    ax3.legend()
    st.pyplot(fig3)


# === Module: heatmap_volumes ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def afficher_heatmap_volumes(data: pd.DataFrame):
    """
    Affiche une heatmap du volume et de la volatilité quotidienne (%).
    """
    st.subheader("🔥 Heatmap Volume & Volatilité")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Volatilité %"] = ((df["High"] - df["Low"]) / df["Open"]) * 100

    df["Jour"] = df["Date"].dt.day
    df["Mois"] = df["Date"].dt.strftime('%b %Y')

    pivot_vol = df.pivot_table(values="Volume", index="Jour", columns="Mois", aggfunc="mean")
    pivot_vola = df.pivot_table(values="Volatilité %", index="Jour", columns="Mois", aggfunc="mean")

    st.write("📊 Volume moyen journalier (par mois)")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_vol, ax=ax1, cmap="Blues", cbar=True, linewidths=0.5)
    st.pyplot(fig1)

    st.write("📈 Volatilité (%) moyenne quotidienne")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_vola, ax=ax2, cmap="Reds", cbar=True, linewidths=0.5)
    st.pyplot(fig2)


# === Module: sentiment ===
import pandas as pd
import random
import streamlit as st

def get_sentiment_simulé():
    """
    Simule un score de sentiment crypto global à partir de données fictives.
    """
    # Simule un score entre 0 et 100
    score = random.randint(30, 80)

    if score >= 70:
        sentiment = "🚀 Très positif"
    elif score >= 55:
        sentiment = "🙂 Positif"
    elif score >= 45:
        sentiment = "😐 Neutre"
    elif score >= 30:
        sentiment = "😟 Négatif"
    else:
        sentiment = "🛑 Très négatif"

    return score, sentiment

def afficher_sentiment():
    """
    Affiche le sentiment du marché simulé (Twitter/Reddit).
    """
    st.subheader("🧠 Sentiment marché (Twitter / Reddit simulé)")

    score, sentiment = get_sentiment_simulé()

    st.metric("Score Sentiment", f"{score}/100")
    st.info(f"Interprétation : {sentiment}")


# === Module: events ===
import streamlit as st
import pandas as pd

def get_evenements_solana():
    """
    Simule quelques événements à venir pour la blockchain Solana.
    Peut être remplacé par une API comme CoinMarketCal (clé nécessaire).
    """
    evenements = [
        {"Date": "2025-06-01", "Événement": "Conférence Solana Breakpoint", "Lieu": "Lisbonne"},
        {"Date": "2025-06-10", "Événement": "Mise à jour réseau v2.0", "Lieu": "Mainnet"},
        {"Date": "2025-06-15", "Événement": "Hackathon Web3 global", "Lieu": "Online"},
        {"Date": "2025-06-22", "Événement": "Annonce partenariat gaming", "Lieu": "Discord Solana"},
    ]
    return pd.DataFrame(evenements)

def afficher_evenements():
    """
    Affiche le calendrier des événements Solana à venir.
    """
    st.subheader("📅 Événements à venir sur Solana")
    df = get_evenements_solana()
    st.table(df)


# === Module: forecast_market ===
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# --- Fonction de prévision de RSI ---
def forecast_rsi(data, forecast_days=3):
    rsi_data = data[['RSI']].dropna().reset_index()
    if len(rsi_data) < 10:
        return None, None  # Pas assez de données
    rsi_data['timestamp'] = pd.to_datetime(rsi_data['Date']).astype(np.int64) // 10**9
    X = rsi_data['timestamp'].values.reshape(-1, 1)
    y = rsi_data['RSI'].values
    model = LinearRegression().fit(X, y)

    future_dates = pd.date_range(start=rsi_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    future_timestamps = (future_dates.astype(np.int64) // 10**9).reshape(-1, 1)
    rsi_forecast = model.predict(future_timestamps)
    return future_dates, rsi_forecast

# --- Fonction de détection anticipée ---
def detect_future_market_signals(future_rsi):
    if future_rsi is None:
        return "Pas assez de données pour prévoir"

    bull_signal = any(rsi > 60 for rsi in future_rsi)
    bear_signal = any(rsi < 40 for rsi in future_rsi)

    if bull_signal and not bear_signal:
        return "🚀 Bull Market probable dans les prochains jours"
    elif bear_signal and not bull_signal:
        return "🛑 Bear Market probable dans les prochains jours"
    elif bull_signal and bear_signal:
        return "⚖️ Forte instabilité attendue : Signaux contradictoires"
    else:
        return "⏳ Aucun signal clair pour les prochains jours"

# --- Intégration Streamlit ---
def afficher_prevision_market(data):
    st.subheader("🔮 Prévision Bull/Bear Market (RSI 3 jours)")

    future_dates, future_rsi = forecast_rsi(data)
    resume = detect_future_market_signals(future_rsi)
    st.info(resume)

    if future_rsi is not None:
        forecast_df = pd.DataFrame({"Date": future_dates, "RSI prévu": future_rsi})
        st.line_chart(forecast_df.set_index("Date"))


# === Module: portefeuille ===
import streamlit as st
import json
from datetime import datetime
import os

wallet_file = "wallet.json"

def charger_wallet():
    if os.path.exists(wallet_file):
        with open(wallet_file, "r") as f:
            return json.load(f)
    else:
        return {"mode": "manuel", "transactions": []}

def sauvegarder_wallet(wallet):
    with open(wallet_file, "w") as f:
        json.dump(wallet, f, indent=2)

def ajouter_transaction(wallet, action, montant, prix):
    wallet["transactions"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "montant": montant,
        "prix": prix
    })
    sauvegarder_wallet(wallet)

def afficher_portefeuille(sol_price):
    st.subheader("💼 Portefeuille fictif Solana")

    wallet = charger_wallet()

    # Sélecteur de mode
    mode = st.radio("Mode de gestion des achats :", ["manuel", "auto"], index=0 if wallet["mode"] == "manuel" else 1)
    wallet["mode"] = mode
    sauvegarder_wallet(wallet)

    if mode == "manuel":
        col1, col2 = st.columns(2)
        with col1:
            montant = st.number_input("Quantité de SOL à acheter", min_value=0.1, step=0.1)
        with col2:
            if st.button("Acheter SOL maintenant"):
                ajouter_transaction(wallet, "achat", montant, sol_price)
                st.success(f"Achat de {montant} SOL à ${sol_price}")
    else:
        st.info("Le mode Auto est activé. Les achats seront simulés automatiquement selon les signaux.")

    # Historique
    if wallet["transactions"]:
        st.write("📜 Historique des transactions :")
        st.table(wallet["transactions"])
    else:
        st.info("Aucune transaction enregistrée.")
