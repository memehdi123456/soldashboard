import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import requests
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import feedparser

st.set_page_config(page_title="Solana Market Signals", layout="wide")
st.title("📊 Solana Market Signals")
st.image("solana_banner.png", use_container_width=True)

# Auto-refresh toutes les 10 secondes
st_autorefresh(interval=10000, key="refresh_live")

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
    if len(data) < 30 or data['RSI'].isnull().all():
        return ["⚠️ Données insuffisantes"], 0.0, 0.0, 0.0
    try:
        rsi = float(data['RSI'].iloc[-1])
        price = float(data['Close'].iloc[-1])
        price_30d = float(data['Close'].iloc[-30])
        change = float(((price - price_30d) / price_30d) * 100)
    except Exception as e:
        return [f"⚠️ Erreur : {e}"], 0.0, 0.0, 0.0

    if fg_index and fg_index < 30:
        signals.append("⚠️ Fear Index < 30")
    if fg_index and fg_index > 60:
        signals.append("✅ Fear Index > 60")
    if rsi < 40:
        signals.append("⚠️ RSI < 40")
    if rsi > 60:
        signals.append("✅ RSI > 60")
    if change < -15:
        signals.append("⚠️ -15% en 30 jours")
    if change > 20:
        signals.append("✅ +20% en 30 jours")

    return signals, round(change, 2), round(rsi, 2), round(price, 2)

# === Prix live ===
try:
    live_price = round(yf.Ticker("SOL-USD").info["regularMarketPrice"], 2)
except:
    live_price = "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("💰 Prix SOL (live)", f"${live_price}")
col2.metric("🔁 Rafraîchissement", "Toutes les 10s")
col3.metric("📈 Source", "Yahoo Finance")

# === Chargement des données ===
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

    if len(data) < 30 or data['RSI'].isnull().all():
        return ["⚠️ Données insuffisantes"], 0.0, 0.0, 0.0

    try:
        rsi = float(data['RSI'].iloc[-1])
        price = float(data['Close'].iloc[-1])
        price_30d = float(data['Close'].iloc[-30])
        change = float(((price - price_30d) / price_30d) * 100)
    except Exception as e:
        return [f"⚠️ Erreur : {e}"], 0.0, 0.0, 0.0

    if fg_index and fg_index < 30:
        signals.append("⚠️ Fear Index < 30")
    if fg_index and fg_index > 60:
        signals.append("✅ Fear Index > 60")
    if rsi < 40:
        signals.append("⚠️ RSI < 40")
    if rsi > 60:
        signals.append("✅ RSI > 60")
    if change < -15:
        signals.append("⚠️ -15% en 30 jours")
    if change > 20:
        signals.append("✅ +20% en 30 jours")

    return signals, round(change, 2), round(rsi, 2), round(price, 2)

def forecast_price(data, future_days=7):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    data = data.dropna().reset_index()
    data['Timestamp'] = pd.to_datetime(data['Date']).astype(np.int64) // 10**9

    X = data['Timestamp'].values.reshape(-1, 1)
    y = data['Close'].values

    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(X, y)

    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    future_ts = future_dates.astype(np.int64) // 10**9
    future_ts = np.array(future_ts).reshape(-1, 1)

    future_preds = model.predict(future_ts)

    # Zone d'incertitude simple = écart-type passé
    std_error = np.std(y - model.predict(X))
    lower_bound = future_preds - std_error
    upper_bound = future_preds + std_error

    forecast_df = pd.DataFrame({
        "Prix prédit": future_preds.flatten(),
        "Min": lower_bound.flatten(),
        "Max": upper_bound.flatten()
    }, index=future_dates)

    return forecast_df


# === NOUVELLE FONCTION DE PRÉVISION RSI ===
def afficher_prevision_market(data):
    st.subheader("🔮 Prévision de tendance RSI (3 jours)")

    last_days = data[['RSI']].dropna().tail(10).reset_index()
    if len(last_days) < 5:
        st.info("Pas assez de données pour prédire le RSI.")
        return

    X = np.arange(len(last_days)).reshape(-1, 1)
    y = last_days['RSI'].values
    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(last_days), len(last_days) + 3).reshape(-1, 1)
    future_rsi = model.predict(future_X)
    df_forecast = pd.DataFrame({
        "Jour": ["J+1", "J+2", "J+3"],
        "RSI Prévu": np.round(future_rsi, 2)
    })

    rsi_trend = future_rsi[-1]
    if rsi_trend > 60:
        st.success("🚀 Le RSI devrait dépasser 60 → Bull Market probable bientôt.")
    elif rsi_trend < 40:
        st.warning("🛑 Le RSI pourrait descendre sous 40 → Bear Market probable.")
    else:
        st.info("ℹ️ Aucune tendance claire à court terme.")

    st.table(df_forecast)

# === Analyse ===
fg_index = get_fear_greed_index()
sol_data = get_sol_data()
sol_data = calculate_rsi(sol_data)
signals, change_30d, rsi_now, last_price = detect_signals(sol_data, fg_index)

# === Historique CSV ===
today = datetime.today().strftime("%Y-%m-%d")
history_file = "signal_history.csv"
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(columns=["Date", "Price", "RSI", "Change30d", "FearGreed", "Signals"])


def analyser_achat_vente(data, fg_index):
    signal = "⚪ Attente / Neutre"
    raisons = []

    # Calcule EMA
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()

    rsi = float(data['RSI'].iloc[-1])
    prix = float(data['Close'].iloc[-1])
    ema9 = float(data['EMA_9'].iloc[-1])
    ema21 = float(data['EMA_21'].iloc[-1])
    prix_7j = float(data['Close'].iloc[-7])

    variation_7j = ((prix - prix_7j) / prix_7j) * 100

    # Conditions d'achat
    conds_achat = 0
    if 45 <= rsi <= 65:
        conds_achat += 1
        raisons.append("✅ RSI dans une bonne zone (45-65)")
    if ema9 > ema21:
        conds_achat += 1
        raisons.append("✅ Croisement haussier EMA 9 > EMA 21")
    if fg_index and fg_index > 50:
        conds_achat += 1
        raisons.append("✅ Sentiment marché positif (FG > 50)")
    if prix > ema21:
        conds_achat += 1
        raisons.append("✅ Prix au-dessus EMA 21")

    if conds_achat >= 3:
        signal = "🟢 Achat conseillé"

    # Conditions de vente
    conds_vente = 0
    if rsi > 75:
        conds_vente += 1
        raisons.append("🔻 RSI en surachat (> 75)")
    if ema9 < ema21:
        conds_vente += 1
        raisons.append("🔻 Croisement baissier EMA 9 < EMA 21")
    if fg_index and fg_index < 40:
        conds_vente += 1
        raisons.append("🔻 Peur sur le marché (FG < 40)")
    if variation_7j < -10:
        conds_vente += 1
        raisons.append(f"🔻 Chute de {round(variation_7j,2)}% sur 7 jours")

    if conds_vente >= 2:
        signal = "🔴 Vente conseillée"

    return signal, raisons
    

if today not in history_df["Date"].values:
    strategy_signal, raisons = analyser_achat_vente(sol_data, fg_index)

    new_row = {
        "Date": today,
        "Price": last_price,
        "RSI": rsi_now,
        "Change30d": change_30d,
        "FearGreed": fg_index,
        "Signals": "; ".join(signals),
        "Action": strategy_signal  # ✅ Cette ligne ajoute la colonne Action
    }

    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(history_file, index=False)

import matplotlib.pyplot as plt

st.subheader("📈 Évolution du prix avec signaux Achat/Vente")

df_plot = history_df.copy()
df_plot["Date"] = pd.to_datetime(df_plot["Date"])
df_plot = df_plot[df_plot["Date"] > (datetime.now() - pd.Timedelta(days=30))]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_plot["Date"], df_plot["Price"], label="Prix SOL", color='blue')

# Points d'achat / vente
if "Action" in df_plot.columns:
    buy_signals = df_plot[df_plot["Action"].str.contains("Achat", na=False)]
    sell_signals = df_plot[df_plot["Action"].str.contains("Vente", na=False)]
    
    ax.scatter(buy_signals["Date"], buy_signals["Price"], color="green", label="Achat", marker="^", s=100)
    ax.scatter(sell_signals["Date"], sell_signals["Price"], color="red", label="Vente", marker="v", s=100)
else:
    st.warning("⚠️ La colonne 'Action' n'est pas encore présente dans le fichier CSV. Elle sera ajoutée après le prochain rafraîchissement.")


ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Prix en $")
st.pyplot(fig)


# === Dashboard ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix SOL", f"${last_price}")
col2.metric("RSI", rsi_now)
col3.metric("Fear & Greed", fg_index)
col4.metric("Variation 30j", f"{change_30d}%")

st.subheader("🧠 Signaux détectés (aujourd’hui)")
if signals:
    for sig in signals:
        st.success(sig) if "✅" in sig else st.warning(sig)
else:
    st.info("Aucun signal clair aujourd’hui.")



# === PRÉVISION RSI / TENDANCE COURT TERME ===
afficher_prevision_market(sol_data)

# === Graphiques ===
st.subheader("📈 Prix SOL + prévision (7j)")
forecast_df = forecast_price(sol_data)
st.subheader("📈 Prévision du prix de SOL (7 jours)")
st.line_chart(forecast_df)

st.subheader("📉 RSI sur 60 jours")
st.line_chart(sol_data['RSI'])

def forecast_rsi(data, future_days=3):
    data = data[['RSI']].dropna().tail(10).reset_index()
    if len(data) < 5:
        st.info("Pas assez de données pour prédire le RSI.")
        return

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['RSI'].values

    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)

    future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
    future_rsi = model.predict(future_X)

    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    future_rsi_df = pd.DataFrame({"RSI prédit": future_rsi}, index=future_dates)

    st.subheader("📉 Prévision du RSI (3 jours)")
    st.line_chart(future_rsi_df)
forecast_rsi(sol_data)

st.subheader("💡 Stratégie actuelle (achat / vente)")
signal, raisons = analyser_achat_vente(sol_data, fg_index)
st.markdown(f"### {signal}")
for r in raisons:
    st.write(r)


import matplotlib.pyplot as plt

def afficher_graphique_actions(history_df):
    st.subheader("📊 Historique des recommandations (30 derniers jours)")

    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] > (datetime.now() - pd.Timedelta(days=30))]

    if "Action" not in df.columns:
        st.info("Aucune recommandation enregistrée pour le moment.")
        return

    # Simplifier le texte pour graphique
    df['Signal'] = df['Action'].str.extract(r"(🟢|🔴|⚪)")
    signal_map = {"🟢": "Achat", "🔴": "Vente", "⚪": "Attente"}
    df['Signal'] = df['Signal'].map(signal_map)

    count_df = df.groupby(['Date', 'Signal']).size().unstack(fill_value=0)

    st.bar_chart(count_df)


# === Historique des signaux ===
st.subheader("📋 Historique des signaux (30 derniers jours)")
history_df["Date"] = pd.to_datetime(history_df["Date"])
last_30_days = history_df[history_df["Date"] > (datetime.now() - pd.Timedelta(days=30))]
st.dataframe(last_30_days.sort_values("Date", ascending=False), use_container_width=True)
st.download_button("📥 Télécharger l'historique CSV", data=history_df.to_csv(index=False), file_name="signal_history.csv")

st.subheader("📊 Recommandations enregistrées (filtrables)")
import streamlit.components.v1 as components

if "Action" in history_df.columns:
    df_table = history_df[["Date", "Price", "Action"]].sort_values("Date", ascending=False)
    st.dataframe(df_table, use_container_width=True)
else:
    st.info("Aucune donnée de recommandation disponible.")


afficher_graphique_actions(history_df)

import feedparser
import pandas as pd

def get_crypto_news():
    url = "https://cointelegraph.com/rss"
    try:
        feed = feedparser.parse(url)
        entries = feed.entries[:10]  # Prend les 10 dernières actus
        df_news = pd.DataFrame([{
            "title": e.title,
            "link": e.link,
            "pubDate": e.published,
            "source": "CoinTelegraph"
        } for e in entries])
        return df_news
    except Exception as e:
        return pd.DataFrame([{
            "title": f"Erreur récupération flux RSS : {e}",
            "link": "",
            "pubDate": "",
            "source": ""
        }])



# === Affichage des actualités crypto ===
st.subheader("📰 Actualités crypto (Solana & marché)")

news_df = get_crypto_news()
for _, row in news_df.iterrows():
    if row["link"]:
        st.markdown(f"- [{row['title']}]({row['link']}) ({row['source']} - {row['pubDate'][:10]})")
    else:
        st.warning(row["title"])


def ajouter_indicateurs_techniques(data):
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['20_STD'] = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['20_SMA'] + 2 * data['20_STD']
    data['Bollinger_Lower'] = data['20_SMA'] - 2 * data['20_STD']

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    stoch_rsi = (data['RSI'] - data['RSI'].rolling(14).min()) / (data['RSI'].rolling(14).max() - data['RSI'].rolling(14).min())
    data['StochRSI'] = stoch_rsi

    return data

st.subheader("📊 Indicateurs Techniques Avancés")
sol_data = ajouter_indicateurs_techniques(sol_data)
st.line_chart(sol_data[['MACD', 'Signal']])
st.line_chart(sol_data[['Close', 'Bollinger_Upper', 'Bollinger_Lower']])
st.line_chart(sol_data[['StochRSI']])

# ✅ 2. Heatmap volumes & volatilité
st.subheader("🔥 Heatmap : Volume & Volatilité")
sol_data['Volatility'] = sol_data['Close'].rolling(5).std()
heatmap_df = sol_data[['Volume', 'Volatility']].tail(30)
st.dataframe(heatmap_df.style.background_gradient(cmap='viridis'))

# ✅ 3. Sentiment crypto Twitter/Reddit (placeholder texte)
st.subheader("📢 Analyse de Sentiment (Twitter/Reddit)")
sentiment = "🔵 Twitter: Neutre\n🔴 Reddit: Négatif (peur sur les forums)"
st.code(sentiment)

# ✅ 4. Événements Solana
st.subheader("📅 Événements Solana à venir")
events = [
    {"date": "2025-06-01", "event": "Mise à jour protocole v2.1"},
    {"date": "2025-06-15", "event": "Conférence Solana Paris"},
    {"date": "2025-07-01", "event": "Lancement d'une dApp DeFi majeure"}
]
for e in events:
    st.markdown(f"- **{e['date']}** : {e['event']}")

# ✅ 5. Portefeuille fictif
st.subheader("💼 Portefeuille fictif")
if 'wallet' not in st.session_state:
    st.session_state.wallet = {'sol_quantity': 5.0, 'entry_price': 100.0}  # 5 SOL achetés à 100$

total_value = st.session_state.wallet['sol_quantity'] * last_price
gain = total_value - (st.session_state.wallet['sol_quantity'] * st.session_state.wallet['entry_price'])

col1, col2 = st.columns(2)
col1.metric("Valeur actuelle du portefeuille", f"${total_value:.2f}")
col2.metric("Gains/Pertes", f"${gain:.2f}", delta_color="normal")

# Bouton d’achat manuel pour test (simulation)
if st.button("Simuler Achat 1 SOL au prix actuel"):
    st.session_state.wallet['sol_quantity'] += 1
    st.session_state.wallet['entry_price'] = (st.session_state.wallet['entry_price'] + last_price) / 2
    st.success("1 SOL ajouté au portefeuille fictif.")
