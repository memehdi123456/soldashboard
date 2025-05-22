import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from indicateurs import ajouter_indicateurs, afficher_indicateurs
from heatmap_volumes import afficher_heatmap_volumes
from sentiment import afficher_sentiment
from events import afficher_evenements
from portefeuille import afficher_portefeuille

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
