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
