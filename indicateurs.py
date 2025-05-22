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
