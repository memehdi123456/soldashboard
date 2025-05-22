import pandas as pd
import ta

def ajouter_indicateurs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs techniques suivants :
    - MACD
    - Bollinger Bands
    - RSI Stochastique (%K et %D)
    """

    # MACD
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd().reset_index(drop=True)
    data['MACD_signal'] = macd.macd_signal().reset_index(drop=True)
    data['MACD'] = macd.macd().reset_index(drop=True).fillna(0)


    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_high'] = bollinger.bollinger_hband()
    data['bb_low'] = bollinger.bollinger_lband()
    data['bb_mavg'] = bollinger.bollinger_mavg()

    # RSI Stochastique
    stoch_rsi = ta.momentum.StochRSIIndicator(close=data['Close'], window=14, smooth1=3, smooth2=3)
    data['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
    data['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

    return data


def afficher_indicateurs(data: pd.DataFrame, st):
    """
    Affiche les indicateurs techniques dans des graphiques Streamlit.
    """
    import matplotlib.pyplot as plt

    st.subheader("📉 Indicateurs Techniques")

    fig1, ax1 = plt.subplots()
    ax1.plot(data['Date'], data['Close'], label="Prix", color='blue')
    ax1.plot(data['Date'], data['bb_high'], label="Bollinger Haut", linestyle='--', color='orange')
    ax1.plot(data['Date'], data['bb_low'], label="Bollinger Bas", linestyle='--', color='green')
    ax1.fill_between(data['Date'], data['bb_low'], data['bb_high'], alpha=0.1)
    ax1.set_title("Bandes de Bollinger")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(data['Date'], data['MACD'], label="MACD", color='purple')
    ax2.plot(data['Date'], data['MACD_signal'], label="Signal", color='red')
    ax2.set_title("MACD")
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(data['Date'], data['stoch_rsi_k'], label="%K", color='blue')
    ax3.plot(data['Date'], data['stoch_rsi_d'], label="%D", color='orange')
    ax3.axhline(80, linestyle='--', color='red', alpha=0.3)
    ax3.axhline(20, linestyle='--', color='green', alpha=0.3)
    ax3.set_title("RSI Stochastique")
    ax3.legend()
    st.pyplot(fig3)
