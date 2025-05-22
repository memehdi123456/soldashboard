import pandas as pd
import ta

def ajouter_indicateurs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs techniques : MACD, Bollinger Bands, RSI Stochastique.
    Corrige les problèmes de dimension et d'alignement.
    """

    # Assure un index propre
    data = data.copy().reset_index(drop=True)

    # MACD
    try:
        macd_ind = ta.trend.MACD(close=data['Close'])
        data['MACD'] = pd.Series(macd_ind.macd().values, index=data.index)
        data['MACD_signal'] = pd.Series(macd_ind.macd_signal().values, index=data.index)
    except Exception as e:
        data['MACD'] = None
        data['MACD_signal'] = None

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['bb_high'] = pd.Series(bb.bollinger_hband().values, index=data.index)
        data['bb_low'] = pd.Series(bb.bollinger_lband().values, index=data.index)
        data['bb_mavg'] = pd.Series(bb.bollinger_mavg().values, index=data.index)
    except:
        data['bb_high'] = data['bb_low'] = data['bb_mavg'] = None

    # RSI Stochastique
    try:
        stoch_rsi = ta.momentum.StochRSIIndicator(close=data['Close'], window=14, smooth1=3, smooth2=3)
        data['stoch_rsi_k'] = pd.Series(stoch_rsi.stochrsi_k().values, index=data.index)
        data['stoch_rsi_d'] = pd.Series(stoch_rsi.stochrsi_d().values, index=data.index)
    except:
        data['stoch_rsi_k'] = data['stoch_rsi_d'] = None

    return data


def afficher_indicateurs(data: pd.DataFrame, st):
    """
    Affiche les indicateurs techniques dans des graphiques Streamlit.
    """
    import matplotlib.pyplot as plt

    st.subheader("📉 Indicateurs Techniques")

    fig1, ax1 = plt.subplots()

    valid_rows = data[['Date', 'bb_low', 'bb_high', 'Close']].dropna()

    ax1.plot(valid_rows['Date'], valid_rows['Close'], label="Prix", color='blue')
    ax1.plot(valid_rows['Date'], valid_rows['bb_high'], label="Bollinger Haut", linestyle='--', color='orange')
    ax1.plot(valid_rows['Date'], valid_rows['bb_low'], label="Bollinger Bas", linestyle='--', color='green')
    ax1.fill_between(valid_rows['Date'], valid_rows['bb_low'], valid_rows['bb_high'], alpha=0.1)

    ax1.set_title("Bandes de Bollinger")
    ax1.legend()
    st.pyplot(fig1)

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
