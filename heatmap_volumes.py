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
