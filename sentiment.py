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
