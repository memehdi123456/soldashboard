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
