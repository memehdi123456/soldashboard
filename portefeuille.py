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
