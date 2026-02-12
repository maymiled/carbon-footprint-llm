import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# --- 1. IMPORT DE TON ARCHITECTURE LOCALE ---
# Comme tu as le fichier model_utils.py à côté, on l'importe normalement
from model_utils import CarbonAttentionModel

# --- 2. CONFIGURATION ---
REPO_ID = "matheoqtb/carbon-qwen-predictor" # Ton repo HF
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. CHARGEMENT DES POIDS ---
print(f"Récupération des poids depuis Hugging Face...")
try:
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pt")
except Exception as e:
    print("exception:", e)
    print("Erreur de téléchargement, tentative avec le fichier local...")
    weights_path = "best_model.pt"

# Initialisation du modèle
model = CarbonAttentionModel(MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
instruction = "Instruct: Estime l'intensité carbone de cette ligne de base de données.\nQuery: "

# --- 4. TRAITEMENT DU FICHIER PRODUITS.CSV ---
# On charge ton fichier
path_csv = r"data\PRODUITS.csv"
try:
    df = pd.read_csv(path_csv, sep=";", encoding="latin-1")
except:
    # Backup si latin-1 ne suffit pas
    df = pd.read_csv(path_csv, sep=";", encoding="cp1252")
def get_prediction(text):
    if pd.isna(text) or text == "":
        return 0.0
    
    # Préparation du texte avec l'instruction
    full_text = f"{instruction}{text}<|endoftext|>"
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    with torch.no_grad():
        output = model(inputs['input_ids'], inputs['attention_mask'])
        # Transformation inverse du Log : exp(x) - 1
        prediction = torch.expm1(output).item()
    
    return round(prediction, 4)

print("Calcul des prédictions en cours...")

# On applique le modèle sur la colonne DB.lib
# Et on crée la colonne formatée : "Nom du produit - 12.34"
df['DB.lib - kgCO2/unite'] = df['DB.LIB'].apply(lambda x: f"{x} - {get_prediction(x)}")

# On crée aussi une colonne numérique pure si tu en as besoin pour des calculs
df['kgCO2_val'] = df['DB.lib'].apply(get_prediction)

# --- 5. SAUVEGARDE ---
df.to_csv("produits_resultats.csv", index=False, sep=";")

print("---")
print("Terminé ! Voici un aperçu :")
print(df['DB.lib - kgCO2/unite'].head())