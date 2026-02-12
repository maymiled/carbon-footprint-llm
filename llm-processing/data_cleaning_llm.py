import os
import pandas as pd
from openai import OpenAI


# Name : OPENAI_API_KEY
# Value : sk-xxxx
os.environ["OPENAI_API_KEY"] = "mykey"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4.1-nano"

SHEET_NAME = "Tab_DE"
COL_SHORT = "Intitulé court (fi)"
COL_LONG  = "Libellé du bien (fi)"
INPUT_XLSX  = "/content/FE_ADEME.xlsx"
OUTPUT_XLSX = "/content/FE_ADEME_final.xlsx"

LABELS = [
    "citadines",
    "berlines",
    "SUV",
    "monospaces",
    "utilitaires",
    "poids lourds",
    "vehicules speciaux",
    "materiel",
    "unknown",
]

def infer_one_line(text: str) -> str:

    prompt = (
        "Tu es un classifieur strict.\n"
        "Tu dois répondre par UN SEUL MOT exactement parmi :\n"
        "citadines, berlines, SUV, monospaces, utilitaires, "
        "poids lourds, vehicules speciaux, materiel.\n\n"
        "Règles :\n"
        "- équipement, borne, chargeur, chaise, aménagement, cloison -> materiel\n"
        "Souvent il y a la plaque d'immatriculation pour les voitures !"
        "- pas de phrase, pas d'explication\n\n"
        f"Texte :\n{text}"
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.0,
        max_output_tokens=16,   # minimum accepté
    )

    out = resp.output_text.strip()

    for lab in LABELS:
        if lab.lower() == out.lower():
            # print(lab, "->", lab)
            return lab

    print(text)
    return "unknown"


def process_excel_vehicule(input_xlsx: str, output_xlsx: str):
    df = pd.read_excel(input_xlsx, sheet_name=SHEET_NAME)

    texts = (
        df[COL_SHORT].fillna("").astype(str)
        + " "
        + df[COL_LONG].fillna("").astype(str)
    )


    df["segment_llm"] = texts.apply(infer_one_line)
    df.to_excel(output_xlsx, index=False)

    # print("OK :", output_xlsx)






# =====================
# LLM
# =====================
def infer_four_fields(context_text: str):
    prompt = (
        "En français. Tu es un extracteur strict de facteurs d’émission.\n"
        "Tu dois produire EXACTEMENT 4 lignes. Pas plus. Pas moins.\n\n"
        "FORMAT STRICT :\n"
        "VALUE: <nombre>\n"
        "UNIT: <unité>\n"
        "PRODUCT: <nom du produit ou du procédé>\n"
        "FUNCTIONAL_UNIT: <phrase complète>\n\n"
        "Règles :\n"
        "- VALUE et UNIT doivent être cohérents avec les données\n"
        "- PRODUCT = nom court du produit ou procédé\n"
        "- FUNCTIONAL_UNIT = une PHRASE COMPLÈTE décrivant l’unité fonctionnelle du service rendu\n"
        "- Aucun texte en dehors des 4 lignes\n\n"
        f"Contexte (ligne ADEME complète) :\n{context_text}"
    )

    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0.0,
        max_output_tokens=64,
    )

    value = ""
    unit = ""
    product = ""
    fu = ""

    for line in resp.output_text.strip().splitlines():
        if line.startswith("VALUE:"):
            value = line.replace("VALUE:", "").strip()
        elif line.startswith("UNIT:"):
            unit = line.replace("UNIT:", "").strip()
        elif line.startswith("PRODUCT:"):
            product = line.replace("PRODUCT:", "").strip()
        elif line.startswith("FUNCTIONAL_UNIT:"):
            fu = line.replace("FUNCTIONAL_UNIT:", "").strip()
    #print(value)
    #print(unit)
    #print(product)
    #print(fu)
    #print("\n\n")
    return value, unit, product, fu


def process_excel():
    df = pd.read_excel(INPUT_XLSX)

    df["Value"] = ""
    df["Unit"] = ""
    df["Product or process"] = ""
    df["Functional unit"] = ""

    for i, row in df.iterrows():
        # CONTEXTE = TOUTE LA LIGNE
        print(i)
        context = "\n".join(f"{k}: {v}" for k, v in row.items())

        value, unit, product, fu = infer_four_fields(context)

        df.at[i, "Value"] = value
        df.at[i, "Unit"] = unit
        df.at[i, "Product or process"] = product
        df.at[i, "Functional unit"] = fu

    df.to_excel(OUTPUT_XLSX, index=False)

# RUN
process_excel()
process_excel_vehicule("VEHICULES.xlsx", "VEHICULES_segmentation_openai_simple.xlsx")
