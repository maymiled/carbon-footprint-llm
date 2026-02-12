# 🌍 CarbonAttention  
## LLM-Based Carbon Footprint Estimation for Hospital Procurement & Mobility

### 🚀 Live Applications

[![Procurement Dashboard](https://img.shields.io/badge/Streamlit-Procurement%20Dashboard-red?logo=streamlit)](https://mathlesage-rdc-app-5zfflh.streamlit.app/)

[![Mobility Dashboard](https://img.shields.io/badge/Streamlit-Mobility%20Simulation-blue?logo=streamlit)](https://mathlesage-rdc-may-app-jaiyji.streamlit.app/)

This project was developed during the **Rennes Data Challenge 2026** in collaboration with CHU Rennes.

The objective was to design a data-driven system capable of:

- Estimating carbon emissions from hospital procurement data  
- Predicting emissions for medical devices without official emission factors  
- Modeling mobility-related emissions  
- Providing operational decision-support dashboards  

This repository presents a hybrid LLM-based architecture combined with domain-specific dataset engineering for sustainable decision support in healthcare systems.

---

# 🧠 1. Core Model — CarbonAttentionModel

**Location:** `train_model/`

The central contribution of this project is a hybrid neural architecture designed to estimate carbon emissions directly from textual product descriptions.

## 🔬 Architecture Overview

The model combines:

- **Frozen Qwen3-Embedding-0.6B encoder**
- Instruction-based embedding prompt
- Layer normalization
- Multi-head self-attention (8 heads)
- Mean pooling
- MLP regression head

Prediction target transformation:

```
y_hat = log(1 + CO2)
```

### Why freeze the encoder?

- Reduces overfitting  
- Lowers VRAM usage  
- Improves environmental efficiency of training  
- Enables lightweight adaptation  

This design leverages powerful semantic embeddings while keeping the model computationally efficient.

---

## ⚙ Training Strategy

Implemented in: `train_model/train.py`

- Dataset split: 80% train / 10% validation / 10% test  
- Loss: Mean Squared Error  
- Optimizer: AdamW  
- Scheduler: OneCycleLR  
- Gradient clipping  
- Early stopping  
- Log transformation of target values  

The encoder remains frozen during training. Only the attention and regression layers are optimized.

---

# 🗂 2. Dataset Engineering

**Locations:**  
`data/`  
`llm-processing/`

The training dataset is a hybrid construction combining multiple sources.

---

## 1️⃣ ADEME Public Emission Database

~14,000 emission factor entries covering various economic sectors.

Challenges:

- Heterogeneous product descriptions  
- Inconsistent units  
- Variable functional units  

---

## 2️⃣ HealthcareLCA (Scraped Medical Dataset)

~1,094 domain-specific medical emission factors including:

- Surgical equipment  
- Pharmaceuticals  
- Medical procedures  

This dataset was necessary because medical devices are poorly represented in generic emission databases.

---

## 🤖 LLM-Based Harmonization Pipeline

Location: `llm-processing/data_cleaning_llm.py`

An LLM extraction pipeline was used to:

- Normalize product descriptions  
- Extract emission values  
- Standardize units  
- Align functional units  

This process enabled the construction of a unified and coherent training dataset.

The final processed dataset used for training is located in:

`data/processed/training_set_finish.csv`

Only strictly positive emission values were retained.

---

## ⚠ Data Availability

Only public and scraped datasets are included in this repository.

Confidential hospital procurement data used during final evaluation cannot be shared due to institutional constraints.

---

# 📊 3. Streamlit Dashboards

Two operational applications were developed to demonstrate real-world usage.

---

## 🖥 App 1 — Procurement Emission Estimation

Location: `dashboards/app1/`

Features:

- Upload procurement list  
- Estimate CO2 emissions per product  
- Aggregate emission visualization  

Run locally:

```bash
cd dashboards/app1
pip install -r requirements.txt
streamlit run app_forecast_carbon.py
```

---

## 🚗 App 2 — Mobility & Vehicle Simulation

Location: `dashboards/app2/`

Features:

- Employee mobility modeling  
- Vehicle segmentation via LLM  
- KMeans hub simulation  
- Heatmap visualization  

Run locally:

```bash
cd dashboards/app2
pip install -r requirements.txt
streamlit run app.py
```

---

# 🧪 4. Repository Structure

```
train_model/        → Model architecture & training
llm-processing/     → LLM-based data normalization
dashboards/         → Streamlit applications
data/               → Processed public dataset
reports/            → Project reports
```

---

# 🔬 Research & Technical Contributions

- Hybrid LLM + attention-based regression architecture  
- Frozen large language model embedding strategy  
- Instruction-aware embedding design  
- LLM-assisted dataset harmonization  
- Domain adaptation to medical emission estimation  
- Operational dashboard deployment for decision support  

---

# 👥 Team

Developed during Rennes Data Challenge 2026.

- May Miled  
- Giuliano Aldarwish  
- Mathéo Quatreboeufs  
- Yacine Abdelouhab  
- Rachelle Nasr  
- Paul Hoerter  

---

# 📄 License

This repository contains only public or scraped datasets.  
Institutional data from CHU Rennes is excluded.

The code is released for research and educational purposes.
