# Dataset Description

## Overview

This repository contains the processed training dataset used to train the CarbonAttentionModel.

File:
- `processed/training_set_finish.csv`

This dataset is the result of a hybrid data engineering pipeline combining:

- Public emission factor databases (ADEME)
- Scraped medical emission factors (HealthcareLCA)
- LLM-based schema harmonization and normalization

---

## Data Sources

### 1. ADEME Emission Factors (Public)

The ADEME database provides ~14,000 emission factor entries
covering various economic sectors.

However:
- Product descriptions are heterogeneous
- Units are inconsistent
- Functional units vary widely

An LLM-based extraction pipeline was used to standardize:
- Emission value
- Unit
- Product name
- Functional unit

---

### 2. HealthcareLCA (Scraped Medical Dataset)

We scraped 1,094 domain-specific medical emission factors
covering:

- Surgical equipment
- Pharmaceuticals
- Medical procedures

This was necessary because medical devices are poorly represented
in generic emission databases.

---

## Data Processing Pipeline

1. Text normalization
2. Unit harmonization
3. Schema alignment via LLM extraction
4. Removal of inconsistent / zero emission entries
5. Log transformation applied during training:
   
   y = log(1 + CO2)

---

## Final Dataset Structure

The file `training_set_finish.csv` contains:

- `DETAILS`: textual description of the product or process
- `FE.VAL`: emission factor value (kgCO2e)

Only strictly positive emission values were kept.

---

## Confidential Data

Hospital procurement data from CHU Rennes used for final evaluation
cannot be shared due to confidentiality constraints.

This repository therefore includes only public and scraped data.
