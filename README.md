# Global Terrorism Intelligence System (GTIS)
> **A Multi-Modal Machine Learning & Spatiotemporal Risk Analytics Suite**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

The **Global Terrorism Intelligence System (GTIS)** is an end-to-end predictive security and analytics platform built on the **Global Terrorism Database (GTD)** (181,691 historical incidents from 1970 to 2017). GTIS combines multi-modal machine learning algorithms with an interactive web dashboard to attribute unclaimed terror attacks, predict attack success probability, estimate casualty risk bounds, and forecast regional incident surges.

---

## Key Features & Machine Learning Suite

* **1. Perpetrator Group Attribution**: Multi-class LightGBM classifier that solves unclaimed terror attacks with **92.71% overall accuracy** and **99.96% Top-3 accuracy**.
* **2. Attack Success Predictor**: Cost-sensitive LightGBM classifier evaluating protective factors and attack vector success (**0.9059 ROC-AUC**).
* **3. Casualty Severity & Lethality Risk**: Dual ordinal classifier and quantile regressors ($q=0.50$ median casualties, $q=0.90$ high-risk upper bound) to bound extreme mass casualty outcomes.
* **4. 3D Geospatial Threat Hotspots**: Unsupervised DBSCAN density clustering identifying **118 global threat hotspots**.
* **5. Regional Volume Forecasting**: Regional monthly time-series lag regressor (**$R^2 = 0.7891$**).
* **6. Interactive Web Dashboard**: Built with Streamlit, Plotly, and Pydeck for real-time scenario simulation, 3D threat mapping, and diagnostic analytics.

---

## Project Directory Structure

```text
PROJECT_1/
├── app.py                     # Main Interactive Streamlit Web Application
├── run_all.py                 # Master Orchestrator Script for complete ML Pipeline
├── src/
│   ├── data_preprocessing.py  # Cleaning, imputation, feature engineering & temporal split
│   ├── model_perpetrator_attribution.py # Perpetrator multi-class classifier
│   ├── model_success_prediction.py      # Imbalanced attack success predictor
│   ├── model_casualty_estimation.py     # Quantile & ordinal casualty regressors
│   ├── model_spatiotemporal.py          # Spatial DBSCAN & regional lag forecaster
│   └── model_inference.py              # Real-time inference wrapper engine
├── data/                      # Preprocessed GTD datasets (.pkl)
├── models/                    # Serialized model binaries (.joblib)
├── outputs/                   # Performance metrics report (metrics.json)
├── .streamlit/                # Custom dark theme configuration
└── README.md                  # Project documentation
```

---

## Quick Start Guide

### 1. Prerequisites & Installation

Ensure you have **Python 3.10+** installed. Clone this repository and install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn plotly pydeck streamlit joblib
```

---
### 2. Download Dataset (`data/` Folder)

Due to file size constraints on GitHub, download the preprocessed `data/` folder from Google Drive:

📥 **[Download GTIS Data Folder (Google Drive)]((https://drive.google.com/drive/folders/1NGp33mCRjEL5wpEgxH63ZnWZnlDP3f1m?usp=sharing))**

After downloading, place the extracted `data/` folder directly inside the project root directory.

---


### 3. Run the Machine Learning Pipeline

To execute data preprocessing, train all 4 machine learning models, evaluate out-of-sample metrics, and export `.joblib` binaries and `outputs/metrics.json`:

```bash
python run_all.py
```

---

### 4. Launch the Interactive Web Dashboard

To start the interactive Streamlit intelligence suite:

```bash
streamlit run app.py
```

Once launched, open your web browser at **`http://localhost:8501`**.

---

## Dashboard Views

1. **Executive Summary & GTD Overview**: High-level KPIs, 47-year temporal trends, regional incident breakdowns, and target taxonomies.
2. **3D Geospatial Threat Hotspots**: Interactive 3D scatter map with region/country/year filters and top spatial cluster tables.
3. **Live ML Prediction Studio**: Configure tactical parameters and run live real-time inference across all 4 ML engines.
4. **Forecasting & Diagnostics**: Regional monthly incident volume forecasts and feature importance rankings.

---

## Technology Stack

* **Core Logic**: Python 3.13, Pandas, NumPy
* **Machine Learning**: LightGBM, Scikit-Learn, XGBoost, CatBoost, Joblib
* **Geospatial & Clustering**: DBSCAN, Pydeck, Mapbox
* **Interactive Dashboard**: Streamlit, Plotly Express, Plotly Graph Objects
