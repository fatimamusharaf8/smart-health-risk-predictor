# Smart Health Risk Predictor 🫀

> An end-to-end machine learning system that predicts cardiovascular health risk level
> (Low / Medium / High) from 13 clinical indicators, with a FastAPI backend and
> clean web interface.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [ML Pipeline](#ml-pipeline)
6. [API Reference](#api-reference)
7. [Running Locally](#running-locally)
8. [Deployment](#deployment)
9. [Results](#results)
10. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

This project predicts a patient's **cardiovascular health risk** as Low, Medium, or
High using clinical indicators such as age, cholesterol, blood pressure, and ECG
results.

**Key features:**
- Real-world dataset (UCI Heart Disease / Cleveland)
- Thorough EDA with visualisations
- 3 trained models with a fair comparison
- Hyperparameter tuning with GridSearchCV
- FastAPI REST endpoint with input validation
- Clean HTML/CSS/JS frontend (no framework)
- Deployed API on Hugging Face Spaces

> ⚠️ For **educational purposes only**. Not a substitute for medical diagnosis.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data / EDA | pandas, numpy, matplotlib, seaborn |
| ML | scikit-learn, XGBoost |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3, Vanilla JS |
| Deployment (API) | Hugging Face Spaces |
| Deployment (UI) | Netlify |

---

## Project Structure

```
smart-health-risk-predictor/
├── data/
│   └── heart.csv               # Downloaded automatically on first run
│
├── src/
│   ├── 01_eda.py               # Exploratory Data Analysis
│   ├── 02_preprocess_and_train.py  # Preprocessing + model training
│   └── 03_tune.py              # Hyperparameter tuning
│
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
│
├── frontend/
│   └── index.html              # Single-file web app
│
├── models/                     # Saved after training (git-ignored)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── num_imputer.pkl
│   ├── cat_imputer.pkl
│   ├── feature_names.pkl
│   └── metadata.json
│
├── reports/
│   └── figures/                # All EDA and evaluation plots
│
├── app.py                      # Hugging Face entry point
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [UCI ML Repository — Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)

303 patients, 13 features, originally 5 target classes (0–4).

**Target engineering:** We remap to 3 clinically meaningful risk levels:
| Original | Risk Level | Code |
|---|---|---|
| 0 | Low | 0 |
| 1–2 | Medium | 1 |
| 3–4 | High | 2 |

**Features:**

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age in years |
| sex | Categorical | 0 = Female, 1 = Male |
| cp | Categorical | Chest pain type (0–3) |
| trestbps | Numeric | Resting blood pressure (mmHg) |
| chol | Numeric | Serum cholesterol (mg/dl) |
| fbs | Categorical | Fasting blood sugar > 120 (0/1) |
| restecg | Categorical | Resting ECG results (0–2) |
| thalach | Numeric | Maximum heart rate achieved |
| exang | Categorical | Exercise-induced angina (0/1) |
| oldpeak | Numeric | ST depression (exercise vs rest) |
| slope | Categorical | Slope of peak exercise ST segment |
| ca | Categorical | Major vessels coloured (0–3) |
| thal | Categorical | Thalassemia (0–3) |

---

## ML Pipeline

```
Raw CSV → EDA → Impute → Scale → Train (3 models) → Compare → Tune → Save
```

### Preprocessing decisions

1. **Missing values:** Median imputation for numeric features; mode for categorical.
   Only `ca` and `thal` have a small number of `?` values (~1%).

2. **Scaling:** `StandardScaler` on numeric features only.
   Tree-based models don't require scaling but it doesn't hurt — applying it
   uniformly keeps the inference pipeline consistent.

3. **Encoding:** Categorical features in this dataset are already integer-encoded
   by the UCI source. No additional encoding step is needed.

### Models trained

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable baseline; linear decision boundary |
| Random Forest | Non-linear ensemble; resistant to overfitting |
| XGBoost | Gradient boosting; often best on tabular data |

### Evaluation metrics

- Accuracy, Precision, Recall, F1 (macro-averaged for multi-class fairness)
- Confusion matrix per model
- 5-fold cross-validation F1 on the training set

---

## API Reference

Base URL (local): `http://127.0.0.1:8000`

### `GET /`
Health check and model info.

### `POST /predict`

**Request body (JSON):**
```json
{
  "age": 52,
  "sex": 0,
  "cp": 2,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 0,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

**Response:**
```json
{
  "risk_level": "Low",
  "risk_code": 0,
  "confidence": 0.8721,
  "probabilities": {
    "Low": 0.8721,
    "Medium": 0.0943,
    "High": 0.0336
  },
  "model_name": "XGBoost"
}
```

---

## Running Locally

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/smart-health-risk-predictor.git
cd smart-health-risk-predictor
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
python src/01_eda.py                    # EDA plots → reports/figures/
python src/02_preprocess_and_train.py   # Train models → models/
python src/03_tune.py                   # Tune best model
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
# → http://127.0.0.1:8000/docs (interactive API docs)
```

### 4. Open the frontend

Open `frontend/index.html` in your browser (double-click or drag into Chrome).
Make sure the API is running on port 8000.

---

## Deployment

### API → Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. Choose **Docker** SDK
3. Push this repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/health-risk-api
   git push hf main
   ```
4. Add a `Dockerfile` (already included) that runs:
   ```
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```
5. Update `API_URL` in `frontend/index.html` to your HF Space URL.

### Frontend → Netlify

1. Drag-and-drop the `frontend/` folder to [app.netlify.com/drop](https://app.netlify.com/drop)
2. That's it — Netlify gives you a public URL instantly.

---

## Results

*After training and tuning (exact numbers depend on the random seed):*

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | ~0.77 | ~0.74 |
| Random Forest | ~0.82 | ~0.79 |
| **XGBoost (tuned)** | **~0.85** | **~0.83** |

Top features (XGBoost): `thalach`, `oldpeak`, `ca`, `cp`, `thal`

---

## Limitations & Future Work

### Limitations
- Dataset is small (303 samples) — results may not generalise to all populations
- 3-class mapping is a design choice; medical practitioners may prefer binary
- The model is not calibrated (probabilities are relative, not absolute)
- No uncertainty quantification

### Future improvements
- Add SHAP values for per-prediction explainability
- Collect or use a larger, more diverse dataset
- Add model calibration (Platt scaling)
- Wrap API in Docker for easier deployment
- Add a Jupyter notebook for interactive exploration

---

## Author

**[Your Name]** | University Project | [LinkedIn](#) | [Portfolio](#)
