# LinkedIn Post — Smart Health Risk Predictor

---

Paste the text below directly into LinkedIn. Add screenshots from reports/figures/ 
as images. Tag relevant people or hashtags as appropriate.

---

🚀 Excited to share my latest portfolio project: a complete end-to-end 
machine learning system for cardiovascular health risk prediction!

🫀 Project: Smart Health Risk Predictor
Built a system that classifies a patient's cardiovascular health risk 
as Low, Medium, or High — using 13 clinical indicators and a trained ML model.

🔬 What I built:
→ Exploratory Data Analysis on the UCI Heart Disease dataset (303 real patients)
→ Full preprocessing pipeline: imputation, scaling, encoding
→ 3 models trained and compared: Logistic Regression, Random Forest, XGBoost
→ Hyperparameter tuning with GridSearchCV (cross-validated)
→ FastAPI REST endpoint with Pydantic input validation
→ Clean responsive frontend (HTML/CSS/JS) — no framework needed
→ Deployed API on Hugging Face Spaces, UI on Netlify

📊 Results:
After tuning, XGBoost achieved ~85% accuracy and 0.83 macro F1 on the held-out 
test set. Top predictive features: max heart rate, ST depression, and vessel count — 
all of which align with clinical evidence.

💡 Key things I learned:
1. Data leakage is subtle — always fit preprocessing on training data only
2. Macro F1 matters more than accuracy when classes are imbalanced
3. FastAPI + Pydantic is an excellent combo for ML inference APIs
4. Deploying a real endpoint forces you to think about input validation 
   in a way that notebook code never does

The entire system is stateless (input → prediction → output), 
making it easy to reason about, test, and deploy.

🔗 GitHub: [link]
🌐 Live demo: [link]

This project is part of my university coursework and was built entirely in 
VS Code — from raw CSV to deployed API.

#MachineLearning #Python #FastAPI #HealthTech #DataScience #XGBoost 
#MLOps #StudentProject #Portfolio #AI
