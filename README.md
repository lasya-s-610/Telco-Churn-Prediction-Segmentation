# Telco Customer Churn Analysis

**Problem:** Telecom companies lose customers every month. 
Using a telecom dataset of 7,043 customers (1,869 churners), this project builds a churn prediction system, prioritizes high-risk users, and segments churners into distinct behavioral groups.

Most churn (≈70%) comes from early-stage customers, while high-value long-term users reveal a critical retention gap.

Three focused notebooks covering EDA, prediction with risk segmentation, and churner clustering.

---


## 📊 Project Overview

Explored customer demographics, service usage patterns, billing data, and churn behavior to:
- Predict which customers are likely to churn
- Segment customers into risk bands based on predictions  
- Profile actual churned customers into distinct personas


**Key results:**
- Random Forest model: **80% churn recall**, 74% accuracy
- Risk bands: High-risk customers actually churn at **57%**
- 3 churner personas identified from 1,869 actual churners

---
## 📁 Data & Files Explained

| File/Folder | What it contains |
|-------------|------------------|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original raw dataset  |
| `Telco_churn.csv` | Preprocessed dataset |
| `test_data_predictions.csv` | Model outputs (test set: Actual Churn, Churn Probability, Risk Segment) |
| `churn_rf.pkl` | Saved Random Forest model |
| `Churn_Prediction_EDA.ipynb` | EDA + churn pattern discovery |
| `Churn_Prediction_Model.ipynb` | Model training + risk segmentation |
| `Churn_segmentation.ipynb` | KMeans clustering of 1,869 actual churners |

---

## 🔍 Churn_Prediction_EDA.ipynb – Understanding Churn Patterns

**Dataset:** 21 features including tenure, MonthlyCharges, TotalCharges, Contract type, PaymentMethod, service flags (OnlineSecurity, TechSupport, etc.), and binary Churn target.

**Data handling:** Fixed blank TotalCharges for short-tenure customers. Noted ~27% churn rate (class imbalance).

Key churn drivers:
- Contract type: Month-to-month customers churn significantly more
- Service usage: Lack of OnlineSecurity/TechSupport correlates with higher churn
- Payment behavior: Electronic check users churn more than auto-payment users
- Lifecycle: 0–12 month customers are most vulnerable


---

## 🤖 Churn_Prediction_Model.ipynb

**Preprocessing:** 80/20 train/test split after encoding categoricals.

**Model comparison:**
| Model | Churn Recall | Notes |
|-------|--------------|--------|
| Logistic Regression | 54% | Missed too many churners |
| Decision Tree | 79% | Poor non-churn precision ~50% |
| **Random Forest** | **80%** | Best overall balance |

**Performance (test set):**
- Accuracy: 74%
- Confusion Matrix: TN=738, FP=295, FN=73, TP=301
- Churn: Precision=51%, Recall=80%
- Non-Churn: Precision=91%, Recall=71%


**Why recall matters:** Missing actual churners is more costly than over-targeting some retained customers.

**Risk segmentation (predicted probabilities):**
| Risk Band | Probability Range | Observed Churn Rate |
|-----------|-------------------|-------------------|
| High | ≥70% | **57%** |
| Medium | 50-70% | 35% |
| Low | 20-50% | 16.5% |
| Minimal | <20% | 6% |

---

## 🎯 Churn_segmentation.ipynb

**Approach:** KMeans clustering on **1,869 actual churned customers only** (independent of model predictions).

**Features used:** tenure, MonthlyCharges, TotalCharges, encoded Contract, PaymentMethod, and service flags. Features scaled using StandardScaler.

**Optimal clusters:** k=3 
- Elbow method shows a clear bend at k=3  
- Silhouette peaks at k=2, but k=3 chosen for better interpretability

**Cluster profiles (mean values):**
| Cluster | Size | Tenure (In Months) | MonthlyCharges | TotalCharges | Key Characteristics |
|---------|------|--------|----------------|--------------|-------------------|
| **0: High-value** | 437 (23.4%) | **45.9** | **$93.5** | **$4,313** | Some 1/2-year contracts, credit card payments |
| **1: Early moderate** | 1,319 (70.6%) | **9.6** | **$73** | **$727** | Mostly month-to-month |
| **2: Low-value** | 113 (6%) | **8.2** | **$20** | **$174** | Heavy mailed check usage |

**PCA visualization:**
- **PC1 (33%)** separates clusters by MonthlyCharges  
- **PC2 (16%)** captures tenure variation 

---


## 💡 Key Insights

- **70% of churners are early-stage customers** → onboarding and early engagement are critical  
- **High-value long-term customers still churn** → retention here requires service quality, not discounts  
- **Low-value users form a small but distinct segment** → retention vs acquisition trade-off  

- High-risk predictions translate into **~56–57% actual churn**, making risk bands actionable  
- Achieving **80% recall** ensures most churners are identified before exit  

→ **Combining prediction + risk stratification + clustering enables targeted and efficient retention strategies**

---


## 🛠 Tech Stack

Python (pandas, numpy, scikit-learn, matplotlib, seaborn), Jupyter Notebook

---

## 🚀 Next Steps

- Hyperparameter optimization (GridSearchCV)
- Model deployment (FastAPI/Flask)
- Experiment with boosted models (XGBoost)
- Build an interactive retention dashboard
