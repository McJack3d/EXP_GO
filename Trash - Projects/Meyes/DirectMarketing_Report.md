# Direct Marketing Project — Results Report

**Team:** EXP_GO  
**Date:** auto-generated  
**Deliverables:** PPTX/PDF (slides) **and** a short video (3–5 minutes). This Markdown can be exported to PDF, and the included outline maps 1‑to‑1 to slides.

---

## 1) Project Overview (Slide 1–2)
**Goal:** Build propensity scores to target households for upcoming *direct marketing* actions (Wine & Gold product lines) and simulate campaign impact.

**Data split & protocol**
- **First part (review/exploration):** *Base dataset* `Project_Dataset_Base.sav` (n=740, p=15).
- **Modeling & evaluation:** *Training dataset* `Project_Dataset_J.sav` (n=1250, p=17).
- **Deployment:** Scores applied back onto the Base file to simulate target selection (Top‑20%).

**Key business question:** Can we rank customers by likelihood to respond (Wine/Gold), and what uplift do we expect in the top quintiles?

---

## 2) Data Summary (Slide 3–5)
### 2.1 Base dataset (used for review/exploration)
- Shape: **740 × 15**
- Missing values: `Income` has **7** missing; all other columns complete.
- Targets present in Base: none labeled as *wine/gold* (the Base file is used for exploration + deployment only).

**Descriptives (Base) — examples**
- `Year_Birth`: 1940–1995 (mean ≈ 1969)
- `Income`: mean ≈ 52,591; p25=34,600; p50=51,983; p75=69,109; max includes outlier (666,666)
- `Kidhome` / `Teenhome`: discrete 0–2

![Distribution analysis](assets/distribution_analysis.png)

### 2.2 Training dataset (used for modeling)
- Shape: **1250 × 17**
- Contains binary targets for **Wine** and **Gold** buyers (detected by column names).
- Categorical variables are one‑hot encoded; numeric features imputed (median) and scaled when relevant.

---

## 3) Score Construction (Slide 6–8)
### 3.1 Preprocessing
- **Pipelines** with `ColumnTransformer` ensure the **same transformations** for training and deployment:
  - Numeric: median imputation + (for LR/NN) scaling
  - Categorical: mode imputation + **OneHotEncoder(handle_unknown='ignore')**

### 3.2 Models trained (on Training set)
- **Logistic Regression** (baseline; stable and interpretable)
- **Neural Network (MLP)** (non‑linear)
- **Decision Tree** (interpretable, non‑linear; no scaling required)

**Validation:** Stratified 5‑fold **AUC** per target and model.  
**Train/test split:** Single split on *Wine* with `stratify=y_wine`, and *Gold* labels reindexed to the same indices (prevents label misalignment).

### 3.3 Score aggregation
- Standardize `SW` (Wine) and `SG` (Gold) → `SA = mean(SW_std, SG_std)` with ε‑guard for zero variance.
- Alternative combinations (harmonic / product) available; mean performed robustly.

---

## 4) Performance (Slide 9–11)
- **ROC/AUC** compared across LR / NN / DT for both targets.
- **Quintile analysis** on combined score:
  - Compute Q1–Q5 on test set; expect **monotonic lift** toward Q5.
  - **Top‑20%** (Q5) should exhibit the highest buyer rates; report **lift vs overall** for Wine and Gold.

> Insert the actual AUC and quintile metrics from your latest run (console logs / saved artifacts).

---

## 5) Deployment & Simulation (Slide 12–15)
- Trained pipelines applied to **Base** file.
- Output CSV: `/mnt/data/deployment_scores.csv` with columns:
  - `score_wine`, `score_gold`, `score_combined`, `quintile` (Q1–Q5), `top20_flag`

### Targeting scenario (simulate a direct marketing action)
- **Strategy:** contact **Top‑20%** (`top20_flag == 1`).
- **Expected effects:** (insert your assumed conversion / margin / cost)
  - Example inputs: Cost per contact = €0.50; Margin per conversion = €40
  - Let `N20` = number of customers in top‑20%; `conv_wine_top20`, `conv_gold_top20` = observed/estimated rates
  - **Expected profit** ≈ `N20 * (conv * margin − cost)`

> Replace with your team’s final assumptions and the counts printed by the script.

---

## 6) Risks, Bias & Next Steps (Slide 16–17)
- **Outliers in Income** (e.g., 666,666) can skew scaling → handled by pipeline but worth capping/winsorization sensitivity check.
- **Class imbalance** (typical for campaign responses): monitor precision‑recall and calibration.
- **Data drift**: re‑score periodically; retrain quarterly or when distributions shift.

**Next steps**
- Add calibrated probabilities (`CalibratedClassifierCV`).
- Consider **Monotonic GBM** / **XGBoost** with proper cross‑validation.
- Uplift modeling (two‑model approach / causal trees) for treatment effect targeting.

---

## 7) Appendix (Slide 18+)
- Environment: see `requirements.txt` (scikit‑learn pipelines, pyreadstat)
- Repro steps:

  ```bash
  python meyes.py
  # output: /mnt/data/deployment_scores.csv
  ```
- Assets included: `assets/distribution_analysis.png`

