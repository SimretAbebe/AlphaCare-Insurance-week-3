# AlphaCare Insurance Solutions (ACIS) - Auto Insurance Risk Analytics

This repository contains the marketing risk and predictive analytics project for AlphaCare Insurance Solutions (ACIS) in South Africa. The objective is to analyze historical car-insurance claim data (Feb 2014 – Aug 2015) to optimize marketing strategies, identify low-risk customer segments for potential premium reductions, and build a predictive pricing engine.

---

## Progress & Tasks Completed

### Task 1: Git, GitHub, CI/CD Pipeline & Exploratory Data Analysis (EDA)
- **CI/CD Pipeline**: Configured GitHub Actions CI (`.github/workflows/ci.yml`) to automatically run code style checks (`flake8`) and unit tests (`pytest`) on every commit or pull request.
- **Modular Architecture**: Built reusable loaders (`src/data_loader.py`) and visualization engines (`src/eda_utils.py`).
- **Comprehensive Jupyter Notebook**: Created a polished, narrative-driven EDA notebook (`notebooks/01_eda.ipynb`) that covers:
  - Data structure & quality checks.
  - Descriptive statistics & financial distribution analysis (Premiums, Claims).
  - Demographic & geographic risk profiling.
  - Correlation analyses & high-quality visual insights.
- **Reporting**: Generated a detailed, business-driven [Final EDA Report](reports/final_report.md) complete with key insights, province risk maps, and actionable recommendations.
- **Testing**: Built comprehensive test cases under `tests/` to guarantee pipeline integrity.

### Task 2: Data Version Control (DVC)
- **DVC Initialization**: Initialized a reproducible DVC workspace tracking major datasets without bloating Git history.
- **Remote Storage Setup**: Configured external local remote storage (`C:\Users\HP\Desktop\dvc_storage`) mapping dataset hashes to lightweight Git metadata pointers.
- **Data Pipeline Scripts**: Developed `scripts/clean_data.py` to automate a complete data cleaning and preprocessing pipeline (handling extreme nulls, removing duplicates, and imputing numeric/categorical features).
- **Multi-Version Tracking**: Tracked and verified both:
  - **Version 1**: The original pipe-delimited Raw Dataset.
  - **Version 2**: The automated cleaned and standardized Dataset.

### Task 3: Statistical Hypothesis Testing
- **Statistical Framework (`src/hypothesis_tests.py`)**: Built clean, reusable functions to automate Welch's t-test and Chi-Squared contingency analyses.
- **Hypotheses Explored (`notebooks/02_hypothesis_testing.ipynb`)**:
  - *Hypothesis 1 (Province Risk)*: Tested if risk/claims are significantly different between provinces (e.g., Gauteng vs others).
  - *Hypothesis 2 (Gender Risk)*: Analyzed risk profiles between male and female policyholders.
  - *Hypothesis 3 (Postal Code Claims)*: Assessed geographic variance in claim distribution across various postal codes.
  - *Hypothesis 4 (Marital Status Differences)*: Examined if marital status plays a significant role in claim probability.
- **A/B Testing Insights**: Used statistical p-values to validate risk differentials, providing statistical backup for geographic and demographic premium adjustments.

### Task 4: Statistical Modeling & Risk-Based Pricing
- **Reusable Modeling Library (`src/modeling.py`)**:
  - Implemented risk feature engineering (`vehicle_age`, `risk_density`, `insured_per_age`) while excluding `totalpremium` to prevent data leakage.
  - *Claim Severity (Regression)*: Log-transforms the target (`np.log1p`) with log-space clipping and `np.expm1` inverse-transformation to handle highly skewed claim distributions.
  - *Claim Probability (Classification)*: Managed severe class imbalance via negative class downsampling (10% ratio) and balance weighting (`class_weight='balanced'` and `scale_pos_weight`).
- **End-to-End Analytics Notebook (`notebooks/03_modeling.ipynb`)**:
  - Fits, compares, and evaluates Linear Regression, Random Forest, and XGBoost models.
- **Actuarial Calibration & Optimization**:
  - Solves the probability scaling inflation from downsampling using an actuarial calibration factor (`0.019544`) to match portfolio claims (64.86 ZAR average).
  - Implements the dynamic risk-based pricing formula: `opt_premium = Calibrated Pure Premium + 50 (Expense) + 20 (Profit)`.
- **Model Interpretability (SHAP)**: Employs SHAP summary visualizations to determine top exposure drivers (`suminsured`, interaction risk density, and vehicle age).

---

## Project Structure

```bash
├── .dvc/                  # DVC internal configuration files
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions Continuous Integration pipeline
├── data/                  # Data directory (tracked by DVC pointer files)
│   └── insurance_data.csv.dvc
├── notebooks/
│   ├── 01_eda.ipynb       # Polished EDA and visual notebook (Task 1)
│   ├── 02_hypothesis_testing.ipynb # A/B Testing & statistical verification (Task 3)
│   └── 03_modeling.ipynb  # Predictive modeling and premium optimization (Task 4)
├── reports/
│   ├── final_report.md    # Actionable Business Insights Report
│   ├── premium_comparison.png # Calibrated premiums vs baseline premium comparison (Task 4)
│   ├── shap_summary_plot.png # XGBoost SHAP value feature importance (Task 4)
│   └── *.png              # Visual plots from EDA analysis
├── scripts/
│   └── clean_data.py      # Automated data cleaning pipeline (Task 2)
├── src/                   # Source code library
│   ├── __init__.py
│   ├── data_loader.py     # Clean loaders and preprocessors (Task 1)
│   ├── eda_utils.py       # Helper functions for modular visual calculations (Task 1)
│   ├── hypothesis_tests.py # Welch's t-test and Chi-Squared contingency tests (Task 3)
│   └── modeling.py        # Dataset preprocessors and regression/classification engines (Task 4)
├── tests/
│   └── test_data_loader.py# Pytest unit tests for Loader
├── .dvcignore
├── .gitignore
├── README.md              # Project overview and documentation
└── requirements.txt       # Project python dependencies
```

---

## 🛠️ How to Get Started

### 1. Clone the repository and setup the environment
```bash
git clone https://github.com/SimretAbebe/AlphaCare-Insurance-week-3.git
cd AlphaCare-Insurance-week-3

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure DVC Remote (Audit / Reproduction)
Ensure DVC is pointing to the appropriate remote local storage path to download the datasets:
```bash
# Pull the specific tracked dataset version from remote
dvc pull
```

### 3. Run Pipeline Scripts & Modeling
* **Data Cleaning**: `python scripts/clean_data.py`
* **Run Unit Tests**: `pytest`
* **Run Notebooks**: Open Jupyter and run `01_eda.ipynb`, `02_hypothesis_testing.ipynb`, and `03_modeling.ipynb` in sequence.
```
