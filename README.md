# AlphaCare Insurance Solutions (ACIS) - Auto Insurance Risk Analytics

This repository contains the marketing risk and predictive analytics project for AlphaCare Insurance Solutions (ACIS) in South Africa. The objective is to analyze historical car-insurance claim data (Feb 2014 – Aug 2015) to optimize marketing strategies, identify low-risk customer segments for potential premium reductions, and build a predictive pricing engine.

---

## Progress & Tasks completed

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
│   └── 01_eda.ipynb       # Polished EDA and visual notebook
├── reports/
│   ├── final_report.md    # Actionable Business Insights Report
│   └── *.png              # Visual plots from EDA analysis
├── scripts/
│   └── clean_data.py      # Automated data cleaning pipeline (Task 2)
├── src/                   # Source code library
│   ├── __init__.py
│   ├── data_loader.py     # Clean loaders and preprocessors (Task 1)
│   └── eda_utils.py       # Helper functions for modular visual calculations
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

### 3. Run Pipeline Scripts
To reproduce the data-cleaning pipeline manually:
```bash
python scripts/clean_data.py
```

### 4. Run Tests & Validation
Verify that the codebase complies with modular requirements and unit tests:
```bash
pytest
```
