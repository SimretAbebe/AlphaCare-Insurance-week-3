# Task 1: Exploratory Data Analysis & Environment Setup Report

This report summarizes the environment setup and exploratory data analysis (EDA) conducted on the AlphaCare Insurance Solutions auto insurance dataset from South Africa (Feb 2014 – Aug 2015).

## 1. Environment Setup & Code Structure

The project has been organized with a clean, modular structure:
- **`src/`**: Main Python package containing reusable modules:
  - `data_loader.py`: A robust `DataLoader` class designed to parse, clean, and preprocess the pipe-delimited data.
  - `eda_utils.py`: Reusable statistics and plotting functions.
- **`tests/`**: Unit testing suite using `pytest` to verify the accuracy of the data loader, derived metrics, and preprocessing functions.
- **`notebooks/`**: Walkthrough Jupyter notebook (`01_eda.ipynb`) displaying code, charts, and analysis.
- **`.github/workflows/`**: Continuous Integration (CI) pipeline using GitHub Actions to automatically run `flake8` linting and `pytest` on every commit and pull request.

---

## 2. Dataset Overview
- **Total Records**: 1,000,100 rows
- **Total Columns**: 52 columns (prior to additions)
- **Derived Columns Added**:
  - `Margin` = `TotalPremium` - `TotalClaims`
  - `LossRatio` = `TotalClaims` / `TotalPremium` (where `TotalPremium > 0`)

---

## 3. Data Quality & Missing Values
An initial analysis of missing values shows:
- Columns like `Gender`, `MaritalStatus`, and `Citizenship` contain placeholder text such as `"Not specified"`, which we have cleaned and set to `NaN` (null) for correct statistical analysis.
- Several optional features have high missingness, but core financial metrics (`TotalPremium`, `TotalClaims`) have 100% data completion.

---

## 4. Addressing Core Business Questions

### Question 1: Loss Ratio by Province, Vehicle Type, and Gender
* **Province**: Gauteng and Western Cape represent the largest premium volumes. The loss ratios differ by province, indicating different risk levels.
* **Vehicle Type**: Passenger and commercial vehicles have distinct claim rates. Commercial vehicles tend to carry higher claims risk.
* **Gender**: Comparing Male, Female, and unspecified genders shows variance in claim rates, which will be valuable for pricing segmentation.

### Question 2: Temporal Trends
* Monthly premiums and claims were aggregated by `TransactionMonth`.
* The claims pattern shows seasonal variation, with spikes in certain months.
* The overall Loss Ratio is monitored monthly to ensure underwriting profitability.

### Question 3: Vehicle Make Performance
* Top vehicle makes (e.g., Toyota, Volkswagen, Ford) represent the majority of policies.
* Underwriting margins vary significantly between makes; some makes exhibit disproportionately high loss ratios.

### Question 4: Financial Distributions
* Both `TotalPremium` and `TotalClaims` have highly skewed distributions with extreme outliers (very high premium and claim values).
* Analyzing these values on a log-scale reveals log-normal characteristics for positive values.
* The `Margin` distribution displays positive value clustering (profitable policies) alongside deep losses (unprofitable claims).

---

## 5. Visualizations Saved
The following visualizations have been successfully generated and saved to the `reports/` folder:
1. `reports/dist_totalpremium.png`: Distribution of Total Premiums
2. `reports/dist_totalclaims.png`: Distribution of Total Claims
3. `reports/dist_margin.png`: Distribution of Underwriting Margin
4. `reports/bar_province.png`: Top Provinces by policy count
5. `reports/bar_gender.png`: Gender distribution of policyholders
6. `reports/premium_vs_claims.png`: Bivariate scatter plot of Premiums vs. Claims
7. `reports/correlation_matrix.png`: Heatmap of correlation coefficients
8. `reports/loss_ratio_by_province.png`: Loss ratio across South African provinces
9. `reports/temporal_trends.png`: Premium, claim, and loss ratio trends over time
10. `reports/premium_by_make.png`: Top vehicle makes by premium volume
11. `reports/outliers_boxplot.png`: Box plots of premium and claim variables to identify extreme outliers
