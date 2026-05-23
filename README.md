# AlphaCare Insurance Solutions (ACIS) - Auto Insurance Risk Analytics

This repository contains the marketing risk and predictive analytics project for AlphaCare Insurance Solutions (ACIS) in South Africa. The objective is to analyze historical car-insurance claim data (Feb 2014 – Aug 2015) to optimize marketing strategies, identify low-risk customer segments for potential premium reductions, and build a predictive pricing engine.

## Business Objectives
*   **Customer Segmentation:** Identify low-risk client segments (by province, postal code, gender, etc.) to target for optimized premium structures.
*   **Risk-Based Pricing Engine:** Develop predictive models to estimate:
    *   **Claim Frequency/Probability:** The probability that a client will make a claim.
    *   **Claim Severity:** The estimated cost of a claim.
*   **Model Explainability:** Utilize SHAP/LIME to explain features driving pricing decisions.
*   **Business Intelligence:** Synthesize findings into a business report with actionable recommendations.

## Key Metrics
*   **Loss Ratio** = $\frac{\text{TotalClaims}}{\text{TotalPremium}}$ (Portfolio profitability measure)
*   **Margin** = $\text{TotalPremium} - \text{TotalClaims}$ (Net profit contribution per policy)

## Technologies Used
*   **Python:** Core data analysis and modeling.
*   **Pandas & NumPy:** Data preprocessing and exploratory analysis.
*   **Scikit-Learn, XGBoost:** Predictive modeling.
*   **SHAP/LIME:** Model interpretability.
*   **Git & GitHub:** Version control and collaboration.
*   **DVC (Data Version Control):** Managing large dataset pipelines.
