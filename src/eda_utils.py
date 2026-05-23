import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

# Set plotting style for clean premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of missing values in the DataFrame.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_pct
    })
    return summary[summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculates lower/upper bounds and identifies outliers using the IQR method.
    """
    data = df[column].dropna()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    print(f"--- Outlier Detection for '{column}' ---")
    print(f"25th Percentile (Q1): {q1:.2f}")
    print(f"75th Percentile (Q3): {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outlier Count: {outlier_count} ({outlier_pct:.2f}% of total rows)\n")
    
    return outliers

def calculate_grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculates key insurance metrics grouped by a specified column.
    Metrics: Total Premium, Total Claims, Loss Ratio, Margin, and Policy Count.
    """
    grouped = df.groupby(group_col).agg(
        PolicyCount=('PolicyID', 'count'),
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).reset_index()
    
    grouped['Margin'] = grouped['TotalPremium'] - grouped['TotalClaims']
    grouped['LossRatio'] = np.where(
        grouped['TotalPremium'] != 0,
        grouped['TotalClaims'] / grouped['TotalPremium'],
        np.nan
    )
    
    return grouped.sort_values(by='PolicyCount', ascending=False)

def plot_univariate_distributions(
    df: pd.DataFrame, 
    num_cols: List[str], 
    cat_cols: List[str], 
    save_dir: Optional[str] = None
):
    """
    Plots histograms for numerical columns and bar charts for categorical columns.
    """
    # 1. Plot numerical columns
    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(7, 3.5))
            sns.histplot(df[col].dropna(), kde=True, bins=50, color='royalblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/dist_{col.lower()}.png", dpi=150)
            plt.show()

    # 2. Plot categorical columns
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            # Limit to top 15 categories if cardinality is very high
            order = df[col].value_counts().iloc[:15].index
            sns.countplot(data=df, y=col, order=order, palette='viridis', hue=col, legend=False)
            plt.title(f'Top Categories in {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/bar_{col.lower()}.png", dpi=150)
            plt.show()

def plot_bivariate_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    hue_col: Optional[str] = None, 
    save_path: Optional[str] = None
):
    """
    Generates a scatter plot to observe relationships between two numerical columns.
    """
    plt.figure(figsize=(7.5, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6, palette='coolwarm')
    plt.title(f'{y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, cols: List[str], save_path: Optional[str] = None):
    """
    Calculates and plots a correlation matrix heatmap for specified numerical columns.
    """
    plt.figure(figsize=(7, 5.5))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Key Numerical Variables')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_outliers_box(df: pd.DataFrame, cols: List[str], save_path: Optional[str] = None):
    """
    Generates box plots to visually detect outliers in numerical features.
    """
    num_plots = len(cols)
    fig, axes = plt.subplots(1, num_plots, figsize=(4.5 * num_plots, 4.5))
    if num_plots == 1:
        axes = [axes]
        
    for i, col in enumerate(cols):
        sns.boxplot(data=df, y=col, ax=axes[i], color='lightcoral')
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_ylabel(col)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
