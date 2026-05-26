import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def run_chi2_test(df: pd.DataFrame, group_col: str, target_col: str):
    """Runs a Chi-Squared Test for categorical variables."""
    logger.info(f"Chi-Squared test: {group_col} vs {target_col}")
    
    # Create contingency table
    contingency_table = pd.crosstab(df[group_col], df[target_col])
    
    # Run test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return contingency_table, chi2, p_value


def run_ttest(df: pd.DataFrame, group_col: str, group_a_val, group_b_val, target_col: str):
    """Runs Welch's t-test comparing means of target_col between two groups."""
    logger.info(f"t-test: {group_col}({group_a_val} vs {group_b_val}) on {target_col}")
    
    # Segment groups
    group_a = df[df[group_col] == group_a_val][target_col].dropna()
    group_b = df[df[group_col] == group_b_val][target_col].dropna()
    
    # Calculate means
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    return mean_a, mean_b, t_stat, p_value
