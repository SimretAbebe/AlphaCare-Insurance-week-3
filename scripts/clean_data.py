import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "MachineLearningRating_v3", "MachineLearningRating_v3.txt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_data.csv")

def clean_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    original_shape = df.shape
    print(f"  Original shape: {original_shape}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop columns with >70% missing values
    threshold = 0.7
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns with >70% missing: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # Drop rows where all values are NaN
    df.dropna(how="all", inplace=True)

    # Drop exact duplicate rows
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before_dedup - len(df)} duplicate rows")

    # Fill missing numeric columns with median (pandas 3 compatible)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Fill missing categorical/string columns with mode (pandas 3 compatible)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Remaining nulls: {df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    df_clean = clean_data(RAW_PATH)
    df_clean.to_csv(OUTPUT_PATH, sep="|", index=False)
    print(f"\n[DONE] Cleaned data saved to: {OUTPUT_PATH}")
