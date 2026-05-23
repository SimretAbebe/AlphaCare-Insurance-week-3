import os
import pandas as pd
import pytest
from src.data_loader import DataLoader

def test_data_loader(tmp_path):
    # Create a mock dataset file in the temporary test directory
    test_data = (
        "PolicyID|TransactionMonth|Gender|TotalPremium|TotalClaims\n"
        "123 | 2015-03-01 00:00:00| Male | 100.0 | 50.0 \n"
        "456 | 2015-04-01 00:00:00| Female | 200.0 | 0.0 \n"
        "789 | 2015-05-01 00:00:00| Not specified| 0.0 | 10.0 \n"
    )
    
    file_path = tmp_path / "mock_dataset.txt"
    file_path.write_text(test_data)
    
    # Initialize and load data
    loader = DataLoader(str(file_path))
    df = loader.load_data()
    
    # Check if loaded shape is correct (3 rows, 5 original columns)
    assert df.shape == (3, 5)
    
    # Run preprocessing
    df_clean = loader.preprocess()
    
    # Verify column whitespaces are stripped
    assert df_clean.columns.tolist() == [
        'PolicyID', 'TransactionMonth', 'Gender', 'TotalPremium', 'TotalClaims', 'Margin', 'LossRatio'
    ]
    
    # Verify string whitespaces are stripped
    assert df_clean.loc[0, 'Gender'] == 'Male'
    
    # Verify Nan conversions (e.g., 'Not specified' should be converted to NaN)
    assert pd.isna(df_clean.loc[2, 'Gender'])
    
    # Verify Datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(df_clean['TransactionMonth'])
    
    # Verify Margin calculations (TotalPremium - TotalClaims)
    assert df_clean.loc[0, 'Margin'] == 50.0
    assert df_clean.loc[1, 'Margin'] == 200.0
    assert df_clean.loc[2, 'Margin'] == -10.0
    
    # Verify LossRatio calculations (TotalClaims / TotalPremium)
    assert df_clean.loc[0, 'LossRatio'] == 0.5
    assert df_clean.loc[1, 'LossRatio'] == 0.0
    # Division by zero handled gracefully (Premium is 0, so LossRatio should be NaN)
    assert pd.isna(df_clean.loc[2, 'LossRatio'])
