import pandas as pd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to load, clean, and preprocess the AlphaCare car insurance claims dataset.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the pipe-delimited dataset and performs initial structural analysis.
        """
        logger.info(f"Attempting to load dataset from {self.file_path}...")
        try:
            # Load the data using pipe delimiter
            self.df = pd.read_csv(self.file_path, sep='|', low_memory=False)
            
            # Clean column names by removing any leading/trailing spaces
            self.df.columns = self.df.columns.str.strip()
            
            logger.info(f"Successfully loaded dataset. Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            logger.info(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found at: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise

    def preprocess(self) -> pd.DataFrame:
        """
        Cleans whitespaces, parses datetime columns, and calculates derived metrics (Loss Ratio, Margin).
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Starting data preprocessing...")
        
        # 1. Strip whitespace from all string columns
        string_cols = self.df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            self.df[col] = self.df[col].astype(str).str.strip()
            # Replace empty strings or whitespace-only values with NaN where appropriate
            self.df[col] = self.df[col].replace({'': np.nan, 'nan': np.nan, 'Not specified': np.nan})

        # 2. Parse Datetime columns
        logger.info("Parsing date columns...")
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
        
        if 'VehicleIntroDate' in self.df.columns:
            self.df['VehicleIntroDate'] = pd.to_datetime(self.df['VehicleIntroDate'], errors='coerce')

        # 3. Calculate derived metrics
        logger.info("Calculating derived insurance metrics (Margin, Loss Ratio)...")
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            # Margin = TotalPremium - TotalClaims
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            
            # Loss Ratio = TotalClaims / TotalPremium (handling division by zero)
            self.df['LossRatio'] = np.where(
                self.df['TotalPremium'] != 0,
                self.df['TotalClaims'] / self.df['TotalPremium'],
                np.nan  # Set to NaN or 0 where premium is zero
            )
            
        logger.info("Preprocessing complete.")
        return self.df
