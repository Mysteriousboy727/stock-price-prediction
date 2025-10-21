"""
Data Cleaning Module
Handles missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Ensure root path resolution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataCleaner:
    """Clean and prepare stock data for analysis"""

    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize DataCleaner
        Parameters:
        -----------
        outlier_threshold : float
            Number of standard deviations for outlier detection
        """
        self.outlier_threshold = outlier_threshold
        self.cleaning_report = {}

    def clean_data(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        """
        print(f"\n{'='*60}")
        print(f"Cleaning data for {ticker}")
        print(f"{'='*60}")

        # Initialize report entry
        self.cleaning_report[ticker] = {
            'original_rows': len(df),
            'issues_found': []
        }

        df_clean = df.copy()

        # 1. Fix datetime index
        df_clean = self._fix_datetime_index(df_clean)

        # 2. Remove duplicates
        df_clean = self._remove_duplicates(df_clean, ticker)

        # 3. Handle missing values
        df_clean = self._handle_missing_values(df_clean, ticker)

        # 4. Handle outliers
        df_clean = self._handle_outliers(df_clean, ticker)

        # 5. Validate data types
        df_clean = self._validate_data_types(df_clean, ticker)

        # 6. Sort by date (ensure chronological)
        df_clean = df_clean.sort_index()

        # Update report
        self.cleaning_report[ticker]['final_rows'] = len(df_clean)
        self.cleaning_report[ticker]['rows_removed'] = (
            self.cleaning_report[ticker]['original_rows'] -
            self.cleaning_report[ticker]['final_rows']
        )

        print(f"\n✓ Cleaning complete for {ticker}")
        print(f"  Original rows: {self.cleaning_report[ticker]['original_rows']}")
        print(f"  Final rows: {self.cleaning_report[ticker]['final_rows']}")
        print(f"  Rows removed: {self.cleaning_report[ticker]['rows_removed']}")

        return df_clean

    def _fix_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime index is properly formatted"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.isna()]  # remove invalid dates
        except Exception as e:
            print(f"  Warning: Invalid datetime index ({e})")
        return df

    def _remove_duplicates(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove duplicate rows"""
        before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        removed = before - len(df)

        if removed > 0:
            print(f"  Removed {removed} duplicate rows")
            self.cleaning_report[ticker]['issues_found'].append(f"Duplicates: {removed}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Handle missing values"""
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()

        if total_missing == 0:
            return df

        print(f"  Found {total_missing} missing values")

        # Forward-fill for price data
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Fill missing volume with 0
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)

        # Drop any remaining NaN rows
        rows_before = len(df)
        df = df.dropna()
        dropped = rows_before - len(df)

        if dropped > 0:
            print(f"  Dropped {dropped} rows after filling")
        self.cleaning_report[ticker]['issues_found'].append(
            f"Missing values: {int(total_missing)} handled"
        )

        return df

    def _handle_outliers(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Detect and handle outliers using z-score on returns"""
        outlier_total = 0
        price_cols = ['Open', 'High', 'Low', 'Close']

        for col in price_cols:
            if col not in df.columns:
                continue

            returns = df[col].pct_change()
            if returns.std() == 0 or returns.std() is np.nan:
                continue

            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outliers = z_scores > self.outlier_threshold
            num_outliers = outliers.sum()

            if num_outliers > 0:
                outlier_total += num_outliers
                print(f"  Capping {num_outliers} outliers in {col}")
                upper = returns.mean() + self.outlier_threshold * returns.std()
                lower = returns.mean() - self.outlier_threshold * returns.std()

                # Clip returns and reconstruct prices
                capped_returns = returns.clip(lower, upper)
                df[col] = df[col].shift(1) * (1 + capped_returns)
                df[col].fillna(method='bfill', inplace=True)

        if outlier_total > 0:
            self.cleaning_report[ticker]['issues_found'].append(f"Outliers capped: {outlier_total}")

        return df

    def _validate_data_types(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Ensure correct types and positive prices"""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove non-positive prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns and (df[col] <= 0).any():
                invalid = (df[col] <= 0).sum()
                print(f"  Removed {invalid} non-positive {col} entries")
                df = df[df[col] > 0]
                self.cleaning_report[ticker]['issues_found'].append(f"Non-positive {col}: {invalid}")

        df = df.dropna()
        return df

    def get_cleaning_report(self) -> Dict:
        """Return cleaning report"""
        return self.cleaning_report

    def save_cleaned_data(self, df: pd.DataFrame, ticker: str, output_dir: str):
        """Save cleaned dataset"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"cleaned_{ticker}.csv")
        df.to_csv(path)
        print(f"  Saved cleaned data → {path}")


def clean_all_stocks(input_dir: str, output_dir: str, stocks: list) -> Dict:
    """Batch clean all stock CSV files"""
    cleaner = DataCleaner()

    print("\n" + "=" * 60)
    print("CLEANING ALL STOCK DATA")
    print("=" * 60)

    for ticker in stocks:
        file_path = os.path.join(input_dir, f"{ticker}.csv")

        if not os.path.exists(file_path):
            print(f"✗ Skipping {ticker} — file not found.")
            continue

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df_clean = cleaner.clean_data(df, ticker)
            cleaner.save_cleaned_data(df_clean, ticker, output_dir)
        except Exception as e:
            print(f"✗ Error processing {ticker}: {e}")

    print("\n" + "=" * 60)
    print("✓ ALL STOCK DATA CLEANED")
    print("=" * 60)
    return cleaner.get_cleaning_report()


if __name__ == "__main__":
    try:
        import config
        input_dir = config.RAW_DATA_DIR
        output_dir = config.PROCESSED_DATA_DIR
        stocks = config.STOCKS
    except ImportError:
        input_dir = "data/raw"
        output_dir = "data/processed"
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC", "XOM", "JNJ", "GS"]

    report = clean_all_stocks(input_dir, output_dir, stocks)

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    for ticker, info in report.items():
        print(f"\n{ticker}:")
        print(f"  Original rows: {info.get('original_rows', 'N/A')}")
        print(f"  Final rows: {info.get('final_rows', 'N/A')}")
        print(f"  Issues found: {info.get('issues_found', [])}")
