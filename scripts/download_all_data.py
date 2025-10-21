"""
Stock Data Download Script
Downloads historical OHLC data for multiple stocks using yfinance
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import json

# Configuration
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'BAC', 'XOM', 'JNJ', 'GS']
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = 'data/raw'

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame : Downloaded stock data
    """
    try:
        print(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return None
        
        # Add ticker column
        df['Ticker'] = ticker
        
        print(f"✓ Successfully downloaded {len(df)} rows for {ticker}")
        return df
    
    except Exception as e:
        print(f"✗ Error downloading {ticker}: {str(e)}")
        return None

def save_to_csv(df, ticker):
    """Save dataframe to CSV file"""
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    df.to_csv(filepath)
    print(f"  Saved to: {filepath}")

def save_to_json(df, ticker):
    """Save dataframe to JSON file"""
    filepath = os.path.join(DATA_DIR, f'{ticker}.json')
    df.reset_index().to_json(filepath, orient='records', date_format='iso')
    print(f"  Saved to: {filepath}")

def create_metadata(stocks_info):
    """Create metadata file with download information"""
    metadata = {
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': START_DATE,
        'end_date': END_DATE,
        'stocks': stocks_info
    }
    
    filepath = os.path.join(DATA_DIR, 'data_info.json')
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\n✓ Metadata saved to: {filepath}")

def main():
    """Main execution function"""
    print("="*60)
    print("STOCK DATA DOWNLOAD")
    print("="*60)
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("="*60 + "\n")
    
    stocks_info = {}
    all_data = []
    
    for ticker in STOCKS:
        df = download_stock_data(ticker, START_DATE, END_DATE)
        
        if df is not None:
            # Save in both CSV and JSON formats
            save_to_csv(df, ticker)
            save_to_json(df, ticker)
            
            # Store metadata
            stocks_info[ticker] = {
                'rows': len(df),
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'columns': list(df.columns)
            }
            
            all_data.append(df)
        
        print()
    
    # Create combined dataset
    if all_data:
        print("Creating combined dataset...")
        combined_df = pd.concat(all_data)
        combined_path = os.path.join(DATA_DIR, 'combined_stocks.csv')
        combined_df.to_csv(combined_path)
        print(f"✓ Combined dataset saved to: {combined_path}")
        print(f"  Total rows: {len(combined_df)}")
    
    # Save metadata
    create_metadata(stocks_info)
    
    print("\n" + "="*60)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Successfully downloaded: {len(stocks_info)}/{len(STOCKS)} stocks")
    print(f"  Data saved in: {DATA_DIR}/")
    print(f"  Formats: CSV and JSON")

if __name__ == "__main__":
    main()