"""
Technical Indicators Module
Calculates various technical indicators for stock price analysis
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import config

def calculate_sma(data, window=20):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, span=20):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: MACD line, Signal line, and Histogram
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    Returns: Upper band, Middle band (SMA), Lower band
    """
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range (ATR) - Volatility indicator"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_obv(close, volume):
    """Calculate On-Balance Volume (OBV)"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def add_all_indicators(df):
    """
    Add all technical indicators to the dataframe
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_10'] = calculate_sma(df['Close'], 10)
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    # Exponential Moving Averages
    df['EMA_12'] = calculate_ema(df['Close'], 12)
    df['EMA_26'] = calculate_ema(df['Close'], 26)
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # ATR (Volatility)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    
    # OBV (Volume)
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    
    # Price Changes
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    
    # Lag Features (for prediction)
    for lag in [1, 2, 3, 5, 7]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Rolling Statistics
    df['Close_Rolling_Mean_7'] = df['Close'].rolling(7).mean()
    df['Close_Rolling_Std_7'] = df['Close'].rolling(7).std()
    df['Volume_Rolling_Mean_7'] = df['Volume'].rolling(7).mean()
    
    return df

def process_stock_features(stock_symbol):
    """
    Load cleaned data and add all technical indicators
    """
    print(f"\n{'='*60}")
    print(f"Processing features for {stock_symbol}")
    print('='*60)
    
    # Load cleaned data
    input_path = os.path.join(config.PROCESSED_DATA_DIR, f'cleaned_{stock_symbol}.csv')
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    print(f"  Original shape: {df.shape}")
    
    # Add all indicators
    df_features = add_all_indicators(df)
    
    # Drop rows with NaN values (from indicators requiring history)
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    rows_dropped = initial_rows - len(df_features)
    
    print(f"  After adding features: {df_features.shape}")
    print(f"  Rows dropped (NaN): {rows_dropped}")
    print(f"  Total features: {len(df_features.columns)}")
    
    # Save
    output_path = os.path.join(config.PROCESSED_DATA_DIR, f'features_{stock_symbol}.csv')
    df_features.to_csv(output_path)
    print(f"  ✓ Saved → {output_path}")
    
    return df_features

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FEATURE ENGINEERING - TECHNICAL INDICATORS")
    print("="*60)
    
    for stock in config.STOCKS:
        try:
            df = process_stock_features(stock)
            print(f"  ✓ Successfully processed {stock}")
        except Exception as e:
            print(f"  ✗ Error processing {stock}: {e}")
    
    print("\n" + "="*60)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("="*60)