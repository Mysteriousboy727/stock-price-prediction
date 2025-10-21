"""
Visualization Module - EDA Plots
Creates all visualizations for exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import config

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_price_trends(stocks_data, output_dir):
    """Plot closing price trends for all stocks"""
    ensure_dir(output_dir)
    
    plt.figure(figsize=(16, 10))
    
    for stock_symbol, df in stocks_data.items():
        plt.plot(df.index, df['Close'], label=stock_symbol, linewidth=2, alpha=0.8)
    
    plt.title('Stock Price Trends (2020-2025)', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price ($)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'price_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_volume_analysis(stocks_data, output_dir):
    """Plot volume analysis for all stocks"""
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for idx, (stock_symbol, df) in enumerate(stocks_data.items()):
        ax = axes[idx]
        ax.bar(df.index, df['Volume'], alpha=0.7, color='steelblue')
        ax.set_title(f'{stock_symbol} - Trading Volume', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Volume', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'volume_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_returns_distribution(stocks_data, output_dir):
    """Plot daily returns distribution"""
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for idx, (stock_symbol, df) in enumerate(stocks_data.items()):
        returns = df['Close'].pct_change().dropna() * 100
        
        ax = axes[idx]
        ax.hist(returns, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax.set_title(f'{stock_symbol} - Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Return (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'returns_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_correlation_heatmap(stocks_data, output_dir):
    """Plot correlation heatmap of closing prices"""
    ensure_dir(output_dir)
    
    # Create dataframe with all closing prices
    close_prices = pd.DataFrame()
    for stock_symbol, df in stocks_data.items():
        close_prices[stock_symbol] = df['Close']
    
    # Calculate correlation
    correlation = close_prices.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Stock Price Correlation Heatmap', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_technical_indicators(stock_symbol, df, output_dir):
    """Plot technical indicators for a single stock"""
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    
    # Price with Moving Averages
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.2, label='Bollinger Bands')
    ax1.set_title(f'{stock_symbol} - Price & Moving Averages', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.set_title(f'{stock_symbol} - RSI', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3 = axes[2]
    ax3.plot(df.index, df['MACD'], label='MACD', linewidth=2)
    ax3.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=2)
    ax3.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3)
    ax3.set_title(f'{stock_symbol} - MACD', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Volume
    ax4 = axes[3]
    ax4.bar(df.index, df['Volume'], alpha=0.7, color='steelblue')
    ax4.set_title(f'{stock_symbol} - Volume', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volume')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'technical_indicators_{stock_symbol}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def run_full_eda():
    """Run complete EDA and generate all visualizations"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # Create output directories
    eda_dir = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'eda')
    features_dir = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'features')
    
    # Load all processed data
    print("\nLoading processed stock data...")
    stocks_data = {}
    for stock in config.STOCKS:
        try:
            file_path = os.path.join(config.PROCESSED_DATA_DIR, f'features_{stock}.csv')
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            stocks_data[stock] = df
            print(f"  ✓ Loaded {stock}: {df.shape}")
        except Exception as e:
            print(f"  ✗ Error loading {stock}: {e}")
    
    # Generate visualizations
    print("\nGenerating EDA visualizations...")
    plot_price_trends(stocks_data, eda_dir)
    plot_volume_analysis(stocks_data, eda_dir)
    plot_returns_distribution(stocks_data, eda_dir)
    plot_correlation_heatmap(stocks_data, eda_dir)
    
    print("\nGenerating technical indicator plots...")
    for stock_symbol, df in stocks_data.items():
        plot_technical_indicators(stock_symbol, df, features_dir)
    
    print("\n" + "="*60)
    print("✓ EDA COMPLETE - All visualizations saved!")
    print("="*60)
    print(f"\nCheck your results in:")
    print(f"  - {eda_dir}")
    print(f"  - {features_dir}")

if __name__ == "__main__":
    run_full_eda()