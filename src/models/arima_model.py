"""
ARIMA Model for Stock Price Prediction
Time Series Forecasting using ARIMA
"""

import pandas as pd
import numpy as np
import sys
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import config

def find_best_arima_params(data, max_p=5, max_d=2, max_q=5):
    """
    Grid search to find best ARIMA parameters (p, d, q)
    Based on AIC (Akaike Information Criterion)
    """
    best_aic = np.inf
    best_params = None
    
    print("  Searching for best ARIMA parameters...")
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    print(f"  ✓ Best parameters: {best_params} with AIC={best_aic:.2f}")
    return best_params

def train_arima_model(stock_symbol, test_size=0.2):
    """
    Train ARIMA model for a stock
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ARIMA MODEL - {stock_symbol}")
    print('='*60)
    
    # Load data
    file_path = os.path.join(config.PROCESSED_DATA_DIR, f'features_{stock_symbol}.csv')
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # Use only Close price for ARIMA
    data = df['Close']
    
    # Split data
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    print(f"  Training data: {len(train_data)} samples")
    print(f"  Test data: {len(test_data)} samples")
    
    # Find best parameters
    best_params = find_best_arima_params(train_data)
    
    # Train model with best parameters
    print(f"  Training ARIMA{best_params}...")
    model = ARIMA(train_data, order=best_params)
    fitted_model = model.fit()
    
    # Make predictions
    print("  Making predictions...")
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mae = mean_absolute_error(test_data, forecast)
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
    
    print(f"\n  Model Performance:")
    print(f"    RMSE: ${rmse:.2f}")
    print(f"    MAE:  ${mae:.2f}")
    print(f"    MAPE: {mape:.2f}%")
    
    # Save results
    results = {
        'stock': stock_symbol,
        'params': best_params,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'train_size': len(train_data),
        'test_size': len(test_data)
    }
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data.values,
        'Predicted': forecast.values
    })
    
    pred_dir = os.path.join(config.PROJECT_ROOT, 'results', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    pred_path = os.path.join(pred_dir, f'arima_predictions_{stock_symbol}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"  ✓ Predictions saved → {pred_path}")
    
    # Plot results
    plot_arima_results(stock_symbol, train_data, test_data, forecast)
    
    return results, fitted_model

def plot_arima_results(stock_symbol, train_data, test_data, forecast):
    """
    Plot ARIMA predictions vs actual values
    """
    plt.figure(figsize=(16, 8))
    
    # Plot training data
    plt.plot(train_data.index, train_data.values, label='Training Data', linewidth=2)
    
    # Plot test data
    plt.plot(test_data.index, test_data.values, label='Actual Test Data', 
             linewidth=2, color='green')
    
    # Plot predictions
    plt.plot(test_data.index, forecast.values, label='ARIMA Forecast', 
             linewidth=2, color='red', linestyle='--')
    
    plt.title(f'{stock_symbol} - ARIMA Model Predictions', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'models')
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_path = os.path.join(plot_dir, f'arima_forecast_{stock_symbol}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved → {plot_path}")

def run_arima_for_all_stocks():
    """
    Train ARIMA models for all stocks
    """
    print("\n" + "="*60)
    print("ARIMA MODEL TRAINING - ALL STOCKS")
    print("="*60)
    
    results_list = []
    
    for stock in config.STOCKS:
        try:
            results, model = train_arima_model(stock)
            results_list.append(results)
            print(f"  ✓ Successfully trained ARIMA for {stock}")
        except Exception as e:
            print(f"  ✗ Error training {stock}: {e}")
    
    # Save summary
    results_df = pd.DataFrame(results_list)
    
    metrics_dir = os.path.join(config.PROJECT_ROOT, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(metrics_dir, 'arima_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    
    print("\n" + "="*60)
    print("✓ ARIMA TRAINING COMPLETE")
    print("="*60)
    print(f"\nMetrics saved to: {metrics_path}")
    print("\nSummary:")
    print(results_df[['stock', 'rmse', 'mae', 'mape']].to_string(index=False))

if __name__ == "__main__":
    run_arima_for_all_stocks()