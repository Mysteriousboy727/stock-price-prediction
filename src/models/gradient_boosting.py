"""
Gradient Boosting Models for Stock Price Prediction
Uses XGBoost with technical indicators and features
"""

import pandas as pd
import numpy as np
import sys
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import config

def prepare_features(df, target='Close'):
    """
    Prepare features for gradient boosting
    """
    # Select relevant features
    feature_cols = [
        'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'ATR', 'OBV',
        'Price_Change', 'Price_Change_Pct',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_7',
        'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_5', 'Volume_Lag_7',
        'Close_Rolling_Mean_7', 'Close_Rolling_Std_7', 'Volume_Rolling_Mean_7'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df[target]
    
    return X, y, available_features

def train_xgboost_model(stock_symbol, test_size=0.2):
    """
    Train XGBoost model for a stock
    """
    print(f"\n{'='*60}")
    print(f"TRAINING XGBOOST MODEL - {stock_symbol}")
    print('='*60)
    
    # Load data
    file_path = os.path.join(config.PROCESSED_DATA_DIR, f'features_{stock_symbol}.csv')
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    print(f"  Dataset shape: {df.shape}")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"  Features: {len(feature_names)}")
    
    # Split data (time series split - no shuffle)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train model
    print("  Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Make predictions
    print("  Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"\n  Model Performance:")
    print(f"    Train RMSE: ${train_rmse:.2f}")
    print(f"    Test RMSE:  ${test_rmse:.2f}")
    print(f"    Test MAE:   ${test_mae:.2f}")
    print(f"    Test MAPE:  {test_mape:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n  Top 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")
    
    # Save results
    results = {
        'stock': stock_symbol,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': len(feature_names)
    }
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual': y_test.values,
        'Predicted': y_pred_test
    })
    
    pred_dir = os.path.join(config.PROJECT_ROOT, 'results', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    pred_path = os.path.join(pred_dir, f'xgboost_predictions_{stock_symbol}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"  ✓ Predictions saved → {pred_path}")
    
    # Save feature importance
    importance_path = os.path.join(pred_dir, f'feature_importance_{stock_symbol}.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved → {importance_path}")
    
    # Plot results
    plot_xgboost_results(stock_symbol, y_train, y_test, y_pred_train, y_pred_test)
    plot_feature_importance(stock_symbol, feature_importance)
    
    return results, model

def plot_xgboost_results(stock_symbol, y_train, y_test, y_pred_train, y_pred_test):
    """
    Plot XGBoost predictions vs actual values
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Training predictions
    ax1 = axes[0]
    ax1.plot(y_train.index, y_train.values, label='Actual', linewidth=2, alpha=0.7)
    ax1.plot(y_train.index, y_pred_train, label='Predicted', linewidth=2, alpha=0.7)
    ax1.set_title(f'{stock_symbol} - Training Data Predictions', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test predictions
    ax2 = axes[1]
    ax2.plot(y_test.index, y_test.values, label='Actual', linewidth=2, color='green')
    ax2.plot(y_test.index, y_pred_test, label='Predicted', linewidth=2, color='red', linestyle='--')
    ax2.set_title(f'{stock_symbol} - Test Data Predictions', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'models')
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_path = os.path.join(plot_dir, f'xgboost_predictions_{stock_symbol}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved → {plot_path}")

def plot_feature_importance(stock_symbol, feature_importance):
    """
    Plot feature importance
    """
    plt.figure(figsize=(12, 8))
    
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'{stock_symbol} - Top 15 Feature Importances', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'features')
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_path = os.path.join(plot_dir, f'feature_importance_{stock_symbol}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance plot saved → {plot_path}")

def run_xgboost_for_all_stocks():
    """
    Train XGBoost models for all stocks
    """
    print("\n" + "="*60)
    print("XGBOOST MODEL TRAINING - ALL STOCKS")
    print("="*60)
    
    results_list = []
    
    for stock in config.STOCKS:
        try:
            results, model = train_xgboost_model(stock)
            results_list.append(results)
            print(f"  ✓ Successfully trained XGBoost for {stock}")
        except Exception as e:
            print(f"  ✗ Error training {stock}: {e}")
    
    # Save summary
    results_df = pd.DataFrame(results_list)
    
    metrics_dir = os.path.join(config.PROJECT_ROOT, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(metrics_dir, 'xgboost_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    
    print("\n" + "="*60)
    print("✓ XGBOOST TRAINING COMPLETE")
    print("="*60)
    print(f"\nMetrics saved to: {metrics_path}")
    print("\nSummary:")
    print(results_df[['stock', 'test_rmse', 'test_mae', 'test_mape']].to_string(index=False))

if __name__ == "__main__":
    run_xgboost_for_all_stocks()