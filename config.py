"""
Configuration file for Stock Price Prediction Project
"""

import os
from datetime import datetime

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Stock Symbols
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'BAC', 'XOM', 'JNJ', 'GS']

# Date Configuration
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
TRAIN_TEST_SPLIT = 0.8

# Random State
RANDOM_STATE = 42