"""
Master Pipeline Script
Runs the entire stock prediction pipeline from start to finish
"""

import os
import sys
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_script(script_path, description):
    """Run a Python script and report results"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Error: {e}")
        return False

def main():
    """Run complete pipeline"""
    start_time = datetime.now()
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "STOCK PRICE PREDICTION PIPELINE" + " "*22 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = [
        ("scripts/download_all_data.py", "1. Download Stock Data"),
        ("src/data/data_cleaner.py", "2. Clean Data"),
        ("src/features/technical_indicators.py", "3. Feature Engineering"),
        ("src/visualization/plots.py", "4. Exploratory Data Analysis"),
        ("src/models/arima_model.py", "5. Train ARIMA Models"),
        ("src/models/gradient_boosting.py", "6. Train XGBoost Models"),
        ("src/models/model_evaluator.py", "7. Model Comparison & Report")
    ]
    
    results = []
    
    for script_path, description in steps:
        success = run_script(script_path, description)
        results.append((description, success))
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for step, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {step}")
    
    successful = sum(1 for _, s in results if s)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"Completed: {successful}/{total} steps")
    print(f"Duration: {duration}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if successful == total:
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
        print("\nYour results are ready:")
        print("  • Visualizations: results/figures/")
        print("  • Predictions: results/predictions/")
        print("  • Metrics: results/metrics/")
        print("  • Report: reports/executive_summary.md")
    else:
        print("\n⚠️ Some steps failed. Please check the errors above.")

if __name__ == "__main__":
    main()