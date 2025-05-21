#!/usr/bin/env python3
# error_handling_wrapper.py
# A wrapper script with additional error checking for the bank profitability pipeline

import os
import sys
import subprocess
import time
import logging
import traceback
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_wrapper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define file paths
DATA_DIR = 'data'
RAW_DATA_PATH = f'{DATA_DIR}/raw/customer_data.csv'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

def check_prerequisites():
    """Check that all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'joblib', 'faker'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing required packages: {', '.join(missing_packages)}")
        logging.info("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logging.info("Successfully installed missing packages")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install packages: {e}")
            return False
    
    return True

def verify_data_file():
    """Verify that the data file exists and is valid"""
    if not os.path.exists(RAW_DATA_PATH):
        logging.warning(f"Data file not found at {RAW_DATA_PATH}")
        return False
    
    try:
        # Try reading the CSV to ensure it's valid
        df = pd.read_csv(RAW_DATA_PATH)
        row_count = len(df)
        col_count = len(df.columns)
        logging.info(f"Data file verified: {row_count} rows, {col_count} columns")
        
        # Check for required columns
        required_columns = ['net_profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Data file is missing required columns: {', '.join(missing_columns)}")
            return False
        
        # Log all columns for debugging
        logging.info(f"Available columns: {df.columns.tolist()}")
        
        # Check for at least a minimum number of rows
        if row_count < 50:
            logging.warning(f"Data file contains only {row_count} rows, which may be insufficient for modeling")
        
        # Verify no missing values in critical columns
        critical_columns = ['net_profit'] 
        for col in critical_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                missing_percent = (missing_count / row_count) * 100
                logging.warning(f"Column '{col}' has {missing_count} missing values ({missing_percent:.2f}%)")
                if missing_percent > 20:
                    logging.error(f"Critical column '{col}' has too many missing values")
                    return False
            
        return True
    except Exception as e:
        logging.error(f"Error validating data file: {e}")
        logging.error(traceback.format_exc())
        return False

def run_pipeline_step(script_name, max_retries=2):
    """Run a pipeline script with retry logic"""
    logging.info(f"Running {script_name}")
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt} for {script_name}")
                
            result = subprocess.run(
                [sys.executable, script_name], 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # Log the output
            for line in result.stdout.splitlines():
                logging.info(f"{script_name} output: {line}")
                
            logging.info(f"Successfully completed {script_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {script_name} (attempt {attempt+1}/{max_retries+1})")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            
            if attempt == max_retries:
                logging.error(f"Failed to run {script_name} after {max_retries+1} attempts")
                return False
            
            # Wait before retrying
            time.sleep(2)

def create_directory_structure():
    """Create the directory structure for the project"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'results',
        'results/plots',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def fix_common_data_issues():
    """
    Attempt to fix common data issues before running the main pipeline
    """
    if not os.path.exists(RAW_DATA_PATH):
        logging.warning("Cannot fix data issues - data file doesn't exist")
        return False
    
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        fixed = False
        
        # Check for customer_id column and add it if missing
        if 'customer_id' not in df.columns:
            logging.warning("'customer_id' column missing - adding it")
            df['customer_id'] = [f'CUST{100000 + i}' for i in range(len(df))]
            fixed = True
        
        # Check for profit_segment and add it if missing
        if 'profit_segment' not in df.columns and 'net_profit' in df.columns:
            logging.warning("'profit_segment' column missing - adding it")
            bins = [-float('inf'), 0, 500, 1000, float('inf')]
            labels = ['loss', 'low_profit', 'medium_profit', 'high_profit']
            df['profit_segment'] = pd.cut(df['net_profit'], bins=bins, labels=labels)
            fixed = True
        
        # Check for is_profitable and add it if missing
        if 'is_profitable' not in df.columns and 'net_profit' in df.columns:
            logging.warning("'is_profitable' column missing - adding it")
            df['is_profitable'] = (df['net_profit'] > 0).astype(int)
            fixed = True
        
        # If we made any changes, save the file
        if fixed:
            backup_path = f"{RAW_DATA_PATH}.bak"
            if os.path.exists(RAW_DATA_PATH):
                logging.info(f"Backing up original data file to {backup_path}")
                import shutil
                shutil.copy2(RAW_DATA_PATH, backup_path)
            
            df.to_csv(RAW_DATA_PATH, index=False)
            logging.info("Fixed data issues and saved updated file")
            return True
        else:
            logging.info("No data issues to fix")
            return True
    
    except Exception as e:
        logging.error(f"Error fixing data issues: {e}")
        logging.error(traceback.format_exc())
        return False

def main():
    """Main function to run the pipeline with error handling"""
    try:
        logging.info("Starting bank profitability analysis pipeline with error handling")
        
        # Create directory structure
        create_directory_structure()
        
        # Check prerequisites
        if not check_prerequisites():
            logging.error("Failed to verify prerequisites. Exiting.")
            return False
        
        # Run data generation if needed
        if not os.path.exists(RAW_DATA_PATH):
            logging.info("Data file not found, running data generation")
            if not run_pipeline_step("generate_synthetic_data.py"):
                logging.error("Data generation failed. Exiting.")
                return False
        else:
            logging.info(f"Data file already exists at {RAW_DATA_PATH}")
        
        # Verify data file
        if not verify_data_file():
            logging.warning("Data validation failed. Attempting to fix common issues.")
            if not fix_common_data_issues():
                logging.error("Failed to fix data issues. Exiting.")
                return False
            
            # Verify again after fixing
            if not verify_data_file():
                logging.error("Data validation failed even after fixes. Exiting.")
                return False
        
        # Run main pipeline
        if not run_pipeline_step("bank_profitability_pipeline.py"):
            logging.error("Main pipeline failed. Exiting.")
            return False
        
        # Run visualization
        if not run_pipeline_step("visualize_results.py"):
            logging.warning("Visualization failed, but continuing.")
            # We continue even if visualization fails
        
        logging.info("Pipeline completed successfully")
        print("\n" + "="*50)
        print("BANK CUSTOMER PROFITABILITY ANALYSIS COMPLETE")
        print("="*50)
        print("\nResults available in the following locations:")
        print("- Raw data: data/raw/customer_data.csv")
        print("- Processed data: data/processed/processed_data.csv")
        print("- Trained models: models/")
        print("- Analysis results: results/")
        print("- Visualizations: results/plots/")
        print("- Executive summary: results/executive_summary.txt")
        print("\nTo view the executive summary, run:")
        print("cat results/executive_summary.txt")
        
        return True
        
    except Exception as e:
        logging.error("Unhandled exception in pipeline wrapper")
        logging.error(traceback.format_exc())
        print("\n" + "="*50)
        print("PIPELINE EXECUTION FAILED")
        print("="*50)
        print("\nCheck the logs for details: pipeline_wrapper.log")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)