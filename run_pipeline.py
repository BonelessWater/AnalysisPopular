#!/usr/bin/env python3
# run_pipeline.py
# Script to execute the entire bank customer profitability analysis pipeline

import os
import subprocess
import time
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_step(step_name, command):
    """Run a pipeline step and log its execution"""
    logging.info(f"Starting step: {step_name}")
    start_time = time.time()
    
    try:
        subprocess.run(command, check=True, shell=True)
        end_time = time.time()
        logging.info(f"Completed step: {step_name} in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in step: {step_name}")
        logging.error(f"Command failed with error code: {e.returncode}")
        logging.error(f"Error details: {e}")
        return False

def create_directory_structure():
    """Create the directory structure for the project"""
    logging.info("Creating directory structure")
    
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
    
    logging.info("Directory structure created")

def main():
    """Run the entire pipeline"""
    logging.info("Starting bank customer profitability analysis pipeline")
    
    # Create directory structure
    create_directory_structure()
    
    # Define pipeline steps
    pipeline_steps = [
        ("Generate synthetic data", "python generate_synthetic_data.py"),
        ("Run profitability analysis", "python bank_profitability_pipeline.py"),
        ("Visualize results", "python visualize_results.py")
    ]
    
    # Execute pipeline steps
    success = True
    for step_name, command in pipeline_steps:
        step_success = run_step(step_name, command)
        if not step_success:
            logging.error(f"Pipeline failed at step: {step_name}")
            success = False
            break
    
    # Pipeline completion
    if success:
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
    else:
        logging.error("Pipeline failed")
        print("\n" + "="*50)
        print("PIPELINE EXECUTION FAILED")
        print("="*50)
        print("\nCheck the logs for details: pipeline_execution.log")

if __name__ == "__main__":
    main()