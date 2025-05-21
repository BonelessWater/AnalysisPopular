# bank_profitability_pipeline.py
# Main pipeline script for bank customer profitability analysis

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report, roc_auc_score, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import datetime

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f"{log_dir}/pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define file paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
RAW_DATA_PATH = f'{DATA_DIR}/raw/customer_data.csv'
PROCESSED_DATA_PATH = f'{DATA_DIR}/processed/processed_data.csv'
FEATURE_IMPORTANCE_PATH = f'{RESULTS_DIR}/feature_importance.png'

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, f'{DATA_DIR}/raw', f'{DATA_DIR}/processed']:
    os.makedirs(directory, exist_ok=True)

def load_data():
    """Load raw data from CSV file"""
    logging.info("Loading raw data")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the raw data"""
    logging.info("Starting data preprocessing")
    
    # Check if customer_id exists, if not create it
    if 'customer_id' not in df.columns:
        logging.warning("'customer_id' column not found in data, creating it")
        df['customer_id'] = [f'CUST{100000 + i}' for i in range(len(df))]
    
    # Log all columns for debugging
    logging.info(f"Available columns: {df.columns.tolist()}")
    
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variables from features
    for col in ['net_profit', 'profit_segment', 'is_profitable', 'customer_id']:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create regression and classification targets
    if 'net_profit' not in df.columns:
        logging.error("Target variable 'net_profit' not found in dataset")
        raise ValueError("Target variable not found")
    
    # Create profit segments if not already present
    if 'profit_segment' not in df.columns:
        logging.info("Creating profit segments from net_profit")
        bins = [-float('inf'), 0, 500, 1000, float('inf')]
        labels = ['loss', 'low_profit', 'medium_profit', 'high_profit']
        df['profit_segment'] = pd.cut(df['net_profit'], bins=bins, labels=labels)
    
    # Create binary profitable flag
    df['is_profitable'] = (df['net_profit'] > 0).astype(int)
    
    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Preprocessed data saved to {PROCESSED_DATA_PATH}")
    
    return df, preprocessor

def train_models(df, preprocessor):
    """Train regression and classification models"""
    logging.info("Starting model training")
    
    # Prepare data - drop columns if they exist, ignore errors if they don't
    columns_to_drop = ['customer_id', 'net_profit', 'profit_segment', 'is_profitable']
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y_reg = df['net_profit']
    y_class = df['profit_segment']
    y_binary = df['is_profitable']
    
    logging.info(f"Feature columns: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test, y_binary_train, y_binary_test = train_test_split(
        X, y_reg, y_class, y_binary, test_size=0.2, random_state=42
    )
    
    # Train regression model
    logging.info("Training regression model")
    reg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    reg_model.fit(X_train, y_reg_train)
    
    # Train classification model
    logging.info("Training classification model")
    class_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    class_model.fit(X_train, y_class_train)
    
    # Train binary classification model
    logging.info("Training binary classification model")
    binary_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    binary_model.fit(X_train, y_binary_train)
    
    # Save models
    joblib.dump(reg_model, f'{MODEL_DIR}/regression_model.pkl')
    joblib.dump(class_model, f'{MODEL_DIR}/classification_model.pkl')
    joblib.dump(binary_model, f'{MODEL_DIR}/binary_model.pkl')
    logging.info("Models saved to disk")
    
    # Evaluate models
    evaluate_models(
        reg_model, class_model, binary_model,
        X_test, y_reg_test, y_class_test, y_binary_test
    )
    
    return reg_model, class_model, binary_model

def evaluate_models(reg_model, class_model, binary_model, X_test, y_reg_test, y_class_test, y_binary_test):
    """Evaluate trained models"""
    logging.info("Evaluating models")
    
    # Regression evaluation
    y_reg_pred = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)
    logging.info(f"Regression model - RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    # Classification evaluation
    y_class_pred = class_model.predict(X_test)
    class_report = classification_report(y_class_test, y_class_pred)
    logging.info(f"Classification model report:\n{class_report}")
    
    # Binary classification evaluation
    y_binary_pred = binary_model.predict(X_test)
    binary_prob = binary_model.predict_proba(X_test)[:, 1]
    binary_report = classification_report(y_binary_test, y_binary_pred)
    auc = roc_auc_score(y_binary_test, binary_prob)
    logging.info(f"Binary classification model report:\n{binary_report}")
    logging.info(f"Binary classification AUC: {auc:.2f}")
    
    # Create results summary
    with open(f'{RESULTS_DIR}/model_evaluation.txt', 'w') as f:
        f.write(f"Regression Model (Net Profit)\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R²: {r2:.2f}\n\n")
        
        f.write(f"Classification Model (Profit Segments)\n")
        f.write(f"{class_report}\n\n")
        
        f.write(f"Binary Classification Model (Profitable vs Unprofitable)\n")
        f.write(f"{binary_report}\n")
        f.write(f"AUC: {auc:.2f}\n")

def analyze_feature_importance(reg_model, X):
    """Analyze and visualize feature importance"""
    logging.info("Analyzing feature importance")
    
    # For gradient boosting regressor
    try:
        # Get preprocessor from pipeline
        preprocessor = reg_model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        cat_features = preprocessor.transformers_[1][2]  # categorical features
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
        
        num_features = preprocessor.transformers_[0][2]  # numerical features
        
        # Combine all feature names
        all_feature_names = np.concatenate([num_features, cat_feature_names])
        
        # Get feature importance
        gbr = reg_model.named_steps['regressor']
        feature_importance = gbr.feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Visualize top 15 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH)
        logging.info(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
        
        # Save feature importance to CSV
        importance_df.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)
        
    except Exception as e:
        logging.error(f"Error analyzing feature importance: {e}")
        logging.error(f"Exception details: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def main():
    """Main pipeline execution function"""
    try:
        logging.info("Starting bank profitability analysis pipeline")
        
        # Load data
        df = load_data()
        
        # Preprocess data
        df, preprocessor = preprocess_data(df)
        
        # Train models
        reg_model, class_model, binary_model = train_models(df, preprocessor)
        
        # Analyze feature importance
        # Use the same method for creating X as in train_models
        columns_to_drop = ['customer_id', 'net_profit', 'profit_segment', 'is_profitable']
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        analyze_feature_importance(reg_model, X)
        
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()