# visualize_results.py
# Script to visualize bank customer profitability analysis results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

# Define file paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
PROCESSED_DATA_PATH = f'{DATA_DIR}/processed/processed_data.csv'
PLOTS_DIR = f'{RESULTS_DIR}/plots'

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data_and_models():
    """Load processed data and trained models"""
    print("Loading data and models...")
    
    try:
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Load models
        reg_model = joblib.load(f'{MODEL_DIR}/regression_model.pkl')
        class_model = joblib.load(f'{MODEL_DIR}/classification_model.pkl')
        binary_model = joblib.load(f'{MODEL_DIR}/binary_model.pkl')
        
        return df, reg_model, class_model, binary_model
    
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}")
        raise
    except Exception as e:
        print(f"Error loading data or models: {e}")
        raise

def visualize_profit_distribution(df):
    """Visualize the distribution of customer profitability"""
    print("Creating profit distribution visualizations...")
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create figure for profit distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['net_profit'], kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.title('Distribution of Customer Net Profit')
    plt.xlabel('Net Profit ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/profit_distribution.png')
    
    # Create pie chart for profit segments
    plt.figure(figsize=(10, 6))
    segment_counts = df['profit_segment'].value_counts()
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 4))
    plt.axis('equal')
    plt.title('Customer Profit Segments')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/profit_segments_pie.png')
    
    # Mean profit by segment
    plt.figure(figsize=(10, 6))
    segment_avg = df.groupby('profit_segment')['net_profit'].mean().sort_values()
    ax = sns.barplot(x=segment_avg.index, y=segment_avg.values)
    plt.title('Average Net Profit by Segment')
    plt.xlabel('Profit Segment')
    plt.ylabel('Average Net Profit ($)')
    # Add value labels
    for i, v in enumerate(segment_avg.values):
        ax.text(i, v + 50, f"${v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/avg_profit_by_segment.png')

def visualize_feature_relationships(df):
    """Visualize relationships between features and profitability"""
    print("Creating feature relationship visualizations...")
    
    # Key features to analyze
    key_numeric_features = [
        'age', 'annual_income', 'years_with_bank', 'credit_score', 
        'digital_ratio', 'num_active_accounts', 'avg_balance'
    ]
    
    key_categorical_features = [
        'sex', 'education_level', 'employment_status', 'residential_status'
    ]
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/correlation_heatmap.png')
    
    # Top correlations with net profit
    profit_corr = corr['net_profit'].drop('net_profit').sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=profit_corr.values, y=profit_corr.index)
    plt.title('Top Features Correlated with Net Profit')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/profit_correlations.png')
    
    # Scatter plots for key numeric features vs profit
    for feature in key_numeric_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature, y='net_profit', data=df, alpha=0.6, hue='profit_segment', palette='viridis')
        plt.title(f'{feature.replace("_", " ").title()} vs. Net Profit')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/scatter_{feature}_vs_profit.png')
    
    # Box plots for key categorical features
    for feature in key_categorical_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=feature, y='net_profit', data=df)
        plt.title(f'Net Profit by {feature.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/boxplot_{feature}_vs_profit.png')
    
    # Create a pair plot of key features
    plt.figure(figsize=(16, 14))
    key_features = ['age', 'annual_income', 'credit_score', 'transaction_frequency', 'net_profit']
    pair_data = df[key_features].copy()
    pair_data['profit_group'] = df['profit_segment']
    sns.pairplot(pair_data, hue='profit_group', palette='viridis')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/feature_pairplot.png')

def visualize_model_performance(df, reg_model, class_model, binary_model):
    """Visualize model performance metrics"""
    print("Creating model performance visualizations...")
    
    # Prepare data
    X = df.drop(['customer_id', 'net_profit', 'profit_segment', 'is_profitable'], axis=1, errors='ignore')
    y_reg = df['net_profit']
    y_class = df['profit_segment']
    y_binary = df['is_profitable']
    
    # Generate predictions
    y_reg_pred = reg_model.predict(X)
    y_class_pred = class_model.predict(X)
    y_binary_pred = binary_model.predict(X)
    y_binary_prob = binary_model.predict_proba(X)[:, 1]
    
    # 1. Regression Model: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_reg, y_reg_pred, alpha=0.5)
    plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'k--', lw=2)
    plt.xlabel('Actual Net Profit')
    plt.ylabel('Predicted Net Profit')
    plt.title('Regression Model: Actual vs Predicted Net Profit')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/regression_actual_vs_predicted.png')
    
    # 2. Classification Model: Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_class, y_class_pred, labels=class_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_model.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix: Profit Segment Classification')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/classification_confusion_matrix.png')
    
    # 3. Binary Model: ROC Curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, thresholds = roc_curve(y_binary, y_binary_prob)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Profitable vs Unprofitable Classification')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/binary_roc_curve.png')
    
    # 4. Binary Model: Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_binary_prob, bins=50, kde=True)
    plt.axvline(0.5, color='red', linestyle='--')
    plt.xlabel('Predicted Probability of Being Profitable')
    plt.ylabel('Count')
    plt.title('Distribution of Profitability Probabilities')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/binary_probability_distribution.png')

def create_executive_dashboard(df):
    """Create an executive dashboard with key profitability insights"""
    print("Creating executive dashboard...")
    
    # Summary statistics
    total_customers = len(df)
    profitable_customers = sum(df['net_profit'] > 0)
    profitable_pct = profitable_customers / total_customers * 100
    total_profit = df['net_profit'].sum()
    avg_profit = df['net_profit'].mean()
    
    # Top segment stats
    top_segment = df[df['profit_segment'] == 'high_profit']
    top_segment_count = len(top_segment)
    top_segment_pct = top_segment_count / total_customers * 100
    top_segment_profit = top_segment['net_profit'].sum()
    top_segment_profit_pct = top_segment_profit / total_profit * 100
    
    # Create a dashboard with 6 subplots
    plt.figure(figsize=(15, 12))
    
    # 1. Profit Distribution
    plt.subplot(2, 3, 1)
    sns.histplot(df['net_profit'], kde=True, bins=20)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.title('Net Profit Distribution')
    
    # 2. Profit Segment Breakdown
    plt.subplot(2, 3, 2)
    segment_counts = df['profit_segment'].value_counts()
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 4))
    plt.axis('equal')
    plt.title('Customer Profit Segments')
    
    # 3. Top 5 Profit-Correlated Features
    plt.subplot(2, 3, 3)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    profit_corr = corr['net_profit'].drop('net_profit').sort_values(ascending=False)[:5]
    sns.barplot(x=profit_corr.values, y=profit_corr.index)
    plt.title('Top 5 Profit Drivers')
    plt.xlabel('Correlation')
    
    # 4. Age vs Profit
    plt.subplot(2, 3, 4)
    sns.scatterplot(x='age', y='net_profit', data=df, alpha=0.6, hue='profit_segment', palette='viridis', legend=False)
    plt.title('Age vs. Net Profit')
    
    # 5. Income vs Profit
    plt.subplot(2, 3, 5)
    sns.scatterplot(x='annual_income', y='net_profit', data=df, alpha=0.6, hue='profit_segment', palette='viridis', legend=False)
    plt.title('Income vs. Net Profit')
    
    # 6. Digital Ratio vs Profit
    plt.subplot(2, 3, 6)
    sns.scatterplot(x='digital_ratio', y='net_profit', data=df, alpha=0.6, hue='profit_segment', palette='viridis', legend=False)
    plt.title('Digital Ratio vs. Net Profit')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/executive_dashboard.png')
    
    # Create a text summary file
    with open(f'{RESULTS_DIR}/executive_summary.txt', 'w') as f:
        f.write("BANK CUSTOMER PROFITABILITY ANALYSIS\n")
        f.write("===================================\n\n")
        f.write(f"Total Customers Analyzed: {total_customers}\n")
        f.write(f"Profitable Customers: {profitable_customers} ({profitable_pct:.1f}%)\n")
        f.write(f"Total Net Profit: ${total_profit:,.2f}\n")
        f.write(f"Average Profit per Customer: ${avg_profit:.2f}\n\n")
        
        f.write("SEGMENT BREAKDOWN\n")
        f.write("-----------------\n")
        for segment, count in segment_counts.items():
            segment_avg = df[df['profit_segment'] == segment]['net_profit'].mean()
            segment_total = df[df['profit_segment'] == segment]['net_profit'].sum()
            segment_pct = count / total_customers * 100
            f.write(f"{segment.title()}: {count} customers ({segment_pct:.1f}%)\n")
            f.write(f"  Average profit: ${segment_avg:.2f}\n")
            f.write(f"  Total segment profit: ${segment_total:,.2f}\n\n")
        
        f.write("TOP PROFIT DRIVERS\n")
        f.write("------------------\n")
        for feature, corr_value in profit_corr.items():
            f.write(f"{feature.replace('_', ' ').title()}: {corr_value:.3f} correlation\n")

def main():
    """Main function for visualization"""
    print("Starting visualization process...")
    
    # Load data and models
    df, reg_model, class_model, binary_model = load_data_and_models()
    
    # Run visualizations
    visualize_profit_distribution(df)
    visualize_feature_relationships(df)
    visualize_model_performance(df, reg_model, class_model, binary_model)
    create_executive_dashboard(df)
    
    print(f"Visualizations complete. Results saved to {PLOTS_DIR}")
    print(f"Full results available in {RESULTS_DIR}")

if __name__ == "__main__":
    main()