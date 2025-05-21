# generate_synthetic_data.py
# Script to generate synthetic bank customer data for profitability analysis

import os
import pandas as pd
import numpy as np
from faker import Faker
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# Define file paths
DATA_DIR = 'data'
RAW_DATA_PATH = f'{DATA_DIR}/raw/customer_data.csv'

# Create directories if they don't exist
os.makedirs(f'{DATA_DIR}/raw', exist_ok=True)

def generate_customer_data(num_samples=1000):
    """Generate synthetic bank customer data"""
    
    customers = []
    
    for i in range(num_samples):
        customer_id = f'CUST{100000 + i}'
        
        # Demographics & Profile
        age = random.randint(18, 85)
        sex = random.choice(['Male', 'Female'])
        marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
        dependents = random.randint(0, 5)
        education_level = random.choice(['High School', 'College', 'Bachelor', 'Master', 'PhD'])
        employment_status = random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'])
        annual_income = max(0, np.random.normal(60000, 30000))
        residential_status = random.choice(['Owner', 'Renter', 'Other'])
        years_with_bank = random.randint(0, 30)
        
        # Account & Transaction Metrics
        num_active_accounts = random.randint(1, 5)
        avg_balance = max(0, np.random.normal(5000, 10000))
        total_deposits = max(0, np.random.normal(30000, 20000))
        transaction_frequency = random.randint(1, 100)
        digital_ratio = random.random()  # Proportion of digital vs. branch transactions
        credit_utilization = min(1, max(0, np.random.normal(0.3, 0.2)))
        overdraft_count = random.randint(0, 10)
        fee_revenue = overdraft_count * random.uniform(25, 35) + random.uniform(0, 200)
        
        # Credit & Risk Indicators
        credit_score = min(850, max(300, np.random.normal(700, 100)))
        open_credit_lines = random.randint(0, 10)
        delinquency_count = random.randint(0, 5)
        total_loan_amount = max(0, np.random.normal(50000, 100000))
        ltv_ratio = min(1, max(0, np.random.normal(0.6, 0.2))) if total_loan_amount > 0 else 0
        ever_defaulted = 1 if random.random() < 0.05 else 0
        
        # Service & Engagement Metrics
        branch_visits = random.randint(0, 20)
        call_center_calls = random.randint(0, 15)
        complaints = random.randint(0, 3)
        digital_logins = random.randint(0, 200)
        
        # Derived & Behavioral Features
        rfm_score = random.randint(1, 5)  # Recency, Frequency, Monetary combined score
        cross_sell_index = random.randint(1, 10)
        churn_risk_score = random.uniform(0, 1)
        
        # Generate profitability based on the above features
        # This is a simplified model for demonstration purposes
        base_profit = 100
        
        # Income and account factors
        profit_factors = [
            annual_income * 0.001,  # Higher income -> higher profit
            avg_balance * 0.02,     # Higher balance -> higher profit
            fee_revenue,            # Direct revenue
            num_active_accounts * 50,  # More products -> higher profit
            years_with_bank * 20,   # Longer relationship -> higher profit
            cross_sell_index * 30,  # More cross-sell -> higher profit
        ]
        
        # Cost factors (subtract from profit)
        cost_factors = [
            branch_visits * 15,     # Branch visits are costly
            call_center_calls * 10, # Call center is costly
            complaints * 100,       # Complaints are costly
            overdraft_count * 5,    # Overdrafts have processing costs
            delinquency_count * 200,  # Delinquencies are costly
            ever_defaulted * 1000,  # Default is very costly
            churn_risk_score * 500  # Higher churn risk -> higher future costs
        ]
        
        # Calculate net profit with some randomness
        net_profit = base_profit + sum(profit_factors) - sum(cost_factors)
        net_profit += np.random.normal(0, 200)  # Add some noise
        
        customers.append({
            'customer_id': customer_id,
            'age': age,
            'sex': sex,
            'marital_status': marital_status,
            'dependents': dependents,
            'education_level': education_level,
            'employment_status': employment_status,
            'annual_income': annual_income,
            'residential_status': residential_status,
            'years_with_bank': years_with_bank,
            'num_active_accounts': num_active_accounts,
            'avg_balance': avg_balance,
            'total_deposits': total_deposits,
            'transaction_frequency': transaction_frequency,
            'digital_ratio': digital_ratio,
            'credit_utilization': credit_utilization,
            'overdraft_count': overdraft_count,
            'fee_revenue': fee_revenue,
            'credit_score': credit_score,
            'open_credit_lines': open_credit_lines,
            'delinquency_count': delinquency_count,
            'total_loan_amount': total_loan_amount,
            'ltv_ratio': ltv_ratio,
            'ever_defaulted': ever_defaulted,
            'branch_visits': branch_visits,
            'call_center_calls': call_center_calls,
            'complaints': complaints,
            'digital_logins': digital_logins,
            'rfm_score': rfm_score,
            'cross_sell_index': cross_sell_index,
            'churn_risk_score': churn_risk_score,
            'net_profit': net_profit
        })
    
    return pd.DataFrame(customers)

def main():
    """Main function to generate and save data"""
    print("Generating synthetic bank customer data...")
    df = generate_customer_data(num_samples=1000)
    
    # Save to CSV
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Data generated and saved to {RAW_DATA_PATH}")
    print(f"Generated {len(df)} customer records")
    
    # Display data summary
    print("\nData Summary:")
    print(f"Shape: {df.shape}")
    print("\nNumeric Features Summary:")
    print(df.describe().round(2).T[['count', 'mean', 'min', 'max']])
    
    # Create profit segments
    bins = [-float('inf'), 0, 500, 1000, float('inf')]
    labels = ['loss', 'low_profit', 'medium_profit', 'high_profit']
    df['profit_segment'] = pd.cut(df['net_profit'], bins=bins, labels=labels)
    
    # Show profit segment distribution
    print("\nProfit Segment Distribution:")
    print(df['profit_segment'].value_counts())
    
    # Show average profit by segment - fix FutureWarning by specifying observed parameter
    print("\nAverage Profit by Segment:")
    print(df.groupby('profit_segment', observed=True)['net_profit'].mean().round(2))

if __name__ == "__main__":
    main()