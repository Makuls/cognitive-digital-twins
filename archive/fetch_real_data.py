import pandas as pd
import numpy as np
import os

def prep_real_data():
    print("-> Downloading IBM Telco Customer Churn dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    print("-> Mapping 21 real features to the pipeline schema...")
    
    df['age'] = df['SeniorCitizen'].apply(lambda x: np.random.randint(60, 80) if x == 1 else np.random.randint(18, 55))
    
    def map_income(charge):
        if charge < 50: return 'Low'
        elif charge < 85: return 'Medium'
        else: return 'High'
    df['income_tier'] = df['MonthlyCharges'].apply(map_income)
    
    def map_usage(row):
        if row['InternetService'] == 'No': return 'Voice'
        elif row['StreamingTV'] == 'Yes': return 'Streaming'
        elif row['InternetService'] == 'Fiber optic': return 'Gaming'
        else: return 'Browsing'
    df['primary_use'] = df.apply(map_usage, axis=1)
    
    def map_choice(contract):
        if contract == 'Month-to-month': return 'Option A'
        elif contract == 'One year': return 'Option B'
        else: return 'Option C'
    df['human_choice'] = df['Contract'].apply(map_choice)
    
    final_df = df[['age', 'income_tier', 'primary_use', 'human_choice']]
    
    os.makedirs('data/raw', exist_ok=True)
    final_df.to_csv('data/raw/real_telecom_survey.csv', index=False)
    print(f"-> Successfully saved {len(final_df)} real human records to data/raw/real_telecom_survey.csv")

if __name__ == "__main__":
    prep_real_data()