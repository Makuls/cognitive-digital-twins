import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

print("🚀 Fetching the Official UCI Online Retail Dataset...")
online_retail = fetch_ucirepo(id=352)

df = online_retail.data.original

df_clean = df.dropna(subset=['CustomerID'])
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean['Total_Spend'] = df_clean['Quantity'] * df_clean['UnitPrice']

os.makedirs('data/raw', exist_ok=True)
output_path = 'data/raw/UCI_Online_Retail_Clean.csv'
df_clean.to_csv(output_path, index=False)

print(f"✅ Dataset saved successfully to: {output_path}")
print(f"Total pure human transactions ready for simulation: {len(df_clean)}")