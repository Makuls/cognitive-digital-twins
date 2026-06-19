import pandas as pd
import os
import warnings
from src.retail_simulator import RetailDigitalTwinSimulator

warnings.filterwarnings('ignore')

def generate_hybrid_retail_dataset():
    print("Initializing hybrid retail dataset generation...\n")
    
    df_raw = pd.read_csv("data/raw/UCI_Online_Retail_Clean.csv")
    
    print("Aggregating transactions into distinct consumer profiles...")
    profiles = df_raw.groupby('CustomerID').agg(
        Total_Spend=('Total_Spend', 'sum'),
        Total_Transactions=('InvoiceNo', 'nunique'),
        Total_Items=('Quantity', 'sum'),
        Country=('Country', 'first')
    ).reset_index()
    
    q50 = profiles['Total_Spend'].quantile(0.50)
    q85 = profiles['Total_Spend'].quantile(0.85)
    
    def assign_tier(spend):
        if spend >= q85: return 'VIP'
        elif spend >= q50: return 'Mid-Tier'
        else: return 'Budget'
        
    profiles['human_choice'] = profiles['Total_Spend'].apply(assign_tier)
    
    sample_df = profiles.sample(n=1000, random_state=42).copy()
    
    calibration_size = int(len(sample_df) * 0.20)
    calibration_df = sample_df.iloc[:calibration_size].copy()
    target_df = sample_df.iloc[calibration_size:].copy()
    
    base_rates = calibration_df['human_choice'].value_counts(normalize=True).to_dict()
    
    print(f"Extracting {len(calibration_df)} real humans for context base (20%)...")
    print(f"Generating {len(target_df)} synthetic twins (80%)...")
    
    sim = RetailDigitalTwinSimulator()
    config = {
        "temperature": 0.40,
        "top_p": 0.85,
        "frequency_penalty": 0.2
    }
    
    results_df = sim.simulate_choices(
        target_profiles_df=target_df,
        calibration_df=calibration_df,
        model_name="phi3",
        config=config,
        base_rates=base_rates
    )
    
    print("\nMerging real and synthetic populations into hybrid system...")
    calibration_df['data_source'] = 'Real Human (20% Calibration Base)'
    calibration_df['final_choice'] = calibration_df['human_choice']
    
    results_df['data_source'] = 'Synthetic Twin (Phi-3 Engine)'
    results_df['final_choice'] = results_df['synthetic_choice']
    
    clean_cols = ['CustomerID', 'Country', 'Total_Transactions', 'Total_Items', 'Total_Spend', 'data_source', 'final_choice']
    clean_calibration = calibration_df[clean_cols]
    clean_results = results_df[clean_cols]
    
    final_hybrid_df = pd.concat([clean_calibration, clean_results], ignore_index=True)
    
    os.makedirs('results', exist_ok=True)
    output_path = "results/final_hybrid_retail_population.csv"
    final_hybrid_df.to_csv(output_path, index=False)
    
    print("\nHybrid retail system successfully generated.")
    print(f"Total population size: {len(final_hybrid_df)} consumers")
    print(f"File saved to: {output_path}\n")
    
    final_dist = final_hybrid_df['final_choice'].value_counts(normalize=True) * 100
    print("Final aggregate market share of hybrid system:")
    for cat, pct in final_dist.items():
        print(f"  {cat.upper()}: {pct:.1f}%")

if __name__ == "__main__":
    generate_hybrid_retail_dataset()