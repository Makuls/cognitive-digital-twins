import pandas as pd
import warnings
from src.retail_simulator import RetailDigitalTwinSimulator

warnings.filterwarnings('ignore')

def run_retail_validation():
    print("Initializing retail validation...\n")
    
    df_raw = pd.read_csv("data/raw/UCI_Online_Retail_Clean.csv")
    
    print("Aggregating transactions into consumer profiles...")
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
    
    sample_df = profiles.sample(n=500, random_state=42).copy()
    
    calibration_size = int(len(sample_df) * 0.20)
    calibration_df = sample_df.iloc[:calibration_size].copy()
    target_df = sample_df.iloc[calibration_size:].copy()
    
    base_rates = calibration_df['human_choice'].value_counts(normalize=True).to_dict()
    
    print(f"Total unique consumers: {len(profiles)}")
    print(f"Calibration pool: {len(calibration_df)}")
    print(f"Target pool: {len(target_df)}")
    print(f"Market priors: {base_rates}\n")
    
    sim = RetailDigitalTwinSimulator()
    config = {"temperature": 0.40, "top_p": 0.85, "frequency_penalty": 0.2}
    
    print("Generating digital twins via Phi-3...")
    
    results_df = sim.simulate_choices(
        target_profiles_df=target_df,
        calibration_df=calibration_df,
        model_name="phi3",
        config=config,
        base_rates=base_rates
    )
    
    real_distribution = target_df['human_choice'].value_counts(normalize=True) * 100
    synthetic_distribution = results_df['synthetic_choice'].value_counts(normalize=True) * 100
    
    print("\nFinal validation matrix:")
    categories = ['Budget', 'Mid-Tier', 'VIP']
    for cat in categories:
        real_pct = real_distribution.get(cat, 0.0)
        syn_pct = synthetic_distribution.get(cat, 0.0)
        print(f"{cat.upper()} tier:")
        print(f"  Real humans: {real_pct:.1f}%")
        print(f"  Synthetic twins: {syn_pct:.1f}%")
        print(f"  Variance: {abs(real_pct - syn_pct):.1f}%")
        print("-" * 30)

if __name__ == "__main__":
    run_retail_validation()