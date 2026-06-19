import pandas as pd
from src.bank_simulator import BankDigitalTwinSimulator

def run_bank_validation():
    print("Initializing banking validation...\n")
    
    df = pd.read_csv("data/raw/BankChurners.csv")
    
    df = df[['Customer_Age', 'Income_Category', 'Education_Level', 'Card_Category']].copy()
    df.rename(columns={'Card_Category': 'human_choice'}, inplace=True)
    
    df['human_choice'] = df['human_choice'].astype(str).str.strip()
    
    sample_df = df.sample(n=500, random_state=42).copy()
    
    calibration_size = int(len(sample_df) * 0.20)
    calibration_df = sample_df.iloc[:calibration_size].copy()
    target_df = sample_df.iloc[calibration_size:].copy()
    
    base_rates = calibration_df['human_choice'].value_counts(normalize=True).to_dict()
    
    print(f"Calibration pool: {len(calibration_df)}")
    print(f"Target pool: {len(target_df)}")
    print(f"Market priors: {base_rates}\n")
    
    sim = BankDigitalTwinSimulator()
    config = {
        "temperature": 0.40,
        "top_p": 0.85,
        "frequency_penalty": 0.2
    }
    
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
    categories = ['Blue', 'Silver', 'Gold', 'Platinum']
    for cat in categories:
        real_pct = real_distribution.get(cat, 0.0)
        syn_pct = synthetic_distribution.get(cat, 0.0)
        print(f"{cat.upper()} tier:")
        print(f"  Real humans: {real_pct:.1f}%")
        print(f"  Synthetic twins: {syn_pct:.1f}%")
        print(f"  Variance: {abs(real_pct - syn_pct):.1f}%")
        print("-" * 30)

if __name__ == "__main__":
    run_bank_validation()