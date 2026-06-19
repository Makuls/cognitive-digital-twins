import pandas as pd
import warnings
from run_multi_agent_bank import MultiAgentBankSimulator

warnings.filterwarnings('ignore')

def run_multi_agent_validation():
    print("Initializing multi-agent debate validation...\n")
    
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
    print(f"Target pool: {len(target_df)}\n")
    
    sim = MultiAgentBankSimulator()
    config = {"temperature": 0.40}
    
    print("Generating digital twins via Phi-3 multi-agent debate...")
    
    results_df = sim.simulate_choices(
        target_profiles_df=target_df,
        calibration_df=calibration_df,
        model_name="phi3",
        config=config,
        base_rates=base_rates
    )
    
    real_dist = target_df['human_choice'].value_counts(normalize=True) * 100
    synthetic_dist = results_df['synthetic_choice'].value_counts(normalize=True) * 100
    
    print("\nFinal validation matrix:")
    for cat in ['Blue', 'Silver', 'Gold', 'Platinum']:
        real_pct = real_dist.get(cat, 0.0)
        syn_pct = synthetic_dist.get(cat, 0.0)
        print(f"{cat.upper()} tier:")
        print(f"  Real humans: {real_pct:.1f}%")
        print(f"  Synthetic twins: {syn_pct:.1f}%")
        print(f"  Variance: {abs(real_pct - syn_pct):.1f}%")
        print("-" * 30)

if __name__ == "__main__":
    run_multi_agent_validation()