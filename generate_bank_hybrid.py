import pandas as pd
import os
from src.bank_simulator import BankDigitalTwinSimulator

def generate_hybrid_bank_dataset():
    print("Initializing hybrid banking dataset generation...\n")
    
    df = pd.read_csv("data/raw/BankChurners.csv")
    df = df[['Customer_Age', 'Income_Category', 'Education_Level', 'Card_Category']].copy()
    df.rename(columns={'Card_Category': 'human_choice'}, inplace=True)
    df['human_choice'] = df['human_choice'].astype(str).str.strip()
    
    sample_df = df.sample(n=1000, random_state=42).copy()
    
    calibration_size = int(len(sample_df) * 0.20)
    calibration_df = sample_df.iloc[:calibration_size].copy()
    target_df = sample_df.iloc[calibration_size:].copy()
    
    base_rates = calibration_df['human_choice'].value_counts(normalize=True).to_dict()
    
    print(f"Extracting {len(calibration_df)} real humans for context base (20%)...")
    print(f"Generating {len(target_df)} synthetic twins (80%)...")
    
    sim = BankDigitalTwinSimulator()
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
    
    clean_calibration = calibration_df[['Customer_Age', 'Income_Category', 'Education_Level', 'data_source', 'final_choice']]
    clean_results = results_df[['Customer_Age', 'Income_Category', 'Education_Level', 'data_source', 'final_choice']]
    
    final_hybrid_df = pd.concat([clean_calibration, clean_results], ignore_index=True)
    
    os.makedirs('results', exist_ok=True)
    output_path = "results/final_hybrid_bank_population.csv"
    final_hybrid_df.to_csv(output_path, index=False)
    
    print("\nHybrid banking system successfully generated.")
    print(f"Total population size: {len(final_hybrid_df)} consumers")
    print(f"File saved to: {output_path}\n")
    
    final_dist = final_hybrid_df['final_choice'].value_counts(normalize=True) * 100
    print("Final aggregate market share of hybrid system:")
    for cat, pct in final_dist.items():
        print(f"  {cat.upper()}: {pct:.1f}%")

if __name__ == "__main__":
    generate_hybrid_bank_dataset()