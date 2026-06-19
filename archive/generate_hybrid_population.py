import os
import pandas as pd
from src.data_handler import DataHandler
from src.simulator import DigitalTwinSimulator  

def generate_final_hybrid_system():
    print("Initializing final hybrid population deployment...")
    print("Thesis phase: Deployment")
    print("Selected architecture: Microsoft Phi-3 (Context-Sensitive Adapter)")
    print("Calibration threshold: 20% Real Humans / 80% Synthetic Twins\n")
    
    handler = DataHandler()
    sim = DigitalTwinSimulator()
    
    calibration_df, target_df, human_control_df = handler.generate_hybrid_split(human_ratio=0.20)
    
    human_base_rates = {
        "Option A": 0.560,
        "Option B": 0.205,
        "Option C": 0.235
    }
    
    phi3_config = {
        "temperature": 0.40,
        "top_p": 0.85,
        "frequency_penalty": 0.2
    }
    
    print(f"Generating {len(target_df)} synthetic consumers via Phi-3...")
    simulated_df = sim.simulate_choices(
        target_profiles_df=target_df.copy(),
        calibration_df=calibration_df,
        model_name="phi3",
        config=phi3_config,       
        base_rates=human_base_rates
    )
    
    print("\nMerging real and synthetic populations into hybrid system...")
    
    real_humans = calibration_df.copy()
    real_humans['data_source'] = 'Real Human (20% Calibration Base)'
    real_humans['final_choice'] = real_humans['human_choice']
    real_humans = real_humans.drop(columns=['human_choice'])
    
    synthetic_twins = simulated_df.copy()
    synthetic_twins['data_source'] = 'Synthetic Twin (Phi-3 Engine)'
    synthetic_twins['final_choice'] = synthetic_twins['synthetic_choice']
    synthetic_twins = synthetic_twins.drop(columns=['synthetic_choice'])
    
    if 'human_choice' in synthetic_twins.columns:
        synthetic_twins = synthetic_twins.drop(columns=['human_choice'])
        
    hybrid_dataset = pd.concat([real_humans, synthetic_twins], ignore_index=True)
    
    hybrid_dataset = hybrid_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/final_hybrid_population.csv'
    hybrid_dataset.to_csv(save_path, index=False)
    
    print("\nHybrid system successfully generated.")
    print(f"Total population size: {len(hybrid_dataset)} consumers")
    print(f"File saved to: {save_path}")
    print("\nFinal aggregate market share of hybrid system:")
    
    market_share = hybrid_dataset['final_choice'].value_counts(normalize=True) * 100
    for option, share in market_share.items():
        print(f"  {option}: {share:.1f}%")

if __name__ == "__main__":
    generate_final_hybrid_system()