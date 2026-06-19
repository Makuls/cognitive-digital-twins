import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from src.data_handler import DataHandler
from src.simulator import DigitalTwinSimulator  

MODEL_CONFIGS = {
    "llama3": {"temperature": 0.85, "top_p": 0.95, "frequency_penalty": 0.1},
    "phi3": {"temperature": 0.40, "top_p": 0.85, "frequency_penalty": 0.2},
    "gemma2:9b": {"temperature": 0.85, "top_p": 0.92, "frequency_penalty": 0.1}
}

def calculate_jsd(human_choices, synthetic_choices):
    """Calculates Jensen-Shannon Divergence between two choice distributions."""
    options = ["Option A", "Option B", "Option C"]
    
    p = np.array([(human_choices == opt).mean() for opt in options])
    q = np.array([(synthetic_choices == opt).mean() for opt in options])
    
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p /= p.sum()
    q /= q.sum()
    
    return distance.jensenshannon(p, q) ** 2  

def generate_validity_frontier():
    print("Initializing RAG validity frontier optimization...\n")
    
    test_ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    models = ["llama3", "phi3", "gemma2:9b"]
    
    handler = DataHandler()
    sim = DigitalTwinSimulator()
    
    frontier_data = {model: [] for model in models}
    
    human_base_rates = {"Option A": 0.560, "Option B": 0.205, "Option C": 0.235}
    
    for ratio in test_ratios:
        print(f"\nTesting calibration ratio: {ratio * 100:.0f}%")
        
        calibration_df, target_df, human_control_df = handler.generate_hybrid_split(human_ratio=ratio)
        
        test_target_df = target_df.sample(n=100, random_state=42).copy()
        
        for model in models:
            model_cfg = MODEL_CONFIGS.get(model)
            
            simulated_df = sim.simulate_choices(
                target_profiles_df=test_target_df.copy(),
                calibration_df=calibration_df,
                model_name=model,
                config=model_cfg,       
                base_rates=human_base_rates
            )
            
            jsd_score = calculate_jsd(human_control_df['human_choice'], simulated_df['synthetic_choice'])
            frontier_data[model].append(jsd_score)
            print(f"  {model.upper()} JSD @ {ratio*100:.0f}%: {jsd_score:.4f}")

    plt.figure(figsize=(12, 7))
    
    colors = {"llama3": "#3498db", "phi3": "#2ecc71", "gemma2:9b": "#9b59b6"}
    markers = {"llama3": "o", "phi3": "s", "gemma2:9b": "^"}
    
    for model in models:
        plt.plot(
            [r * 100 for r in test_ratios], 
            frontier_data[model], 
            marker=markers[model], 
            color=colors[model], 
            linewidth=2, 
            markersize=8,
            label=f"Synthetic ({model.upper()})"
        )
        
    plt.axvline(x=20, color='#e74c3c', linestyle='--', linewidth=2, label="Chosen Optimal Ratio (20%)")

    plt.title("RAG Validity Frontier: JSD vs. Human Calibration Ratio\n(Lower JSD = Higher Digital Twin Fidelity)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Human Calibration Ratio in RAG Vector Space (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Jensen-Shannon Divergence (JSD)", fontsize=12, fontweight='bold')
    plt.ylim(0, max([max(scores) for scores in frontier_data.values()]) * 1.2)
    plt.xticks([r * 100 for r in test_ratios])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/updated_validity_frontier.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFrontier analysis complete. Chart saved to: {save_path}")

if __name__ == "__main__":
    generate_validity_frontier()