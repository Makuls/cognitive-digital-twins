import pandas as pd
from src.data_handler import DataHandler
from src.simulator import DigitalTwinSimulator  
from src.evaluator import StatisticalEvaluator
from src.visualizer import generate_dynamic_comparison_chart  

MODEL_CONFIGS = {
    "llama3": {
        "temperature": 0.85,
        "top_p": 0.95,
        "frequency_penalty": 0.1
    },
    "phi3": {
        "temperature": 0.40,
        "top_p": 0.85,
        "frequency_penalty": 0.2
    },
    "gemma2:9b": {
        "temperature": 0.85,
        "top_p": 0.92,
        "frequency_penalty": 0.1
    }
}

def run_comparative_analysis():
    print("Initializing automated multi-model digital twin research suite...\n")
    
    optimal_ratio = 0.20
    models_to_test = ["llama3", "phi3", "gemma2:9b"]  
    
    handler = DataHandler()
    sim = DigitalTwinSimulator()  
    evaluator = StatisticalEvaluator()
    
    calibration_df, target_df, human_control_df = handler.generate_hybrid_split(human_ratio=optimal_ratio)
    
    human_base_rates = {
        "Option A": 0.560,
        "Option B": 0.205,
        "Option C": 0.235
    }
    
    test_target_df = target_df.copy() 
    model_results = {}
    
    for model in models_to_test:
        print(f"\nGenerating cognitive digital twins via {model}...")
        print(f"Starting local RAG simulation for {len(test_target_df)} digital twins...")
        
        run_df = test_target_df.copy()
        
        model_cfg = MODEL_CONFIGS.get(model, {"temperature": 0.7, "top_p": 0.9, "frequency_penalty": 0.0})
        
        simulated_df = sim.simulate_choices(
            target_profiles_df=run_df,
            calibration_df=calibration_df,
            model_name=model,
            config=model_cfg,       
            base_rates=human_base_rates
        )
        model_results[model] = simulated_df
        
    print("\nFinal multi-model digital twin performance evaluation matrix:")
    
    for model, results_df in model_results.items():
        print(f"\nSystem architecture: {model}")
        evaluator.evaluate_tier1_jsd(human_control_df, results_df)
        print("  Sub-group divergence matrices:")
        evaluator.evaluate_tier2_subgroups(human_control_df, results_df, feature="income_tier")
        evaluator.evaluate_tier2_subgroups(human_control_df, results_df, feature="primary_use")
        print("-" * 40)

    generate_dynamic_comparison_chart(
        human_df=human_control_df, 
        model_results=model_results
    )

if __name__ == "__main__":
    run_comparative_analysis()