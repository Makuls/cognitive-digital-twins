from src.data_handler import DataHandler
from src.simulator import ConsumerSimulator
from src.evaluator import StatisticalEvaluator

def run_tier2_analysis():
    print("Starting Tier 2 sub-group validation...\n")
    
    optimal_ratio = 0.20
    target_model = "llama3" 
    
    handler = DataHandler()
    sim = ConsumerSimulator()
    evaluator = StatisticalEvaluator()
    
    print(f"Splitting data using {optimal_ratio*100}% human calibration...")
    calibration_df, target_df, human_control_df = handler.generate_hybrid_split(human_ratio=optimal_ratio)
    
    test_target_df = target_df.copy() 
    
    simulated_results_df = sim.simulate_choices(
        target_profiles_df=test_target_df, 
        calibration_df=calibration_df, 
        model_name=target_model, 
        temperature=0.8
    )
    
    print(f"\nResults for {target_model.upper()}:")
    evaluator.evaluate_tier1_jsd(human_control_df, simulated_results_df)
    
    evaluator.evaluate_tier2_subgroups(human_control_df, simulated_results_df, feature="income_tier")
    evaluator.evaluate_tier2_subgroups(human_control_df, simulated_results_df, feature="primary_use")

if __name__ == "__main__":
    run_tier2_analysis()