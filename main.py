from src.data_handler import DataHandler
from src.simulator import ConsumerSimulator
from src.evaluator import StatisticalEvaluator

def run_pipeline():
    print("Initializing full dataset calibrated RAG pipeline...\n")
    
    handler = DataHandler()
    calibration_df, target_df, human_control_df = handler.generate_hybrid_split(human_ratio=0.2)
    
    sim = ConsumerSimulator()
    
    test_target_df = target_df.copy() 
    
    simulated_results_df = sim.simulate_choices(
        target_profiles_df=test_target_df, 
        calibration_df=calibration_df, 
        temperature=0.8
    )
    
    evaluator = StatisticalEvaluator()
    evaluator.evaluate_tier1_jsd(human_control_df, simulated_results_df)

if __name__ == "__main__":
    run_pipeline()