import os
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, data_path: str = "data/raw/telecom_survey.csv", mode: str = "draft"):
        self.data_path = data_path
        
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        if not os.path.exists(self.data_path):
            self._prepare_dataset()
            
        base_df = pd.read_csv('data/raw/real_telecom_survey.csv')
        
        if mode == "draft":
            self.df = base_df.sample(n=1000, random_state=42)
        elif mode == "production":
            self.df = base_df
        else:
            self.df = base_df.sample(n=200, random_state=42)
            
    def _prepare_dataset(self):
        real_file_path = "data/raw/real_telecom_survey.csv"
        
        if os.path.exists(real_file_path):
            print(f"Found real dataset at {real_file_path}. Processing...")
            df = pd.read_csv(real_file_path)
            df = df.reset_index(drop=True)
            df.to_csv(self.data_path, index=False)
            print(f"Successfully synchronized target dataset to: {self.data_path}")
        else:
            print(f"Real dataset missing at {real_file_path}. Generating synthetic fallback data...")
            n_samples = 1000
            np.random.seed(42)
            
            ages = np.random.randint(18, 65, size=n_samples)
            income_tiers = np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.3, 0.5, 0.2])
            primary_use = np.random.choice(["Streaming", "Gaming", "Browsing"], size=n_samples)
            
            choices = []
            for inc, use in zip(income_tiers, primary_use):
                if inc == "Low":
                    choices.append(np.random.choice(["Option A", "Option B", "Option C"], p=[0.7, 0.2, 0.1]))
                elif use == "Streaming" or inc == "High":
                    choices.append(np.random.choice(["Option A", "Option B", "Option C"], p=[0.1, 0.3, 0.6]))
                else:
                    choices.append(np.random.choice(["Option A", "Option B", "Option C"], p=[0.3, 0.4, 0.3]))
                    
            df = pd.DataFrame({
                "user_id": range(1, n_samples + 1),
                "age": ages,
                "income_tier": income_tiers,
                "primary_use": primary_use,
                "human_choice": choices
            })
            
            df.to_csv(self.data_path, index=False)
            print(f"Successfully created fallback survey dataset at: {self.data_path}\n")

    def generate_hybrid_split(self, human_ratio: float, test_size: float = 0.2):
        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        test_split_idx = int(len(shuffled_df) * (1 - test_size))
        train_pool = shuffled_df.iloc[:test_split_idx].reset_index(drop=True)
        hidden_human_test = shuffled_df.iloc[test_split_idx:].reset_index(drop=True)
        
        calibration_idx = int(len(train_pool) * human_ratio)
        
        calibration_human_slice = train_pool.iloc[:calibration_idx].reset_index(drop=True)
        synthetic_target_profiles = train_pool.iloc[calibration_idx:].reset_index(drop=True)
        
        print("Data pipeline split complete:")
        print(f"Hidden human validation pool: {len(hidden_human_test)} records")
        print(f"Active calibration human slice ({human_ratio * 100}%): {len(calibration_human_slice)} records")
        print(f"Synthetic digital twins to simulate ({(1.0 - human_ratio) * 100:.1f}%): {len(synthetic_target_profiles)} records\n")
        
        return calibration_human_slice, synthetic_target_profiles, hidden_human_test

if __name__ == "__main__":
    handler = DataHandler(mode="draft")
    handler.generate_hybrid_split(human_ratio=0.20)