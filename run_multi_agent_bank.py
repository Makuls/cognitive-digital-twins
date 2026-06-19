import pandas as pd
from openai import OpenAI
from tqdm import tqdm

class MultiAgentBankSimulator:
    """
    Multi-agent architecture resolving hyper-frugality bias via persona debate.
    """
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama', 
        )
        
    def _get_lookalike_context(self, age, income, education, calibration_df):
        subset = calibration_df[
            (calibration_df['Income_Category'] == income) & 
            (calibration_df['Education_Level'] == education)
        ].copy()
        if len(subset) < 3:
            subset = calibration_df[calibration_df['Income_Category'] == income].copy()
        if len(subset) < 3:
            subset = calibration_df.copy()
            
        subset['age_diff'] = abs(subset['Customer_Age'] - age)
        best_matches = subset.sort_values('age_diff').head(3)
            
        context_strings = []
        for i, row in enumerate(best_matches.iterrows(), 1):
            data = row[1]
            context_strings.append(
                f"- Peer {i}: A {data['Customer_Age']}-year-old (Income: {data['Income_Category']}, Education: {data['Education_Level']}) chose the {data['human_choice']} card."
            )
        return "\n".join(context_strings)

    def simulate_choices(self, target_profiles_df: pd.DataFrame, calibration_df: pd.DataFrame, model_name: str, config: dict, base_rates: dict) -> pd.DataFrame:
        synthetic_choices = []
        
        progress_bar = tqdm(
            target_profiles_df.iterrows(), 
            total=len(target_profiles_df), 
            desc="Generating multi-agent twins"
        )
        
        for index, row in progress_bar:
            age, income, education = row['Customer_Age'], row['Income_Category'], row['Education_Level']
            context_text = self._get_lookalike_context(age, income, education, calibration_df)
            
            profile_summary = f"Customer Profile: {age} years old, Income: {income}, Education: {education}\nPeer Behavior:\n{context_text}"
            
            try:
                agent_a_response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a sales advocate. Read the customer profile and argue in 2 sentences why they deserve the premium benefits of the Gold or Platinum card."} ,
                        {"role": "user", "content": profile_summary}
                    ],
                    temperature=0.6, max_tokens=60
                ).choices[0].message.content.strip()
                
                agent_b_response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a frugal financial advisor. Read the customer profile and argue in 2 sentences why they should save money and stick to the free Blue or Silver card."} ,
                        {"role": "user", "content": profile_summary}
                    ],
                    temperature=0.6, max_tokens=60
                ).choices[0].message.content.strip()
                
                judge_prompt = f"""{profile_summary}
                
                Argument A (Premium): {agent_a_response}
                Argument B (Frugal): {agent_b_response}
                
                Instructions: Weigh these arguments against the customer's actual income and peer context. Decide the final card tier.
                Output ONLY your final decision as a single word: "Blue", "Silver", "Gold", or "Platinum"."""
                
                judge_response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are the final judge evaluating financial trade-offs. Output exactly one word."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=config.get("temperature", 0.4), max_tokens=10
                ).choices[0].message.content.strip().title()
                
                choice = judge_response
                if "Blue" in choice: choice = "Blue"
                elif "Silver" in choice: choice = "Silver"
                elif "Gold" in choice: choice = "Gold"
                elif "Platinum" in choice: choice = "Platinum"
                else: choice = "Blue"
                
                synthetic_choices.append(choice)
                
            except Exception as e:
                print(f"Error simulating profile {index}: {e}")
                synthetic_choices.append("Blue")
                
        target_profiles_df['synthetic_choice'] = synthetic_choices
        return target_profiles_df

if __name__ == "__main__":
    print("Multi-agent bank simulator loaded.")