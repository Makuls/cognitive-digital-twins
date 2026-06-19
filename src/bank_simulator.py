import pandas as pd
from openai import OpenAI
from tqdm import tqdm

class BankDigitalTwinSimulator:
    """
    Engine for generating cognitive digital twins of banking consumers 
    predicting credit card tier adoption using asymmetric temperature scaling.
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
        subset = subset.sort_values('age_diff')
        
        best_matches = subset.head(3)
            
        context_strings = []
        for i, row in enumerate(best_matches.iterrows(), 1):
            data = row[1]
            context_strings.append(
                f"- Peer {i}: A {data['Customer_Age']}-year-old (Income: {data['Income_Category']}, Education: {data['Education_Level']}) chose the {data['human_choice']} card."
            )
            
        return "\n".join(context_strings)

    def _build_calibrated_prompt(self, age, income, education, context_text, base_rates):
        return f"""You are simulating a realistic banking customer choosing a credit card tier.

Your demographics:
- Age: {age}
- Income category: {income}
- Education level: {education}

General market context (real world baseline):
Historically, the general population makes these highly imbalanced choices:
- Blue card: {base_rates.get('Blue', 0)*100:.1f}%
- Silver card: {base_rates.get('Silver', 0)*100:.1f}%
- Gold card: {base_rates.get('Gold', 0)*100:.1f}%
- Platinum card: {base_rates.get('Platinum', 0)*100:.1f}%

Your specific peer context:
Here is what similar customers chose:
{context_text}

The purchasing options:
- Blue: Standard free tier (No annual fee).
- Silver: Mid-tier with some benefits (Moderate fee).
- Gold: Premium tier with high benefits (High fee).
- Platinum: Elite tier for very high spenders (Very high fee).

Psychological constraint:
Real-world consumers evaluate trade-offs carefully. While high-income individuals have the capacity for premium cards (Gold/Platinum), many still prefer standard free tiers (Blue) if the benefits do not justify the cost. Weigh both the desire for premium perks and natural human frugality based on your specific income bracket.

Instructions:
Act as this specific consumer. Based on your income, education, peer behavior, and balanced psychological constraints, which card do you choose?
Output ONLY your final decision as a single word: "Blue", "Silver", "Gold", or "Platinum". Do not include any other text.
"""

    def simulate_choices(self, target_profiles_df: pd.DataFrame, calibration_df: pd.DataFrame, model_name: str, config: dict, base_rates: dict) -> pd.DataFrame:
        synthetic_choices = []
        
        progress_bar = tqdm(
            target_profiles_df.iterrows(), 
            total=len(target_profiles_df), 
            desc="Generating banking twins"
        )
        
        for index, row in progress_bar:
            age = row['Customer_Age']
            income = row['Income_Category']
            education = row['Education_Level']
            
            context_text = self._get_lookalike_context(age, income, education, calibration_df)
            prompt = self._build_calibrated_prompt(age, income, education, context_text, base_rates)
            
            if income in ["Less than $40K", "$40K - $60K", "Unknown"]:
                dynamic_temp = 0.2
            elif income in ["$60K - $80K"]:
                dynamic_temp = 0.4
            else:
                dynamic_temp = 0.65 
                
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a realistic consumer simulator evaluating cost vs value. Output exactly one word from the choices."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=dynamic_temp,
                    top_p=config.get("top_p", 0.85),
                    frequency_penalty=config.get("frequency_penalty", 0.2),
                    max_tokens=10 
                )
                
                choice = response.choices[0].message.content.strip().title()
                
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