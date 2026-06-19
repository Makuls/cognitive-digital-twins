import pandas as pd
from openai import OpenAI
from tqdm import tqdm

class RetailDigitalTwinSimulator:
    """
    Engine for simulating retail e-commerce tier adoption based on 
    RFM consumer dynamics.
    """
    
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama', 
        )
        
    def _get_lookalike_context(self, country, transactions, items, calibration_df):
        subset = calibration_df[calibration_df['Country'] == country].copy()
        
        if len(subset) < 3:
            subset = calibration_df.copy()
            
        subset['diff'] = abs(subset['Total_Transactions'] - transactions) + (abs(subset['Total_Items'] - items) * 0.1)
        best_matches = subset.sort_values('diff').head(3)
            
        context_strings = []
        for i, row in enumerate(best_matches.iterrows(), 1):
            data = row[1]
            context_strings.append(
                f"- Peer {i}: A shopper from {data['Country']} with {data['Total_Transactions']} distinct orders ({data['Total_Items']} total items) is a {data['human_choice']} spender."
            )
            
        return "\n".join(context_strings)

    def _build_retail_prompt(self, country, transactions, items, context_text, base_rates):
        return f"""You are simulating a real consumer in an online retail environment.

Your Shopping Profile:
- Location: {country}
- Total Distinct Orders (Frequency): {transactions}
- Total Items Purchased (Volume): {items}

General Market Context (Real World Baseline):
Historically, the customer base distributes into these spending tiers:
- Budget Spender (Bottom 50% of revenue): {base_rates.get('Budget', 0)*100:.1f}%
- Mid-Tier Spender (Next 35% of revenue): {base_rates.get('Mid-Tier', 0)*100:.1f}%
- VIP Spender (Top 15% of revenue): {base_rates.get('VIP', 0)*100:.1f}%

Your Specific Peer Context:
Here is what similar shoppers were classified as:
{context_text}

Instructions:
Act as this specific consumer. Based on your order frequency, item volume, and peer context, which spending tier do you fall into?
Output ONLY your final decision as a single word: "Budget", "Mid-Tier", or "VIP". Do not include any other text.
"""

    def simulate_choices(self, target_profiles_df: pd.DataFrame, calibration_df: pd.DataFrame, model_name: str, config: dict, base_rates: dict) -> pd.DataFrame:
        synthetic_choices = []
        
        progress_bar = tqdm(
            target_profiles_df.iterrows(), 
            total=len(target_profiles_df), 
            desc=f"Generating {model_name.upper()} retail twins"
        )
        
        for index, row in progress_bar:
            context_text = self._get_lookalike_context(
                row['Country'], row['Total_Transactions'], row['Total_Items'], calibration_df
            )
            
            prompt = self._build_retail_prompt(
                row['Country'], row['Total_Transactions'], row['Total_Items'], context_text, base_rates
            )
                
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a realistic e-commerce consumer simulator. Output exactly one word from the choices."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.get("temperature", 0.4), 
                    top_p=config.get("top_p", 0.85),
                    frequency_penalty=config.get("frequency_penalty", 0.2),
                    max_tokens=10 
                )
                
                choice = response.choices[0].message.content.strip().title()
                
                if "Budget" in choice: choice = "Budget"
                elif "Mid" in choice: choice = "Mid-Tier"
                elif "Vip" in choice: choice = "VIP"
                else: choice = "Budget"
                
                synthetic_choices.append(choice)
                
            except Exception as e:
                print(f"Error simulating profile {index}: {e}")
                synthetic_choices.append("Budget")
                
        target_profiles_df['synthetic_choice'] = synthetic_choices
        return target_profiles_df