import pandas as pd
import os
import time
from tqdm import tqdm
from openai import OpenAI
from src.retail_simulator import RetailDigitalTwinSimulator

def run_frontier_ablation():
    print("Initializing frontier model ablation study (Phi-3 vs Llama-3.3-70B)...\n")
    
    local_sim = RetailDigitalTwinSimulator() 
    
    frontier_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key='Your API Key Here',
    )
    frontier_model_name = "llama-3.3-70b-versatile"
    
    df_raw = pd.read_csv("data/raw/UCI_Online_Retail_Clean.csv")
    
    profiles = df_raw.groupby('CustomerID').agg(
        Total_Spend=('Total_Spend', 'sum'),
        Total_Transactions=('InvoiceNo', 'nunique'),
        Total_Items=('Quantity', 'sum'),
        Country=('Country', 'first')
    ).reset_index()
    
    q50 = profiles['Total_Spend'].quantile(0.50)
    q85 = profiles['Total_Spend'].quantile(0.85)
    
    def assign_tier(spend):
        if spend >= q85: return 'VIP'
        elif spend >= q50: return 'Mid-Tier'
        else: return 'Budget'
        
    profiles['human_choice'] = profiles['Total_Spend'].apply(assign_tier)
    
    sample_df = profiles.sample(n=100, random_state=42).copy()
    calibration_df = sample_df.iloc[:20].copy()
    target_df = sample_df.iloc[20:].copy()
    
    base_rates = calibration_df['human_choice'].value_counts(normalize=True).to_dict()
    
    print(f"Extracting {len(calibration_df)} real humans for context base...")
    print(f"Running head-to-head simulation on {len(target_df)} target profiles...\n")
    
    phi3_choices = []
    frontier_choices = []
    
    progress_bar = tqdm(target_df.iterrows(), total=len(target_df), desc="Running ablation")
    
    for index, row in progress_bar:
        age, income, education = row['Country'], row['Total_Transactions'], row['Total_Items']
        
        context_text = local_sim._get_lookalike_context(age, income, education, calibration_df)
        prompt = local_sim._build_retail_prompt(age, income, education, context_text, base_rates)
        
        try:
            phi3_response = local_sim.client.chat.completions.create(
                model="phi3",
                messages=[{"role": "system", "content": "Output exactly one word from the choices."}, {"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=10
            )
            raw_phi3 = phi3_response.choices[0].message.content.strip().title()
            
            if "Budget" in raw_phi3: phi3_choices.append("Budget")
            elif "Mid" in raw_phi3: phi3_choices.append("Mid-Tier")
            elif "Vip" in raw_phi3: phi3_choices.append("VIP")
            else: phi3_choices.append("Budget")
            
        except Exception as e:
            print(f"\nLocal Phi-3 error on profile {index}: {e}")
            phi3_choices.append("Budget")
            
        try:
            frontier_response = frontier_client.chat.completions.create(
                model=frontier_model_name,
                messages=[{"role": "system", "content": "Output exactly one word from the choices."}, {"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=10
            )
            raw_frontier = frontier_response.choices[0].message.content.strip().title()
            
            if "Budget" in raw_frontier: frontier_choices.append("Budget")
            elif "Mid" in raw_frontier: frontier_choices.append("Mid-Tier")
            elif "Vip" in raw_frontier: frontier_choices.append("VIP")
            else: frontier_choices.append("Budget")
            
        except Exception as e:
            print(f"\nGroq API error on profile {index}: {e}")
            frontier_choices.append("Budget")
            
        time.sleep(2.1) 

    target_df['Phi3_Choice'] = phi3_choices
    target_df['Frontier_Choice'] = frontier_choices
    
    real_dist = target_df['human_choice'].value_counts(normalize=True) * 100
    phi3_dist = target_df['Phi3_Choice'].value_counts(normalize=True) * 100
    frontier_dist = target_df['Frontier_Choice'].value_counts(normalize=True) * 100
    
    print("\nAblation results: small vs frontier architecture")
    for cat in ['Budget', 'Mid-Tier', 'VIP']:
        r_pct = real_dist.get(cat, 0.0)
        p_pct = phi3_dist.get(cat, 0.0)
        f_pct = frontier_dist.get(cat, 0.0)
        print(f"{cat.upper()} tier (True baseline: {r_pct:.1f}%):")
        print(f"  Phi-3 (local):    {p_pct:.1f}% (Variance: {abs(r_pct - p_pct):.1f}%)")
        print(f"  Llama-3.3 70B:    {f_pct:.1f}% (Variance: {abs(r_pct - f_pct):.1f}%)")
        print("-" * 50)

if __name__ == "__main__":
    run_frontier_ablation()