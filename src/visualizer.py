import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_dynamic_comparison_chart(human_df: pd.DataFrame, model_results: dict):
    print("Parsing multi-model payload and generating chart...")
    
    def get_percentages(df, column_name):
        total = len(df)
        if total == 0:
            return [0.0, 0.0, 0.0]
        counts = df[column_name].value_counts()
        return [
            (counts.get('Option A', 0) / total) * 100,
            (counts.get('Option B', 0) / total) * 100,
            (counts.get('Option C', 0) / total) * 100
        ]

    labels = ['Option A (Budget)', 'Option B (Mid-Tier)', 'Option C (Premium)']
    x = np.arange(len(labels))
    
    num_models = len(model_results)
    total_groups = num_models + 1
    width = 0.8 / total_groups
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    human_shares = get_percentages(human_df, 'human_choice')
    human_offset = x - (total_groups - 1) * width / 2
    rects = ax.bar(human_offset, human_shares, width, label='Real Humans (IBM Baseline)', color='#2c3e50')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    autolabel(rects)
    
    color_palette = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#f1c40f']
    
    for idx, (model_name, results_df) in enumerate(model_results.items()):
        synthetic_shares = get_percentages(results_df, 'synthetic_choice')
        
        model_offset = human_offset + ((idx + 1) * width)
        
        current_color = color_palette[idx % len(color_palette)]
        
        model_rects = ax.bar(model_offset, synthetic_shares, width, 
                             label=f'Synthetic ({model_name.upper()})', color=current_color)
        autolabel(model_rects)
    
    ax.set_ylabel('Market Share Percentage (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Cross-model algorithmic brand equity validation', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    all_values = human_shares + [share for res in model_results.values() for share in get_percentages(res, 'synthetic_choice')]
    ax.set_ylim(0, max(all_values) + 15)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('model_market_comparison.png', dpi=300)
    print("Chart saved as 'model_market_comparison.png'.")