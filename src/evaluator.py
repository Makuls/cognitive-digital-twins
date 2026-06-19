import pandas as pd
import numpy as np
from scipy.spatial import distance

class StatisticalEvaluator:
    def __init__(self, options=["Option A", "Option B", "Option C"]):
        self.options = options
        
    def _get_distribution(self, choices_series):
        counts = choices_series.value_counts().reindex(self.options, fill_value=0)
        probabilities = counts / counts.sum()
        return probabilities.values

    def evaluate_tier1_jsd(self, human_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        human_dist = self._get_distribution(human_df['human_choice'])
        synthetic_dist = self._get_distribution(synthetic_df['synthetic_choice'])
        
        js_distance = distance.jensenshannon(human_dist, synthetic_dist)
        js_divergence = js_distance ** 2
        
        print("\nTier 1 validation: aggregate alignment")
        print(f"Human market share:     A: {human_dist[0]*100:.1f}% | B: {human_dist[1]*100:.1f}% | C: {human_dist[2]*100:.1f}%")
        print(f"Synthetic market share: A: {synthetic_dist[0]*100:.1f}% | B: {synthetic_dist[1]*100:.1f}% | C: {synthetic_dist[2]*100:.1f}%")
        print(f"Jensen-Shannon divergence (JSD): {js_divergence:.4f}\n")
        
        return js_divergence

    def evaluate_tier2_subgroups(self, human_df: pd.DataFrame, synthetic_df: pd.DataFrame, feature: str):
        print(f"\nTier 2 validation: sub-group alignment ({feature.upper()})")
        unique_groups = human_df[feature].unique()
        
        group_scores = {}
        for group in unique_groups:
            h_sub = human_df[human_df[feature] == group]
            s_sub = synthetic_df[synthetic_df[feature] == group]
            
            if len(h_sub) == 0 or len(s_sub) == 0:
                continue
                
            h_dist = self._get_distribution(h_sub['human_choice'])
            s_dist = self._get_distribution(s_sub['synthetic_choice'])
            
            js_distance = distance.jensenshannon(h_dist, s_dist)
            js_divergence = js_distance ** 2
            group_scores[group] = js_divergence
            
            print(f"[{group}] JSD: {js_divergence:.4f}")
            print(f"  Human:     A: {h_dist[0]*100:.1f}% | B: {h_dist[1]*100:.1f}% | C: {h_dist[2]*100:.1f}%")
            print(f"  Synthetic: A: {s_dist[0]*100:.1f}% | B: {s_dist[1]*100:.1f}% | C: {s_dist[2]*100:.1f}%\n")
            
        avg_jsd = np.mean(list(group_scores.values()))
        print(f"Average sub-group JSD for {feature}: {avg_jsd:.4f}\n")
        
        return group_scores