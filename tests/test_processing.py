import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.processing import create_pools, compile_risk_set, create_ambiguous_test_set

class TestProcessing(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        # Risk Density: High = Looks like Risk. Low = Looks like Safe.
        self.df = pd.DataFrame({
            'dataset_type': ['mental_health', 'mental_health', 'safe', 'safe'],
            'risk_density': [0.9, 0.1, 0.1, 0.9], 
            # 0.9 Risk -> Easy Risk. 0.1 Risk -> Hard Risk (looks safe).
            # 0.1 Safe -> Easy Safe. 0.9 Safe -> Hard Safe (looks risk).
            'text': ['risk_easy', 'risk_hard', 'safe_easy', 'safe_hard'],
            'embedding_vec': [np.zeros(5) for _ in range(4)]
        })

    def test_create_pools(self):
        risk, safe = create_pools(self.df)
        self.assertEqual(len(risk), 2)
        self.assertEqual(len(safe), 2)
        self.assertTrue(all(risk['label'] == 1))
        self.assertTrue(all(safe['label'] == 0))

    def test_compile_risk_set_oversampling(self):
        risk, _ = create_pools(self.df)
        # compile_risk_set oversamples. Input 2.
        # factor 2.0 -> target 4.
        oversampled = compile_risk_set(risk, oversample_factor=2.0)
        self.assertEqual(len(oversampled), 4)
        
        # Verify it runs without error and returns DataFrame
        self.assertIsInstance(oversampled, pd.DataFrame)

    def test_create_ambiguous_test_set(self):
        risk, safe = create_pools(self.df)
        
        # Create a set where we request fewer than available to test sorting/selection
        # In this dummy set, all have 0.4 distance from 0.5 (0.1 and 0.9).
        # Let's add a clearer ambiguous case.
        
        df_ambiguous = pd.DataFrame({
            'dataset_type': ['mental_health', 'safe'],
            'risk_density': [0.45, 0.55], # Very ambiguous
            'text': ['ambig_risk', 'ambig_safe'],
            'embedding_vec': [np.zeros(5) for _ in range(2)]
        })
        
        risk_amb, safe_amb = create_pools(df_ambiguous)
        
        # Combine with original
        risk = pd.concat([risk, risk_amb])
        safe = pd.concat([safe, safe_amb])
        
        # Now we have 3 risk, 3 safe.
        # Most ambiguous are 0.45 and 0.55 (dist 0.05).
        # Others are 0.1/0.9 (dist 0.4).
        
        ambiguous_set = create_ambiguous_test_set(risk, safe, size=2)
        
        self.assertEqual(len(ambiguous_set), 2)
        # Should contain the two new ones
        texts = ambiguous_set['text'].values
        self.assertIn('ambig_risk', texts)
        self.assertIn('ambig_safe', texts)

if __name__ == '__main__':
    unittest.main()

