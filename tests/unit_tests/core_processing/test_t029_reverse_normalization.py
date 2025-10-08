#!/usr/bin/env python3
"""
T029: Test Reverse Normalization
Verifiera att reverse normalization ger ursprungliga värden
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår master POC unified normalization modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_unified_normalization", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_unified_normalization.py')
)
master_poc_unified_normalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_unified_normalization)

# Använd modulen
MasterPOCUnifiedNormalizer = master_poc_unified_normalization.MasterPOCUnifiedNormalizer
create_master_poc_unified_normalizer = master_poc_unified_normalization.create_master_poc_unified_normalizer

class TestT029ReverseNormalization(unittest.TestCase):
    """T029: Test Reverse Normalization"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t029_reverse_normalization_basic(self):
        """
        T029: Test Reverse Normalization
        Verifiera att reverse normalization ger ursprungliga värden
        """
        # Arrange
        # Testa reverse normalization med olika features
        test_cases = [
            # (feature_name, original_values, description)
            ('HR', np.array([20, 110, 200]), "Heart Rate reverse normalization"),
            ('SPO2', np.array([70, 85, 100]), "Oxygen Saturation reverse normalization"),
            ('ETCO2', np.array([2.0, 5.0, 8.0]), "End-Tidal CO2 reverse normalization"),
            ('age', np.array([0, 60, 120]), "Age reverse normalization"),
            ('height', np.array([100, 165, 230]), "Height reverse normalization"),
            ('sex', np.array([-1, 0, 1]), "Sex reverse normalization"),
        ]
        
        # Act & Assert
        for feature_name, original_values, description in test_cases:
            # Normalisera
            normalized = self.normalizer.normalize_feature(original_values, feature_name)
            
            # Reverse normalisera
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, original_values, decimal=3,
                                                err_msg=f"{description}: Reverse normalization ska ge ursprungliga värden")
            
            # Verifiera att längden bevaras
            self.assertEqual(len(denormalized), len(original_values),
                           f"{description}: Längd ska bevaras")
        
        print("✅ T029 PASSED: Basic reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_vital_signs(self):
        """
        Verifiera reverse normalization för vital signs
        """
        # Arrange
        vital_signs = ['HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS']
        
        # Act & Assert
        for feature_name in vital_signs:
            # Skapa test värden inom clinical range
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                                err_msg=f"{feature_name}: Reverse normalization ska fungera korrekt")
        
        print("✅ T029 PASSED: Vital signs reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_drug_infusions(self):
        """
        Verifiera reverse normalization för drug infusions
        """
        # Arrange
        drug_infusions = ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF']
        
        # Act & Assert
        for feature_name in drug_infusions:
            # Skapa test värden inom clinical range
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            test_values = np.array([min_clinical, max_clinical / 2, max_clinical])
            
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                                err_msg=f"{feature_name}: Reverse normalization ska fungera korrekt")
        
        print("✅ T029 PASSED: Drug infusions reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_ventilator_settings(self):
        """
        Verifiera reverse normalization för ventilator settings
        """
        # Arrange
        ventilator_settings = ['TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev']
        
        # Act & Assert
        for feature_name in ventilator_settings:
            # Skapa test värden inom clinical range
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                                err_msg=f"{feature_name}: Reverse normalization ska fungera korrekt")
        
        print("✅ T029 PASSED: Ventilator settings reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_static_features(self):
        """
        Verifiera reverse normalization för static features
        """
        # Arrange
        static_features = ['age', 'sex', 'height', 'weight', 'bmi', 'asa']
        
        # Act & Assert
        for feature_name in static_features:
            # Skapa test värden inom clinical range
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                                err_msg=f"{feature_name}: Reverse normalization ska fungera korrekt")
        
        print("✅ T029 PASSED: Static features reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_pandas_series(self):
        """
        Verifiera reverse normalization med pandas Series
        """
        # Arrange
        test_series = pd.Series([20, 110, 200], name='HR')
        
        # Act
        normalized = self.normalizer.normalize_feature(test_series, 'HR')
        denormalized = self.normalizer.denormalize_feature(normalized, 'HR')
        
        # Assert
        # Verifiera att reverse normalization fungerar korrekt
        np.testing.assert_array_almost_equal(denormalized, test_series.values, decimal=3,
                                            err_msg="Pandas Series reverse normalization ska fungera korrekt")
        
        # Verifiera att längden bevaras
        self.assertEqual(len(denormalized), len(test_series),
                       "Pandas Series längd ska bevaras")
        
        print("✅ T029 PASSED: Pandas Series reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_dataframe(self):
        """
        Verifiera reverse normalization med DataFrame
        """
        # Arrange
        df = pd.DataFrame({
            'HR': [20, 110, 200],
            'SPO2': [70, 85, 100],
            'age': [25, 60, 95],
            'height': [150, 175, 200]
        })
        
        # Act
        normalized_df = self.normalizer.normalize_dataframe(df)
        denormalized_df = self.normalizer.denormalize_dataframe(normalized_df)
        
        # Assert
        # Verifiera att reverse normalization fungerar korrekt för alla kolumner
        for column in df.columns:
            np.testing.assert_array_almost_equal(denormalized_df[column].values, df[column].values, decimal=3,
                                                err_msg=f"{column}: DataFrame reverse normalization ska fungera korrekt")
        
        # Verifiera att DataFrame struktur bevaras
        self.assertEqual(denormalized_df.shape, df.shape,
                       "DataFrame shape ska bevaras")
        self.assertEqual(list(denormalized_df.columns), list(df.columns),
                       "DataFrame kolumner ska bevaras")
        
        print("✅ T029 PASSED: DataFrame reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_edge_cases(self):
        """
        Verifiera reverse normalization med edge cases
        """
        # Arrange
        edge_cases = [
            # (feature_name, values, description)
            ('HR', np.array([20, 200]), "Heart Rate edge values"),
            ('SPO2', np.array([70, 100]), "Oxygen Saturation edge values"),
            ('ETCO2', np.array([2.0, 8.0]), "End-Tidal CO2 edge values"),
            ('age', np.array([0, 120]), "Age edge values"),
            ('sex', np.array([-1, 1]), "Sex edge values"),
        ]
        
        # Act & Assert
        for feature_name, values, description in edge_cases:
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, values, decimal=3,
                                                err_msg=f"{description}: Edge case reverse normalization ska fungera korrekt")
        
        print("✅ T029 PASSED: Edge cases reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_precision(self):
        """
        Verifiera precision av reverse normalization
        """
        # Arrange
        # Testa med värden som ska ge exakta resultat
        precision_cases = [
            ('HR', np.array([20, 110, 200])),      # Min, midpoint, max
            ('SPO2', np.array([70, 85, 100])),     # Min, midpoint, max
            ('ETCO2', np.array([2.0, 5.0, 8.0])), # Min, midpoint, max
            ('age', np.array([0, 60, 120])),       # Min, midpoint, max
            ('sex', np.array([-1, 0, 1])),         # Min, midpoint, max
        ]
        
        # Act & Assert
        for feature_name, values in precision_cases:
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att precision är hög (decimal=6 för högre precision)
            np.testing.assert_array_almost_equal(denormalized, values, decimal=6,
                                                err_msg=f"{feature_name}: Reverse normalization ska ha hög precision")
        
        print("✅ T029 PASSED: Reverse normalization precision fungerar korrekt")
    
    def test_t029_reverse_normalization_round_trip(self):
        """
        Verifiera round-trip normalization (normalize -> denormalize -> normalize)
        """
        # Arrange
        test_values = np.array([20, 110, 200])
        
        # Act
        # Round-trip: normalize -> denormalize -> normalize
        normalized1 = self.normalizer.normalize_feature(test_values, 'HR')
        denormalized = self.normalizer.denormalize_feature(normalized1, 'HR')
        normalized2 = self.normalizer.normalize_feature(denormalized, 'HR')
        
        # Assert
        # Verifiera att round-trip normalization fungerar korrekt
        np.testing.assert_array_almost_equal(normalized1, normalized2, decimal=6,
                                            err_msg="Round-trip normalization ska fungera korrekt")
        
        # Verifiera att ursprungliga värden återställs
        np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                            err_msg="Round-trip ska återställa ursprungliga värden")
        
        print("✅ T029 PASSED: Round-trip reverse normalization fungerar korrekt")
    
    def test_t029_reverse_normalization_comprehensive(self):
        """
        Omfattande test av reverse normalization med alla features
        """
        # Arrange
        all_features = self.normalizer.get_all_features()
        
        # Act & Assert
        for feature_name in all_features:
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            
            # Testa med min, midpoint och max värden
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            
            # Normalisera och reverse normalisera
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            denormalized = self.normalizer.denormalize_feature(normalized, feature_name)
            
            # Verifiera att reverse normalization fungerar korrekt
            np.testing.assert_array_almost_equal(denormalized, test_values, decimal=3,
                                                err_msg=f"{feature_name}: Comprehensive reverse normalization ska fungera korrekt")
            
            # Verifiera att längden bevaras
            self.assertEqual(len(denormalized), len(test_values),
                           f"{feature_name}: Längd ska bevaras")
        
        print("✅ T029 PASSED: Comprehensive reverse normalization test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
