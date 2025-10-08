#!/usr/bin/env python3
"""
T018: Test Clinical Zeros
Verifiera att kliniska nollor appliceras korrekt per parameter
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår smart forward fill modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_smart_forward_fill", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_smart_forward_fill.py')
)
master_poc_smart_forward_fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_smart_forward_fill)

# Använd modulen
MasterPOCSmartForwardFill = master_poc_smart_forward_fill.MasterPOCSmartForwardFill
create_smart_forward_fill = master_poc_smart_forward_fill.create_smart_forward_fill

class TestT018ClinicalZeros(unittest.TestCase):
    """T018: Test Clinical Zeros"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t018_clinical_zeros_basic(self):
        """
        T018: Test Clinical Zeros
        Verifiera att kliniska nollor appliceras korrekt per parameter
        """
        # Arrange
        # Skapa test data med NaN värden
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Expected clinical zeros enligt Master POC spec
        expected_clinical_zeros = {
            'ETCO2': 0.0,
            'BIS': 0.0,
            'Propofol_INF': 0.0,
            'Remifentanil_INF': 0.0,
            'Noradrenalin_INF': 0.0,
            'TV': 0.0,
            'PEEP': 0.0,
            'FIO2': 0.0,
            'RR': 0.0,
            'etSEV': 0.0,
            'inSev': 0.0,
        }
        
        # Act & Assert
        for feature_name, expected_zero in expected_clinical_zeros.items():
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att alla NaN ersatts med klinisk nolla
            self.assertTrue(result.notna().all(), f"{feature_name} ska ha inga NaN efter clinical zero")
            self.assertTrue((result == expected_zero).all(), 
                           f"{feature_name} ska ha klinisk nolla {expected_zero}, fick {result.iloc[0]}")
        
        print("✅ T018 PASSED: Basic clinical zeros fungerar korrekt")
    
    def test_t018_clinical_zeros_mixed_data(self):
        """
        Verifiera clinical zeros med blandad data (vissa giltiga värden, vissa NaN)
        """
        # Arrange
        test_data = pd.Series([1.5, np.nan, 2.0, np.nan, 3.5])
        
        # Act & Assert
        for feature_name in ['ETCO2', 'BIS', 'Propofol_INF', 'TV', 'PEEP']:
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att NaN ersatts med 0.0, men giltiga värden behålls
            expected = pd.Series([1.5, 0.0, 2.0, 0.0, 3.5])
            pd.testing.assert_series_equal(result, expected, 
                                         f"{feature_name} ska ersätta NaN med 0.0 men behålla giltiga värden")
        
        print("✅ T018 PASSED: Clinical zeros med blandad data fungerar korrekt")
    
    def test_t018_clinical_zeros_no_nan(self):
        """
        Verifiera clinical zeros när det inte finns några NaN
        """
        # Arrange
        test_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Act & Assert
        for feature_name in ['ETCO2', 'BIS', 'Propofol_INF']:
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att data inte ändras när det inte finns NaN
            pd.testing.assert_series_equal(result, test_data,
                                         f"{feature_name} ska inte ändras när det inte finns NaN")
        
        print("✅ T018 PASSED: Clinical zeros utan NaN fungerar korrekt")
    
    def test_t018_clinical_zeros_empty_series(self):
        """
        Verifiera clinical zeros med tom serie
        """
        # Arrange
        test_data = pd.Series([])
        
        # Act & Assert
        for feature_name in ['ETCO2', 'BIS', 'Propofol_INF']:
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att tom serie returneras oförändrad
            self.assertTrue(result.empty, f"{feature_name} ska returnera tom serie")
        
        print("✅ T018 PASSED: Clinical zeros med tom serie fungerar korrekt")
    
    def test_t018_clinical_zeros_non_clinical_feature(self):
        """
        Verifiera att non-clinical features inte påverkas
        """
        # Arrange
        test_data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        
        # Act
        result = self.imputer.apply_clinical_zeros(test_data, 'HR')  # HR har inte clinical zero
        
        # Assert
        # Verifiera att HR inte påverkas (ska ha NaN kvar)
        self.assertTrue(result.isna().any(), "HR ska inte påverkas av clinical zeros")
        pd.testing.assert_series_equal(result, test_data,
                                       "Non-clinical features ska inte påverkas")
        
        print("✅ T018 PASSED: Non-clinical features påverkas inte av clinical zeros")
    
    def test_t018_clinical_zeros_all_nan_series(self):
        """
        Verifiera clinical zeros med serie som bara innehåller NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Act & Assert
        for feature_name in ['ETCO2', 'BIS', 'Propofol_INF', 'TV', 'PEEP']:
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att alla NaN ersatts med 0.0
            expected = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
            pd.testing.assert_series_equal(result, expected,
                                         f"{feature_name} ska ersätta alla NaN med 0.0")
        
        print("✅ T018 PASSED: Clinical zeros med alla NaN fungerar korrekt")
    
    def test_t018_clinical_zeros_edge_values(self):
        """
        Verifiera clinical zeros med edge values (0.0, negativa värden)
        """
        # Arrange
        test_data = pd.Series([0.0, -1.0, np.nan, 0.0, np.nan])
        
        # Act & Assert
        for feature_name in ['ETCO2', 'BIS', 'Propofol_INF']:
            result = self.imputer.apply_clinical_zeros(test_data, feature_name)
            
            # Verifiera att NaN ersatts med 0.0, men andra värden behålls
            expected = pd.Series([0.0, -1.0, 0.0, 0.0, 0.0])
            pd.testing.assert_series_equal(result, expected,
                                         f"{feature_name} ska ersätta NaN med 0.0 men behålla andra värden")
        
        print("✅ T018 PASSED: Clinical zeros med edge values fungerar korrekt")
    
    def test_t018_clinical_zeros_verification(self):
        """
        Verifiera att clinical zeros är korrekt definierade enligt Master POC spec
        """
        # Arrange
        expected_clinical_zeros = {
            'ETCO2': 0.0,
            'BIS': 0.0,
            'Propofol_INF': 0.0,
            'Remifentanil_INF': 0.0,
            'Noradrenalin_INF': 0.0,
            'TV': 0.0,
            'PEEP': 0.0,
            'FIO2': 0.0,
            'RR': 0.0,
            'etSEV': 0.0,
            'inSev': 0.0,
        }
        
        # Act & Assert
        for feature_name, expected_zero in expected_clinical_zeros.items():
            # Verifiera att feature finns i clinical_zeros
            self.assertIn(feature_name, self.imputer.clinical_zeros,
                         f"{feature_name} ska finnas i clinical_zeros")
            
            # Verifiera att värdet är korrekt
            actual_zero = self.imputer.clinical_zeros[feature_name]
            self.assertEqual(actual_zero, expected_zero,
                           f"{feature_name} ska ha clinical zero {expected_zero}, fick {actual_zero}")
        
        print("✅ T018 PASSED: Clinical zeros är korrekt definierade enligt Master POC spec")
    
    def test_t018_clinical_zeros_comprehensive(self):
        """
        Omfattande test av clinical zeros med olika scenarier
        """
        # Arrange
        test_scenarios = [
            # (data, expected_result, description)
            (pd.Series([np.nan]), pd.Series([0.0]), "Single NaN"),
            (pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 0.0, 3.0]), "Mixed valid and NaN"),
            (pd.Series([0.0, np.nan, 0.0]), pd.Series([0.0, 0.0, 0.0]), "Already zero values"),
            (pd.Series([np.nan, np.nan]), pd.Series([0.0, 0.0]), "Multiple consecutive NaN"),
        ]
        
        # Act & Assert
        for test_data, expected_result, description in test_scenarios:
            for feature_name in ['ETCO2', 'BIS', 'Propofol_INF']:
                result = self.imputer.apply_clinical_zeros(test_data, feature_name)
                
                pd.testing.assert_series_equal(result, expected_result,
                                             f"{feature_name} - {description}: Expected {expected_result.tolist()}, got {result.tolist()}")
        
        print("✅ T018 PASSED: Comprehensive clinical zeros test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
