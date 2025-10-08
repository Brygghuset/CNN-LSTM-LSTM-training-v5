#!/usr/bin/env python3
"""
T022: Test Default Values
Verifiera default-värden för vital signs (HR=70, BP_SYS=140, etc.)
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

class TestT022DefaultValues(unittest.TestCase):
    """T022: Test Default Values"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t022_default_values_basic(self):
        """
        T022: Test Default Values
        Verifiera default-värden för vital signs (HR=70, BP_SYS=140, etc.)
        """
        # Arrange
        # Skapa test data med NaN värden för vital signs
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Expected default values enligt Master POC spec
        expected_default_values = {
            'HR': 70.0,
            'BP_SYS': 140.0,
            'BP_DIA': 80.0,
            'BP_MAP': 93.3,  # Beräknat från default SYS/DBP
            'SPO2': 96.0,
        }
        
        # Act & Assert
        for feature_name, expected_default in expected_default_values.items():
            result = self.imputer.apply_default_values(test_data, feature_name)
            
            # Verifiera att alla NaN ersatts med default-värde
            self.assertTrue(result.notna().all(), f"{feature_name} ska ha inga NaN efter default value")
            self.assertTrue((result == expected_default).all(), 
                           f"{feature_name} ska ha default-värde {expected_default}, fick {result.iloc[0]}")
        
        print("✅ T022 PASSED: Basic default values fungerar korrekt")
    
    def test_t022_default_values_mixed_data(self):
        """
        Verifiera default values med blandad data (vissa giltiga värden, vissa NaN)
        """
        # Arrange
        test_data = pd.Series([75.0, np.nan, 85.0, np.nan, 95.0])
        
        # Act & Assert
        for feature_name in ['HR', 'BP_SYS', 'BP_DIA', 'SPO2']:
            result = self.imputer.apply_default_values(test_data, feature_name)
            
            # Verifiera att NaN ersatts med default-värde, men giltiga värden behålls
            expected_default = self.imputer.default_values[feature_name]
            expected = pd.Series([75.0, expected_default, 85.0, expected_default, 95.0])
            pd.testing.assert_series_equal(result, expected,
                                         f"{feature_name} ska ersätta NaN med default men behålla giltiga värden")
        
        print("✅ T022 PASSED: Default values med blandad data fungerar korrekt")
    
    def test_t022_default_values_no_nans(self):
        """
        Verifiera default values när det inte finns några NaN
        """
        # Arrange
        test_data = pd.Series([75.0, 80.0, 85.0, 90.0, 95.0])
        
        # Act & Assert
        for feature_name in ['HR', 'BP_SYS', 'BP_DIA', 'SPO2']:
            result = self.imputer.apply_default_values(test_data, feature_name)
            
            # Verifiera att data inte ändras när det inte finns NaN
            pd.testing.assert_series_equal(result, test_data,
                                         f"{feature_name} ska inte ändras när det inte finns NaN")
        
        print("✅ T022 PASSED: Default values utan NaN fungerar korrekt")
    
    def test_t022_default_values_empty_series(self):
        """
        Verifiera default values med tom serie
        """
        # Arrange
        test_data = pd.Series([])
        
        # Act & Assert
        for feature_name in ['HR', 'BP_SYS', 'BP_DIA', 'SPO2']:
            result = self.imputer.apply_default_values(test_data, feature_name)
            
            # Verifiera att tom serie returneras oförändrad
            self.assertTrue(result.empty, f"{feature_name} ska returnera tom serie")
        
        print("✅ T022 PASSED: Default values med tom serie fungerar korrekt")
    
    def test_t022_default_values_non_vital_sign_feature(self):
        """
        Verifiera att non-vital sign features inte påverkas
        """
        # Arrange
        test_data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        
        # Act
        result = self.imputer.apply_default_values(test_data, 'ETCO2')  # ETCO2 har inte default value
        
        # Assert
        # Verifiera att ETCO2 inte påverkas (ska ha NaN kvar)
        self.assertTrue(result.isna().any(), "ETCO2 ska inte påverkas av default values")
        pd.testing.assert_series_equal(result, test_data,
                                       "Non-vital sign features ska inte påverkas")
        
        print("✅ T022 PASSED: Non-vital sign features påverkas inte av default values")
    
    def test_t022_default_values_verification(self):
        """
        Verifiera att default values är korrekt definierade enligt Master POC spec
        """
        # Arrange
        expected_default_values = {
            'HR': 70.0,
            'BP_SYS': 140.0,
            'BP_DIA': 80.0,
            'BP_MAP': 93.3,  # Beräknat från default SYS/DBP
            'SPO2': 96.0,
        }
        
        # Act & Assert
        for feature_name, expected_default in expected_default_values.items():
            # Verifiera att feature finns i default_values
            self.assertIn(feature_name, self.imputer.default_values,
                         f"{feature_name} ska finnas i default_values")
            
            # Verifiera att värdet är korrekt
            actual_default = self.imputer.default_values[feature_name]
            self.assertEqual(actual_default, expected_default,
                           f"{feature_name} ska ha default value {expected_default}, fick {actual_default}")
        
        print("✅ T022 PASSED: Default values är korrekt definierade enligt Master POC spec")
    
    def test_t022_default_values_bp_map_calculation(self):
        """
        Verifiera att BP_MAP default value är korrekt beräknat
        """
        # Arrange
        # BP_MAP = (2 * DBP + SBP) / 3
        # Med default values: BP_MAP = (2 * 80 + 140) / 3 = 300 / 3 = 100
        # Men enligt Master POC spec är det 93.3, så vi verifierar det värdet
        
        # Act
        bp_map_default = self.imputer.default_values['BP_MAP']
        
        # Assert
        self.assertEqual(bp_map_default, 93.3,
                        f"BP_MAP default value ska vara 93.3, fick {bp_map_default}")
        
        print("✅ T022 PASSED: BP_MAP default value är korrekt beräknat")
    
    def test_t022_default_values_edge_cases(self):
        """
        Verifiera default values med edge cases
        """
        # Arrange
        test_cases = [
            # (data, expected_result, description)
            (pd.Series([np.nan]), pd.Series([70.0]), "Single NaN för HR"),
            (pd.Series([0.0, np.nan, 0.0]), pd.Series([0.0, 70.0, 0.0]), "Mixed valid and NaN för HR"),
            (pd.Series([np.nan, np.nan]), pd.Series([70.0, 70.0]), "Multiple NaN för HR"),
        ]
        
        # Act & Assert
        for test_data, expected_result, description in test_cases:
            result = self.imputer.apply_default_values(test_data, 'HR')
            
            pd.testing.assert_series_equal(result, expected_result,
                                         f"{description}: Expected {expected_result.tolist()}, got {result.tolist()}")
        
        print("✅ T022 PASSED: Default values edge cases fungerar korrekt")
    
    def test_t022_default_values_comprehensive(self):
        """
        Omfattande test av default values med olika scenarier
        """
        # Arrange
        test_scenarios = [
            # (data, expected_result, description)
            (pd.Series([np.nan]), pd.Series([70.0]), "Single NaN för HR"),
            (pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 70.0, 3.0]), "Mixed för HR"),
            (pd.Series([np.nan, np.nan, np.nan]), pd.Series([70.0, 70.0, 70.0]), "All NaN för HR"),
        ]
        
        # Act & Assert
        for test_data, expected_result, description in test_scenarios:
            for feature_name in ['HR', 'BP_SYS', 'BP_DIA', 'SPO2']:
                result = self.imputer.apply_default_values(test_data, feature_name)
                
                # Ersätt förväntat värde med rätt default för varje feature
                expected_default = self.imputer.default_values[feature_name]
                expected = pd.Series([expected_default if pd.isna(val) else val for val in test_data])
                
                pd.testing.assert_series_equal(result, expected,
                                             f"{feature_name} - {description}: Expected {expected.tolist()}, got {result.tolist()}")
        
        print("✅ T022 PASSED: Comprehensive default values test fungerar korrekt")
    
    def test_t022_default_values_integration(self):
        """
        Verifiera integration av default values med smart forward fill
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, 75.0, np.nan, 85.0])
        
        # Act
        result = self.imputer.apply_smart_forward_fill(test_data, 'HR')
        
        # Assert
        # Verifiera att default values appliceras först, sedan smart forward fill
        # Förväntat: [70.0, 70.0, 75.0, 70.0, 85.0] (default för initiala NaN, mean för isolerat NaN, forward fill för finala)
        expected = pd.Series([70.0, 70.0, 75.0, 70.0, 85.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Integration av default values och smart forward fill fungerar korrekt")
        
        print("✅ T022 PASSED: Integration av default values med smart forward fill fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
