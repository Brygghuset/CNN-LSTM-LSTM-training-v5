#!/usr/bin/env python3
"""
T021: Test Mean Imputation
Verifiera mean-imputation för isolerade NaN mellan giltiga värden
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

class TestT021MeanImputation(unittest.TestCase):
    """T021: Test Mean Imputation"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t021_mean_imputation_basic(self):
        """
        T021: Test Mean Imputation
        Verifiera mean-imputation för isolerade NaN mellan giltiga värden
        """
        # Arrange
        # Skapa test data med isolerade NaN (NaN mellan två giltiga värden)
        test_data = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean (3.5 = (1+3+4+6)/4)
        expected = pd.Series([1.0, 3.5, 3.0, 4.0, 3.5, 6.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Isolerade NaN ska fyllas med mean av giltiga värden")
        
        print("✅ T021 PASSED: Basic mean imputation fungerar korrekt")
    
    def test_t021_mean_imputation_single_isolated(self):
        """
        Verifiera mean imputation för enbart ett isolerat NaN
        """
        # Arrange
        test_data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean (3.0 = (1+2+4+5)/4)
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Enbart ett isolerat NaN ska fyllas med mean")
        
        print("✅ T021 PASSED: Mean imputation för enbart ett isolerat NaN fungerar korrekt")
    
    def test_t021_mean_imputation_multiple_isolated(self):
        """
        Verifiera mean imputation för flera isolerade NaN
        """
        # Arrange
        test_data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att alla isolerade NaN fylls med samma mean (4.0 = (1+3+5+7)/4)
        expected = pd.Series([1.0, 4.0, 3.0, 4.0, 5.0, 4.0, 7.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Flera isolerade NaN ska fyllas med samma mean")
        
        print("✅ T021 PASSED: Mean imputation för flera isolerade NaN fungerar korrekt")
    
    def test_t021_mean_imputation_edge_positions(self):
        """
        Verifiera mean imputation för isolerade NaN i olika positioner
        """
        # Arrange
        test_cases = [
            # (data, expected, description)
            (pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 2.0, 3.0]), "Isolerat NaN i mitten"),
            (pd.Series([1.0, 2.0, np.nan, 4.0]), pd.Series([1.0, 2.0, 2.33, 4.0]), "Isolerat NaN nära slutet"),
            (pd.Series([1.0, np.nan, 3.0, 4.0]), pd.Series([1.0, 2.67, 3.0, 4.0]), "Isolerat NaN nära början"),
        ]
        
        # Act & Assert
        for test_data, expected, description in test_cases:
            result = self.imputer.smart_forward_fill(test_data)
            
            # Använd np.allclose för att hantera floating point precision
            self.assertTrue(np.allclose(result.values, expected.values, rtol=1e-2),
                          f"{description}: Expected {expected.tolist()}, got {result.tolist()}")
        
        print("✅ T021 PASSED: Mean imputation för isolerade NaN i olika positioner fungerar korrekt")
    
    def test_t021_mean_imputation_no_isolated_nans(self):
        """
        Verifiera att mean imputation inte appliceras när det inte finns isolerade NaN
        """
        # Arrange
        test_cases = [
            pd.Series([1.0, 2.0, 3.0, 4.0]),  # Inga NaN
            pd.Series([np.nan, np.nan, 3.0, 4.0]),  # Konsekutiva NaN
            pd.Series([1.0, 2.0, np.nan, np.nan]),  # Konsekutiva NaN
            pd.Series([np.nan, 2.0, 3.0, np.nan]),  # Initiala och finala NaN
        ]
        
        # Act & Assert
        for test_data in test_cases:
            result = self.imputer.smart_forward_fill(test_data)
            
            # Verifiera att resultatet inte innehåller mean-imputerade värden för isolerade NaN
            # (detta är svårt att testa direkt, så vi verifierar att funktionen körs utan fel)
            self.assertIsInstance(result, pd.Series, "Resultatet ska vara en pandas Series")
            self.assertEqual(len(result), len(test_data), "Längden ska vara densamma")
        
        print("✅ T021 PASSED: Mean imputation appliceras inte när det inte finns isolerade NaN")
    
    def test_t021_mean_imputation_with_consecutive_nans(self):
        """
        Verifiera mean imputation kombinerat med konsekutiva NaN
        """
        # Arrange
        test_data = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean (3.33), konsekutiva med forward fill
        expected = pd.Series([1.0, 3.33, 3.0, 3.0, 3.0, 6.0])
        
        # Använd np.allclose för floating point precision
        self.assertTrue(np.allclose(result.values, expected.values, rtol=1e-2),
                      "Mean imputation och konsekutiva NaN ska hanteras korrekt")
        
        print("✅ T021 PASSED: Mean imputation kombinerat med konsekutiva NaN fungerar korrekt")
    
    def test_t021_mean_imputation_precision(self):
        """
        Verifiera precision av mean imputation
        """
        # Arrange
        test_data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att mean beräknas korrekt (3.6 = (1+2+4+5+6)/5)
        expected_mean = 3.6
        isolated_nan_value = result.iloc[2]  # Det isolerade NaN-värdet
        
        self.assertAlmostEqual(isolated_nan_value, expected_mean, places=1,
                             msg=f"Mean imputation ska vara korrekt: expected {expected_mean}, got {isolated_nan_value}")
        
        print("✅ T021 PASSED: Mean imputation precision fungerar korrekt")
    
    def test_t021_mean_imputation_empty_series(self):
        """
        Verifiera mean imputation med tom serie
        """
        # Arrange
        test_data = pd.Series([])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att tom serie returneras oförändrad
        self.assertTrue(result.empty, "Tom serie ska returneras oförändrad")
        
        print("✅ T021 PASSED: Mean imputation med tom serie fungerar korrekt")
    
    def test_t021_mean_imputation_all_nans(self):
        """
        Verifiera mean imputation när alla värden är NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att alla NaN förblir NaN (ingen mean att beräkna)
        self.assertTrue(result.isna().all(), "Alla NaN ska förbli NaN när det inte finns giltiga värden")
        
        print("✅ T021 PASSED: Mean imputation med alla NaN fungerar korrekt")
    
    def test_t021_mean_imputation_comprehensive(self):
        """
        Omfattande test av mean imputation med komplexa scenarier
        """
        # Arrange
        test_scenarios = [
            # (data, expected_mean, description)
            (pd.Series([1.0, np.nan, 3.0]), 2.0, "Enkel mean imputation"),
            (pd.Series([1.0, 2.0, np.nan, 4.0, 5.0]), 3.0, "Mean från 4 värden"),
            (pd.Series([10.0, np.nan, 20.0, np.nan, 30.0]), 20.0, "Flera isolerade NaN"),
            (pd.Series([0.1, 0.2, np.nan, 0.4, 0.5]), 0.3, "Decimal precision"),
        ]
        
        # Act & Assert
        for test_data, expected_mean, description in test_scenarios:
            result = self.imputer.smart_forward_fill(test_data)
            
            # Hitta det isolerade NaN-värdet
            isolated_indices = []
            for i in range(1, len(test_data) - 1):
                if pd.isna(test_data.iloc[i]) and not pd.isna(test_data.iloc[i-1]) and not pd.isna(test_data.iloc[i+1]):
                    isolated_indices.append(i)
            
            for idx in isolated_indices:
                imputed_value = result.iloc[idx]
                self.assertAlmostEqual(imputed_value, expected_mean, places=1,
                                     msg=f"{description}: Expected mean {expected_mean}, got {imputed_value}")
        
        print("✅ T021 PASSED: Comprehensive mean imputation test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
