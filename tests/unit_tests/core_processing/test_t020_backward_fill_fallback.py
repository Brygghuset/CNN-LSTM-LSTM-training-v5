#!/usr/bin/env python3
"""
T020: Test Backward Fill Fallback
Verifiera backward fill som fallback för initiala NaN
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

class TestT020BackwardFillFallback(unittest.TestCase):
    """T020: Test Backward Fill Fallback"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t020_backward_fill_fallback_basic(self):
        """
        T020: Test Backward Fill Fallback
        Verifiera backward fill som fallback för initiala NaN
        """
        # Arrange
        # Skapa test data med initiala NaN som inte kan fyllas med forward fill
        test_data = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att initiala NaN fylls med backward fill (3.0)
        expected = pd.Series([3.0, 3.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Initiala NaN ska fyllas med backward fill som fallback")
        
        print("✅ T020 PASSED: Basic backward fill fallback fungerar korrekt")
    
    def test_t020_backward_fill_fallback_initial_only(self):
        """
        Verifiera backward fill fallback när endast initiala värden är NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att initiala NaN fylls med första giltiga värdet
        expected = pd.Series([3.0, 3.0, 3.0, 4.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Endast initiala NaN ska fyllas med backward fill")
        
        print("✅ T020 PASSED: Backward fill fallback för endast initiala NaN fungerar korrekt")
    
    def test_t020_backward_fill_fallback_mixed_initial_final(self):
        """
        Verifiera backward fill fallback med både initiala och finala NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, 3.0, 4.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att initiala NaN fylls med backward fill, finala med forward fill
        expected = pd.Series([3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Initiala NaN fylls med backward fill, finala med forward fill")
        
        print("✅ T020 PASSED: Backward fill fallback med blandade initiala och finala NaN fungerar korrekt")
    
    def test_t020_backward_fill_fallback_all_nans_except_one(self):
        """
        Verifiera backward fill fallback när endast ett värde är giltigt
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, np.nan, 5.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att alla NaN fylls med det enda giltiga värdet
        expected = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Alla NaN ska fyllas med det enda giltiga värdet")
        
        print("✅ T020 PASSED: Backward fill fallback med endast ett giltigt värde fungerar korrekt")
    
    def test_t020_backward_fill_fallback_no_initial_nans(self):
        """
        Verifiera att backward fill inte appliceras när det inte finns initiala NaN
        """
        # Arrange
        test_data = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att endast forward fill används
        expected = pd.Series([1.0, 2.0, 3.0, 3.0, 3.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Backward fill ska inte användas när det inte finns initiala NaN")
        
        print("✅ T020 PASSED: Backward fill fallback används inte när det inte behövs")
    
    def test_t020_backward_fill_fallback_edge_cases(self):
        """
        Verifiera backward fill fallback med edge cases
        """
        # Arrange
        test_cases = [
            # (data, expected, description)
            (pd.Series([np.nan, 2.0]), pd.Series([2.0, 2.0]), "Single initial NaN"),
            (pd.Series([np.nan, np.nan, 1.0]), pd.Series([1.0, 1.0, 1.0]), "Two initial NaN"),
            (pd.Series([np.nan]), pd.Series([np.nan]), "Single NaN only"),
            (pd.Series([1.0, np.nan]), pd.Series([1.0, 1.0]), "Single final NaN"),
        ]
        
        # Act & Assert
        for test_data, expected, description in test_cases:
            result = self.imputer.smart_forward_fill(test_data)
            
            if test_data.iloc[0] is pd.NA and len(test_data) == 1:
                # Special case: single NaN should remain NaN
                self.assertTrue(result.isna().all(), f"{description}: Single NaN ska förbli NaN")
            else:
                pd.testing.assert_series_equal(result, expected,
                                             f"{description}: Expected {expected.tolist()}, got {result.tolist()}")
        
        print("✅ T020 PASSED: Backward fill fallback edge cases fungerar korrekt")
    
    def test_t020_backward_fill_fallback_with_isolated_nans(self):
        """
        Verifiera backward fill fallback kombinerat med mean imputation för isolerade NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, 2.0, np.nan, 4.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean (3.0), initiala med backward fill, finala med forward fill
        expected = pd.Series([2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Kombination av mean imputation och backward fill fungerar korrekt")
        
        print("✅ T020 PASSED: Backward fill fallback med isolerade NaN fungerar korrekt")
    
    def test_t020_backward_fill_fallback_max_consecutive_limit(self):
        """
        Verifiera backward fill fallback med max_consecutive_nans limit
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, 5.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data, max_consecutive_nans=2)
        
        # Assert
        # Verifiera att backward fill används för initiala NaN även med limit
        expected = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Backward fill fallback fungerar även med max consecutive limit")
        
        print("✅ T020 PASSED: Backward fill fallback med max consecutive limit fungerar korrekt")
    
    def test_t020_backward_fill_fallback_comprehensive(self):
        """
        Omfattande test av backward fill fallback med komplexa scenarier
        """
        # Arrange
        test_scenarios = [
            # (data, expected, description)
            (pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0]), 
             pd.Series([1.0, 1.0, 1.0, 2.0, 3.0]), 
             "Initiala NaN med backward fill"),
            
            (pd.Series([1.0, 2.0, 3.0, np.nan, np.nan]), 
             pd.Series([1.0, 2.0, 3.0, 3.0, 3.0]), 
             "Finala NaN med forward fill"),
            
            (pd.Series([np.nan, 2.0, np.nan, 4.0, np.nan]), 
             pd.Series([2.0, 2.0, 3.0, 4.0, 4.0]), 
             "Blandade NaN med mean imputation och fills"),
            
            (pd.Series([np.nan, np.nan, np.nan, 1.0, np.nan, np.nan]), 
             pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
             "Alla NaN förutom ett"),
        ]
        
        # Act & Assert
        for test_data, expected, description in test_scenarios:
            result = self.imputer.smart_forward_fill(test_data)
            
            pd.testing.assert_series_equal(result, expected,
                                         f"{description}: Expected {expected.tolist()}, got {result.tolist()}")
        
        print("✅ T020 PASSED: Comprehensive backward fill fallback test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
