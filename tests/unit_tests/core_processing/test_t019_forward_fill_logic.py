#!/usr/bin/env python3
"""
T019: Test Forward Fill Logic
Verifiera forward fill för isolerade NaN-värden
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

class TestT019ForwardFillLogic(unittest.TestCase):
    """T019: Test Forward Fill Logic"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t019_forward_fill_isolated_nans(self):
        """
        T019: Test Forward Fill Logic
        Verifiera forward fill för isolerade NaN-värden
        """
        # Arrange
        # Skapa test data med isolerade NaN (NaN mellan giltiga värden)
        test_data = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean (3.5 = (1+3+4+6)/4)
        expected = pd.Series([1.0, 3.5, 3.0, 4.0, 3.5, 6.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Isolerade NaN ska fyllas med mean av giltiga värden")
        
        print("✅ T019 PASSED: Forward fill för isolerade NaN fungerar korrekt")
    
    def test_t019_forward_fill_consecutive_nans(self):
        """
        Verifiera forward fill för konsekutiva NaN-värden
        """
        # Arrange
        # Skapa test data med konsekutiva NaN
        test_data = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att isolerade NaN fylls med mean, konsekutiva NaN fylls med forward fill
        expected = pd.Series([1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 8.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Konsekutiva NaN ska fyllas med forward fill")
        
        print("✅ T019 PASSED: Forward fill för konsekutiva NaN fungerar korrekt")
    
    def test_t019_forward_fill_initial_nans(self):
        """
        Verifiera forward fill för initiala NaN-värden
        """
        # Arrange
        # Skapa test data med initiala NaN
        test_data = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att initiala NaN fylls med backward fill (3.0)
        expected = pd.Series([3.0, 3.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Initiala NaN ska fyllas med backward fill")
        
        print("✅ T019 PASSED: Forward fill för initiala NaN fungerar korrekt")
    
    def test_t019_forward_fill_final_nans(self):
        """
        Verifiera forward fill för finala NaN-värden
        """
        # Arrange
        # Skapa test data med finala NaN
        test_data = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att finala NaN fylls med forward fill (3.0)
        expected = pd.Series([1.0, 2.0, 3.0, 3.0, 3.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Finala NaN ska fyllas med forward fill")
        
        print("✅ T019 PASSED: Forward fill för finala NaN fungerar korrekt")
    
    def test_t019_forward_fill_mixed_scenarios(self):
        """
        Verifiera forward fill med blandade scenarier
        """
        # Arrange
        # Skapa test data med blandade NaN-scenarier
        test_data = pd.Series([np.nan, 2.0, np.nan, np.nan, 5.0, np.nan, 7.0, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att olika NaN-typer hanteras korrekt
        expected = pd.Series([2.0, 2.0, 2.0, 2.0, 5.0, 4.666666666666667, 7.0, 7.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Blandade NaN-scenarier ska hanteras korrekt")
        
        print("✅ T019 PASSED: Forward fill med blandade scenarier fungerar korrekt")
    
    def test_t019_forward_fill_no_nans(self):
        """
        Verifiera forward fill när det inte finns några NaN
        """
        # Arrange
        test_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att data inte ändras när det inte finns NaN
        pd.testing.assert_series_equal(result, test_data,
                                     "Data ska inte ändras när det inte finns NaN")
        
        print("✅ T019 PASSED: Forward fill utan NaN fungerar korrekt")
    
    def test_t019_forward_fill_all_nans(self):
        """
        Verifiera forward fill när alla värden är NaN
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att alla NaN förblir NaN (ingen mean att beräkna)
        self.assertTrue(result.isna().all(), "Alla NaN ska förbli NaN när det inte finns giltiga värden")
        
        print("✅ T019 PASSED: Forward fill med alla NaN fungerar korrekt")
    
    def test_t019_forward_fill_single_value(self):
        """
        Verifiera forward fill med enbart ett giltigt värde
        """
        # Arrange
        test_data = pd.Series([np.nan, np.nan, 5.0, np.nan, np.nan])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data)
        
        # Assert
        # Verifiera att alla NaN fylls med det enda giltiga värdet
        expected = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Alla NaN ska fyllas med det enda giltiga värdet")
        
        print("✅ T019 PASSED: Forward fill med enbart ett giltigt värde fungerar korrekt")
    
    def test_t019_forward_fill_max_consecutive_limit(self):
        """
        Verifiera forward fill med max_consecutive_nans limit
        """
        # Arrange
        test_data = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])
        
        # Act
        result = self.imputer.smart_forward_fill(test_data, max_consecutive_nans=2)
        
        # Assert
        # Verifiera att endast 2 konsekutiva NaN fylls med forward fill, resten med backward fill
        expected = pd.Series([1.0, 1.0, 1.0, 6.0, 6.0, 6.0])
        pd.testing.assert_series_equal(result, expected,
                                     "Max consecutive limit ska begränsa forward fill, resten fylls med backward fill")
        
        print("✅ T019 PASSED: Forward fill med max consecutive limit fungerar korrekt")
    
    def test_t019_forward_fill_edge_cases(self):
        """
        Verifiera forward fill med edge cases
        """
        # Arrange
        test_cases = [
            # (data, expected, description)
            (pd.Series([np.nan]), pd.Series([np.nan]), "Single NaN"),
            (pd.Series([1.0]), pd.Series([1.0]), "Single value"),
            (pd.Series([]), pd.Series([]), "Empty series"),
            (pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 2.0, 3.0]), "Isolated NaN"),
        ]
        
        # Act & Assert
        for test_data, expected, description in test_cases:
            result = self.imputer.smart_forward_fill(test_data)
            
            if test_data.empty:
                self.assertTrue(result.empty, f"{description}: Empty series ska förbli empty")
            else:
                pd.testing.assert_series_equal(result, expected,
                                             f"{description}: Expected {expected.tolist()}, got {result.tolist()}")
        
        print("✅ T019 PASSED: Forward fill edge cases fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
