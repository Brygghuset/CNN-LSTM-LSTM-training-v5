#!/usr/bin/env python3
"""
T023: Test Imputation Edge Cases
Verifiera hantering av helt tomma serier
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

class TestT023ImputationEdgeCases(unittest.TestCase):
    """T023: Test Imputation Edge Cases"""
    
    def setUp(self):
        """Setup för varje test."""
        self.imputer = create_smart_forward_fill()
    
    def test_t023_imputation_edge_cases_empty_series(self):
        """
        T023: Test Imputation Edge Cases
        Verifiera hantering av helt tomma serier
        """
        # Arrange
        empty_series = pd.Series([])
        
        # Act & Assert
        # Testa alla imputation metoder med tom serie
        result_smart_fill = self.imputer.smart_forward_fill(empty_series)
        result_clinical_zeros = self.imputer.apply_clinical_zeros(empty_series, 'ETCO2')
        result_default_values = self.imputer.apply_default_values(empty_series, 'HR')
        
        # Verifiera att tomma serier returneras oförändrade
        self.assertTrue(result_smart_fill.empty, "Smart forward fill ska returnera tom serie")
        self.assertTrue(result_clinical_zeros.empty, "Clinical zeros ska returnera tom serie")
        self.assertTrue(result_default_values.empty, "Default values ska returnera tom serie")
        
        print("✅ T023 PASSED: Hantering av tomma serier fungerar korrekt")
    
    def test_t023_imputation_edge_cases_all_nans(self):
        """
        Verifiera hantering av serier som bara innehåller NaN
        """
        # Arrange
        all_nans_series = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Act & Assert
        # Testa smart forward fill med alla NaN
        result_smart_fill = self.imputer.smart_forward_fill(all_nans_series)
        
        # Verifiera att alla NaN förblir NaN (ingen mean att beräkna)
        self.assertTrue(result_smart_fill.isna().all(), "Alla NaN ska förbli NaN när det inte finns giltiga värden")
        
        # Testa clinical zeros med alla NaN
        result_clinical_zeros = self.imputer.apply_clinical_zeros(all_nans_series, 'ETCO2')
        expected_clinical_zeros = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result_clinical_zeros, expected_clinical_zeros,
                                     "Clinical zeros ska ersätta alla NaN med 0.0")
        
        # Testa default values med alla NaN
        result_default_values = self.imputer.apply_default_values(all_nans_series, 'HR')
        expected_default_values = pd.Series([70.0, 70.0, 70.0, 70.0, 70.0])
        pd.testing.assert_series_equal(result_default_values, expected_default_values,
                                     "Default values ska ersätta alla NaN med default-värde")
        
        print("✅ T023 PASSED: Hantering av serier med alla NaN fungerar korrekt")
    
    def test_t023_imputation_edge_cases_single_value(self):
        """
        Verifiera hantering av serier med enbart ett värde
        """
        # Arrange
        single_value_series = pd.Series([5.0])
        
        # Act & Assert
        # Testa smart forward fill med enbart ett värde
        result_smart_fill = self.imputer.smart_forward_fill(single_value_series)
        
        # Verifiera att enbart ett värde returneras oförändrat
        pd.testing.assert_series_equal(result_smart_fill, single_value_series,
                                     "Enbart ett värde ska returneras oförändrat")
        
        print("✅ T023 PASSED: Hantering av serier med enbart ett värde fungerar korrekt")
    
    def test_t023_imputation_edge_cases_single_nan(self):
        """
        Verifiera hantering av serier med enbart ett NaN
        """
        # Arrange
        single_nan_series = pd.Series([np.nan])
        
        # Act & Assert
        # Testa smart forward fill med enbart ett NaN
        result_smart_fill = self.imputer.smart_forward_fill(single_nan_series)
        
        # Verifiera att enbart ett NaN förblir NaN (ingen mean att beräkna)
        self.assertTrue(result_smart_fill.isna().all(), "Enbart ett NaN ska förbli NaN")
        
        # Testa clinical zeros med enbart ett NaN
        result_clinical_zeros = self.imputer.apply_clinical_zeros(single_nan_series, 'ETCO2')
        expected_clinical_zeros = pd.Series([0.0])
        pd.testing.assert_series_equal(result_clinical_zeros, expected_clinical_zeros,
                                     "Clinical zeros ska ersätta enbart ett NaN med 0.0")
        
        # Testa default values med enbart ett NaN
        result_default_values = self.imputer.apply_default_values(single_nan_series, 'HR')
        expected_default_values = pd.Series([70.0])
        pd.testing.assert_series_equal(result_default_values, expected_default_values,
                                     "Default values ska ersätta enbart ett NaN med default-värde")
        
        print("✅ T023 PASSED: Hantering av serier med enbart ett NaN fungerar korrekt")
    
    def test_t023_imputation_edge_cases_extreme_values(self):
        """
        Verifiera hantering av serier med extrema värden
        """
        # Arrange
        extreme_values_series = pd.Series([0.0, np.nan, 1000000.0, np.nan, -1000000.0])
        
        # Act & Assert
        # Testa smart forward fill med extrema värden
        result_smart_fill = self.imputer.smart_forward_fill(extreme_values_series)
        
        # Verifiera att extrema värden hanteras korrekt
        self.assertFalse(result_smart_fill.isna().any(), "Extrema värden ska hanteras korrekt")
        self.assertEqual(result_smart_fill.iloc[0], 0.0, "Första extrema värdet ska bevaras")
        self.assertEqual(result_smart_fill.iloc[2], 1000000.0, "Andra extrema värdet ska bevaras")
        self.assertEqual(result_smart_fill.iloc[4], -1000000.0, "Tredje extrema värdet ska bevaras")
        
        print("✅ T023 PASSED: Hantering av serier med extrema värden fungerar korrekt")
    
    def test_t023_imputation_edge_cases_mixed_types(self):
        """
        Verifiera hantering av serier med blandade datatyper
        """
        # Arrange
        # Skapa serie med olika numeriska typer
        mixed_types_series = pd.Series([1, np.nan, 3.0, np.nan, 5])
        
        # Act & Assert
        # Testa smart forward fill med blandade typer
        result_smart_fill = self.imputer.smart_forward_fill(mixed_types_series)
        
        # Verifiera att blandade typer hanteras korrekt
        self.assertFalse(result_smart_fill.isna().any(), "Blandade typer ska hanteras korrekt")
        self.assertEqual(len(result_smart_fill), len(mixed_types_series), "Längden ska bevaras")
        
        print("✅ T023 PASSED: Hantering av serier med blandade datatyper fungerar korrekt")
    
    def test_t023_imputation_edge_cases_max_consecutive_nans(self):
        """
        Verifiera hantering av max_consecutive_nans med edge cases
        """
        # Arrange
        test_series = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])
        
        # Act & Assert
        # Testa med max_consecutive_nans = 1
        result_one_limit = self.imputer.smart_forward_fill(test_series, max_consecutive_nans=1)
        
        # Verifiera att endast 1 NaN fylls med forward fill
        self.assertEqual(result_one_limit.iloc[1], 1.0, "Endast 1 NaN ska fyllas med forward fill")
        # Resten fylls med backward fill som fallback
        self.assertEqual(result_one_limit.iloc[2], 6.0, "Resten fylls med backward fill")
        self.assertEqual(result_one_limit.iloc[3], 6.0, "Resten fylls med backward fill")
        self.assertEqual(result_one_limit.iloc[4], 6.0, "Resten fylls med backward fill")
        
        # Testa med max_consecutive_nans = 2
        result_two_limit = self.imputer.smart_forward_fill(test_series, max_consecutive_nans=2)
        
        # Verifiera att endast 2 NaN fylls med forward fill
        self.assertEqual(result_two_limit.iloc[1], 1.0, "Första NaN ska fyllas med forward fill")
        self.assertEqual(result_two_limit.iloc[2], 1.0, "Andra NaN ska fyllas med forward fill")
        # Resten fylls med backward fill som fallback
        self.assertEqual(result_two_limit.iloc[3], 6.0, "Resten fylls med backward fill")
        self.assertEqual(result_two_limit.iloc[4], 6.0, "Resten fylls med backward fill")
        
        print("✅ T023 PASSED: Hantering av max_consecutive_nans edge cases fungerar korrekt")
    
    def test_t023_imputation_edge_cases_infinity_values(self):
        """
        Verifiera hantering av serier med infinity värden
        """
        # Arrange
        infinity_series = pd.Series([1.0, np.nan, np.inf, np.nan, -np.inf])
        
        # Act & Assert
        # Testa smart forward fill med infinity värden
        result_smart_fill = self.imputer.smart_forward_fill(infinity_series)
        
        # Verifiera att infinity värden hanteras korrekt
        self.assertFalse(result_smart_fill.isna().any(), "Infinity värden ska hanteras korrekt")
        self.assertTrue(np.isinf(result_smart_fill.iloc[2]), "Positiv infinity ska bevaras")
        self.assertTrue(np.isinf(result_smart_fill.iloc[4]), "Negativ infinity ska bevaras")
        
        print("✅ T023 PASSED: Hantering av serier med infinity värden fungerar korrekt")
    
    def test_t023_imputation_edge_cases_comprehensive(self):
        """
        Omfattande test av edge cases med komplexa scenarier
        """
        # Arrange
        test_scenarios = [
            # (data, description)
            (pd.Series([]), "Tom serie"),
            (pd.Series([np.nan]), "Enbart ett NaN"),
            (pd.Series([1.0]), "Enbart ett värde"),
            (pd.Series([np.nan, np.nan, np.nan]), "Alla NaN"),
            (pd.Series([0.0, np.nan, 0.0]), "Nollor med NaN"),
            (pd.Series([1.0, np.nan, np.nan, 4.0]), "Konsekutiva NaN"),
            (pd.Series([np.nan, 2.0, np.nan, 4.0, np.nan]), "Blandade NaN"),
        ]
        
        # Act & Assert
        for test_data, description in test_scenarios:
            # Testa smart forward fill
            result_smart_fill = self.imputer.smart_forward_fill(test_data)
            
            # Verifiera att resultatet är korrekt
            self.assertIsInstance(result_smart_fill, pd.Series, f"{description}: Resultatet ska vara pandas Series")
            self.assertEqual(len(result_smart_fill), len(test_data), f"{description}: Längden ska bevaras")
            
            # Testa clinical zeros
            result_clinical_zeros = self.imputer.apply_clinical_zeros(test_data, 'ETCO2')
            self.assertIsInstance(result_clinical_zeros, pd.Series, f"{description}: Clinical zeros ska returnera Series")
            
            # Testa default values
            result_default_values = self.imputer.apply_default_values(test_data, 'HR')
            self.assertIsInstance(result_default_values, pd.Series, f"{description}: Default values ska returnera Series")
        
        print("✅ T023 PASSED: Comprehensive edge cases test fungerar korrekt")
    
    def test_t023_imputation_edge_cases_performance(self):
        """
        Verifiera prestanda med stora serier
        """
        # Arrange
        import time
        large_series = pd.Series([np.nan] * 1000 + [1.0] + [np.nan] * 1000)
        
        # Act
        start_time = time.time()
        result = self.imputer.smart_forward_fill(large_series)
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0, f"Stor serie ska processas snabbt, tog {processing_time:.3f}s")
        self.assertEqual(len(result), len(large_series), "Längden ska bevaras")
        self.assertFalse(result.isna().any(), "Inga NaN ska finnas kvar")
        
        print("✅ T023 PASSED: Prestanda med stora serier fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
