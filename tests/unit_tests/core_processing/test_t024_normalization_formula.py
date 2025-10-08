#!/usr/bin/env python3
"""
T024: Test Normalization Formula
Verifiera unified normalization formula: (value - min) / (max - min) × 2 - 1
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

class TestT024NormalizationFormula(unittest.TestCase):
    """T024: Test Normalization Formula"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t024_normalization_formula_basic(self):
        """
        T024: Test Normalization Formula
        Verifiera unified normalization formula: (value - min) / (max - min) × 2 - 1
        """
        # Arrange
        # Testa med HR (Heart Rate) - range 20-200 BPM
        test_values = np.array([20.0, 60.0, 110.0, 200.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'HR')
        
        # Assert
        # Verifiera att formula appliceras korrekt
        # För HR: normalized = (value - 20) / (200 - 20) × (1 - (-1)) + (-1)
        # normalized = (value - 20) / 180 × 2 - 1
        
        expected_values = np.array([
            (20.0 - 20) / (200 - 20) * 2 - 1,  # = -1.0
            (60.0 - 20) / (200 - 20) * 2 - 1,  # = -0.555...
            (110.0 - 20) / (200 - 20) * 2 - 1, # = 0.0
            (200.0 - 20) / (200 - 20) * 2 - 1  # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="Normalization formula ska appliceras korrekt")
        
        print("✅ T024 PASSED: Basic normalization formula fungerar korrekt")
    
    def test_t024_normalization_formula_edge_values(self):
        """
        Verifiera normalization formula med edge values (min och max)
        """
        # Arrange
        # Testa med SPO2 (Oxygen Saturation) - range 70-100%
        min_value = 70.0
        max_value = 100.0
        
        # Act
        normalized_min = self.normalizer.normalize_feature(min_value, 'SPO2')
        normalized_max = self.normalizer.normalize_feature(max_value, 'SPO2')
        
        # Assert
        # Min värde ska normaliseras till -1, max värde till 1
        self.assertAlmostEqual(normalized_min[0], -1.0, places=3,
                             msg=f"Min värde {min_value} ska normaliseras till -1.0")
        self.assertAlmostEqual(normalized_max[0], 1.0, places=3,
                             msg=f"Max värde {max_value} ska normaliseras till 1.0")
        
        print("✅ T024 PASSED: Normalization formula med edge values fungerar korrekt")
    
    def test_t024_normalization_formula_midpoint(self):
        """
        Verifiera normalization formula med midpoint värden
        """
        # Arrange
        # Testa med BP_SYS (Systolic Blood Pressure) - range 60-250 mmHg
        midpoint = (60.0 + 250.0) / 2  # = 155.0
        
        # Act
        normalized_midpoint = self.normalizer.normalize_feature(midpoint, 'BP_SYS')
        
        # Assert
        # Midpoint ska normaliseras till 0 (mitten av [-1, 1])
        self.assertAlmostEqual(normalized_midpoint[0], 0.0, places=3,
                             msg=f"Midpoint {midpoint} ska normaliseras till 0.0")
        
        print("✅ T024 PASSED: Normalization formula med midpoint fungerar korrekt")
    
    def test_t024_normalization_formula_multiple_features(self):
        """
        Verifiera normalization formula med flera olika features
        """
        # Arrange
        test_cases = [
            # (feature_name, value, expected_normalized, description)
            ('ETCO2', 2.0, -1.0, "ETCO2 min värde"),
            ('ETCO2', 5.0, 0.0, "ETCO2 midpoint"),
            ('ETCO2', 8.0, 1.0, "ETCO2 max värde"),
            ('BIS', 0.0, -1.0, "BIS min värde"),
            ('BIS', 50.0, 0.0, "BIS midpoint"),
            ('BIS', 100.0, 1.0, "BIS max värde"),
        ]
        
        # Act & Assert
        for feature_name, value, expected_normalized, description in test_cases:
            normalized = self.normalizer.normalize_feature(value, feature_name)
            
            self.assertAlmostEqual(normalized[0], expected_normalized, places=3,
                                 msg=f"{description}: {value} ska normaliseras till {expected_normalized}")
        
        print("✅ T024 PASSED: Normalization formula med flera features fungerar korrekt")
    
    def test_t024_normalization_formula_array_input(self):
        """
        Verifiera normalization formula med array input
        """
        # Arrange
        test_array = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_array, 'HR')
        
        # Assert
        # Verifiera att alla värden normaliseras korrekt
        self.assertEqual(len(normalized), len(test_array), "Array längd ska bevaras")
        
        # Verifiera att första värde (10) är utanför range och blir clamped till -1
        self.assertAlmostEqual(normalized[0], -1.0, places=3,
                             msg="Värde utanför range ska clampas till -1")
        
        # Verifiera att sista värde (90) normaliseras korrekt
        expected_last = (90.0 - 20) / (200 - 20) * 2 - 1
        self.assertAlmostEqual(normalized[-1], expected_last, places=3,
                             msg="Sista värde ska normaliseras korrekt")
        
        print("✅ T024 PASSED: Normalization formula med array input fungerar korrekt")
    
    def test_t024_normalization_formula_pandas_series(self):
        """
        Verifiera normalization formula med pandas Series input
        """
        # Arrange
        test_series = pd.Series([25.0, 50.0, 75.0, 100.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_series, 'SPO2')
        
        # Assert
        # Verifiera att resultatet är numpy array
        self.assertIsInstance(normalized, np.ndarray, "Resultatet ska vara numpy array")
        
        # Verifiera att alla värden normaliseras korrekt
        expected_values = np.array([
            -1.0,  # 25.0 är utanför range och clampas till -1
            -1.0,  # 50.0 är utanför range och clampas till -1
            (75.0 - 70) / (100 - 70) * 2 - 1,  # = -0.333...
            1.0   # 100.0 är max range och blir 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="Pandas Series ska normaliseras korrekt")
        
        print("✅ T024 PASSED: Normalization formula med pandas Series fungerar korrekt")
    
    def test_t024_normalization_formula_single_value(self):
        """
        Verifiera normalization formula med enbart ett värde
        """
        # Arrange
        single_value = 150.0
        
        # Act
        normalized = self.normalizer.normalize_feature(single_value, 'BP_SYS')
        
        # Assert
        # Verifiera att enbart ett värde normaliseras korrekt
        self.assertEqual(len(normalized), 1, "Enbart ett värde ska returneras")
        
        expected = (150.0 - 60) / (250 - 60) * 2 - 1
        self.assertAlmostEqual(normalized[0], expected, places=3,
                             msg=f"Enbart ett värde {single_value} ska normaliseras korrekt")
        
        print("✅ T024 PASSED: Normalization formula med enbart ett värde fungerar korrekt")
    
    def test_t024_normalization_formula_nan_handling(self):
        """
        Verifiera normalization formula med NaN värden
        """
        # Arrange
        test_values = np.array([50.0, np.nan, 100.0, np.nan])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'HR')
        
        # Assert
        # Verifiera att NaN värden hanteras korrekt
        self.assertTrue(np.isnan(normalized[1]), "NaN ska förbli NaN")
        self.assertTrue(np.isnan(normalized[3]), "NaN ska förbli NaN")
        
        # Verifiera att giltiga värden normaliseras korrekt
        expected_first = (50.0 - 20) / (200 - 20) * 2 - 1
        expected_third = (100.0 - 20) / (200 - 20) * 2 - 1
        
        self.assertAlmostEqual(normalized[0], expected_first, places=3,
                             msg="Giltiga värden ska normaliseras korrekt")
        self.assertAlmostEqual(normalized[2], expected_third, places=3,
                             msg="Giltiga värden ska normaliseras korrekt")
        
        print("✅ T024 PASSED: Normalization formula med NaN hantering fungerar korrekt")
    
    def test_t024_normalization_formula_precision(self):
        """
        Verifiera precision av normalization formula
        """
        # Arrange
        # Testa med värden som ska ge exakta resultat
        test_values = np.array([0.0, 0.5, 1.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'Propofol_INF')
        
        # Assert
        # Verifiera att precision är korrekt
        expected_values = np.array([
            (0.0 - 0) / (12 - 0) * 2 - 1,    # = -1.0
            (0.5 - 0) / (12 - 0) * 2 - 1,    # = -0.916...
            (1.0 - 0) / (12 - 0) * 2 - 1     # = -0.833...
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=6,
                                            err_msg="Normalization precision ska vara korrekt")
        
        print("✅ T024 PASSED: Normalization formula precision fungerar korrekt")
    
    def test_t024_normalization_formula_comprehensive(self):
        """
        Omfattande test av normalization formula med komplexa scenarier
        """
        # Arrange
        test_scenarios = [
            # (feature_name, values, description)
            ('HR', np.array([20, 110, 200]), "HR edge och midpoint"),
            ('SPO2', np.array([70, 85, 100]), "SPO2 edge och midpoint"),
            ('ETCO2', np.array([2.0, 5.0, 8.0]), "ETCO2 edge och midpoint"),
            ('BIS', np.array([0, 50, 100]), "BIS edge och midpoint"),
            ('Propofol_INF', np.array([0.0, 6.0, 12.0]), "Propofol edge och midpoint"),
        ]
        
        # Act & Assert
        for feature_name, values, description in test_scenarios:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att första värde är -1, sista är 1, mittenvärde är nära 0
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{description}: Första värde ska vara -1.0")
            self.assertAlmostEqual(normalized[-1], 1.0, places=3,
                                 msg=f"{description}: Sista värde ska vara 1.0")
            
            if len(values) == 3:  # Om vi har 3 värden, kontrollera mittenvärde
                self.assertAlmostEqual(normalized[1], 0.0, places=3,
                                     msg=f"{description}: Mittenvärde ska vara 0.0")
        
        print("✅ T024 PASSED: Comprehensive normalization formula test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
