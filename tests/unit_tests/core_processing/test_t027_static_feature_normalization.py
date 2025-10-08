#!/usr/bin/env python3
"""
T027: Test Static Feature Normalization
Verifiera normalization av age, height, weight, bmi, asa
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

class TestT027StaticFeatureNormalization(unittest.TestCase):
    """T027: Test Static Feature Normalization"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t027_static_feature_normalization_basic(self):
        """
        T027: Test Static Feature Normalization
        Verifiera normalization av age, height, weight, bmi, asa
        """
        # Arrange
        # Testa static features med olika värden
        test_cases = [
            # (feature_name, values, description)
            ('age', np.array([0, 60, 120]), "Age normalization"),
            ('height', np.array([100, 165, 230]), "Height normalization"),
            ('weight', np.array([20, 70, 200]), "Weight normalization"),
            ('bmi', np.array([10, 25, 50]), "BMI normalization"),
            ('asa', np.array([1, 3, 6]), "ASA Score normalization"),
        ]
        
        # Act & Assert
        for feature_name, values, description in test_cases:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att normalization fungerar
            self.assertEqual(len(normalized), len(values),
                           f"{description}: Längd ska bevaras")
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0),
                           f"{description}: Alla värden ska vara >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0),
                           f"{description}: Alla värden ska vara <= 1.0")
            
            # Verifiera att min och max värden ger -1 och 1
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{description}: Min värde ska ge -1.0")
            self.assertAlmostEqual(normalized[-1], 1.0, places=3,
                                 msg=f"{description}: Max värde ska ge 1.0")
        
        print("✅ T027 PASSED: Basic static feature normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_age(self):
        """
        Verifiera age normalization specifikt
        """
        # Arrange
        # Age range: 0-120 years
        age_values = np.array([0, 30, 60, 90, 120])
        
        # Act
        normalized = self.normalizer.normalize_feature(age_values, 'age')
        
        # Assert
        # Verifiera att age normalization fungerar korrekt
        expected_values = np.array([
            (0 - 0) / (120 - 0) * 2 - 1,    # = -1.0
            (30 - 0) / (120 - 0) * 2 - 1,   # = -0.5
            (60 - 0) / (120 - 0) * 2 - 1,   # = 0.0
            (90 - 0) / (120 - 0) * 2 - 1,   # = 0.5
            (120 - 0) / (120 - 0) * 2 - 1   # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="Age normalization ska fungera korrekt")
        
        print("✅ T027 PASSED: Age normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_height(self):
        """
        Verifiera height normalization specifikt
        """
        # Arrange
        # Height range: 100-230 cm
        height_values = np.array([100, 150, 165, 200, 230])
        
        # Act
        normalized = self.normalizer.normalize_feature(height_values, 'height')
        
        # Assert
        # Verifiera att height normalization fungerar korrekt
        expected_values = np.array([
            (100 - 100) / (230 - 100) * 2 - 1,  # = -1.0
            (150 - 100) / (230 - 100) * 2 - 1,  # = -0.230...
            (165 - 100) / (230 - 100) * 2 - 1,  # = 0.0
            (200 - 100) / (230 - 100) * 2 - 1,  # = 0.538...
            (230 - 100) / (230 - 100) * 2 - 1   # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="Height normalization ska fungera korrekt")
        
        print("✅ T027 PASSED: Height normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_weight(self):
        """
        Verifiera weight normalization specifikt
        """
        # Arrange
        # Weight range: 20-200 kg
        weight_values = np.array([20, 60, 110, 160, 200])
        
        # Act
        normalized = self.normalizer.normalize_feature(weight_values, 'weight')
        
        # Assert
        # Verifiera att weight normalization fungerar korrekt
        expected_values = np.array([
            (20 - 20) / (200 - 20) * 2 - 1,   # = -1.0
            (60 - 20) / (200 - 20) * 2 - 1,   # = -0.555...
            (110 - 20) / (200 - 20) * 2 - 1,  # = 0.0
            (160 - 20) / (200 - 20) * 2 - 1,  # = 0.555...
            (200 - 20) / (200 - 20) * 2 - 1   # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="Weight normalization ska fungera korrekt")
        
        print("✅ T027 PASSED: Weight normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_bmi(self):
        """
        Verifiera BMI normalization specifikt
        """
        # Arrange
        # BMI range: 10-50 kg/m²
        bmi_values = np.array([10, 20, 30, 40, 50])
        
        # Act
        normalized = self.normalizer.normalize_feature(bmi_values, 'bmi')
        
        # Assert
        # Verifiera att BMI normalization fungerar korrekt
        expected_values = np.array([
            (10 - 10) / (50 - 10) * 2 - 1,  # = -1.0
            (20 - 10) / (50 - 10) * 2 - 1,  # = -0.5
            (30 - 10) / (50 - 10) * 2 - 1,  # = 0.0
            (40 - 10) / (50 - 10) * 2 - 1,  # = 0.5
            (50 - 10) / (50 - 10) * 2 - 1   # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="BMI normalization ska fungera korrekt")
        
        print("✅ T027 PASSED: BMI normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_asa(self):
        """
        Verifiera ASA Score normalization specifikt
        """
        # Arrange
        # ASA range: 1-6
        asa_values = np.array([1, 2, 3, 4, 5, 6])
        
        # Act
        normalized = self.normalizer.normalize_feature(asa_values, 'asa')
        
        # Assert
        # Verifiera att ASA normalization fungerar korrekt
        expected_values = np.array([
            (1 - 1) / (6 - 1) * 2 - 1,  # = -1.0
            (2 - 1) / (6 - 1) * 2 - 1,  # = -0.6
            (3 - 1) / (6 - 1) * 2 - 1,  # = -0.2
            (4 - 1) / (6 - 1) * 2 - 1,  # = 0.2
            (5 - 1) / (6 - 1) * 2 - 1,  # = 0.6
            (6 - 1) / (6 - 1) * 2 - 1   # = 1.0
        ])
        
        np.testing.assert_array_almost_equal(normalized, expected_values, decimal=3,
                                            err_msg="ASA normalization ska fungera korrekt")
        
        print("✅ T027 PASSED: ASA Score normalization fungerar korrekt")
    
    def test_t027_static_feature_normalization_dataframe(self):
        """
        Verifiera static feature normalization med DataFrame
        """
        # Arrange
        # Skapa DataFrame med static features
        df = pd.DataFrame({
            'age': [25, 45, 65, 85],
            'height': [150, 165, 180, 195],
            'weight': [50, 70, 90, 110],
            'bmi': [18, 25, 28, 35],
            'asa': [1, 2, 3, 4]
        })
        
        # Act
        normalized_df = self.normalizer.normalize_dataframe(df)
        
        # Assert
        # Verifiera att alla static features normaliseras korrekt
        static_features = ['age', 'height', 'weight', 'bmi', 'asa']
        
        for feature in static_features:
            normalized_values = normalized_df[feature].values
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized_values >= -1.0),
                           f"{feature}: DataFrame värden ska vara >= -1.0")
            self.assertTrue(np.all(normalized_values <= 1.0),
                           f"{feature}: DataFrame värden ska vara <= 1.0")
            
            # Verifiera att längden bevaras
            self.assertEqual(len(normalized_values), len(df[feature]),
                           f"{feature}: Längd ska bevaras")
        
        print("✅ T027 PASSED: Static feature normalization med DataFrame fungerar korrekt")
    
    def test_t027_static_feature_normalization_edge_cases(self):
        """
        Verifiera static feature normalization med edge cases
        """
        # Arrange
        # Testa med värden utanför ranges
        edge_cases = [
            ('age', np.array([-10, 0, 120, 150])),      # -10 och 150 är utanför range
            ('height', np.array([50, 100, 230, 300])),  # 50 och 300 är utanför range
            ('weight', np.array([5, 20, 200, 250])),    # 5 och 250 är utanför range
            ('bmi', np.array([5, 10, 50, 60])),        # 5 och 60 är utanför range
            ('asa', np.array([0, 1, 6, 7])),            # 0 och 7 är utanför range
        ]
        
        # Act & Assert
        for feature_name, values in edge_cases:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1] (clamped)
            self.assertTrue(np.all(normalized >= -1.0),
                           f"{feature_name}: Edge case värden ska clampas till >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0),
                           f"{feature_name}: Edge case värden ska clampas till <= 1.0")
        
        print("✅ T027 PASSED: Static feature normalization med edge cases fungerar korrekt")
    
    def test_t027_static_feature_normalization_precision(self):
        """
        Verifiera precision av static feature normalization
        """
        # Arrange
        # Testa med värden som ska ge exakta resultat
        precision_cases = [
            ('age', 60, 0.0),      # Midpoint ska ge 0.0
            ('height', 165, 0.0),  # Midpoint ska ge 0.0
            ('weight', 110, 0.0),  # Midpoint ska ge 0.0
            ('bmi', 30, 0.0),      # Midpoint ska ge 0.0
            ('asa', 3.5, 0.0),     # Midpoint ska ge 0.0
        ]
        
        # Act & Assert
        for feature_name, value, expected_normalized in precision_cases:
            normalized = self.normalizer.normalize_feature(value, feature_name)
            
            # Verifiera att precision är korrekt
            self.assertAlmostEqual(normalized[0], expected_normalized, places=6,
                                 msg=f"{feature_name}: Midpoint {value} ska normaliseras till {expected_normalized}")
        
        print("✅ T027 PASSED: Static feature normalization precision fungerar korrekt")
    
    def test_t027_static_feature_normalization_comprehensive(self):
        """
        Omfattande test av static feature normalization
        """
        # Arrange
        static_features = ['age', 'height', 'weight', 'bmi', 'asa']
        
        # Act & Assert
        for feature_name in static_features:
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            
            # Testa med min, midpoint och max värden
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            
            # Verifiera att normalization fungerar korrekt
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{feature_name}: Min värde ska ge -1.0")
            self.assertAlmostEqual(normalized[1], 0.0, places=3,
                                 msg=f"{feature_name}: Midpoint ska ge 0.0")
            self.assertAlmostEqual(normalized[2], 1.0, places=3,
                                 msg=f"{feature_name}: Max värde ska ge 1.0")
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0),
                           f"{feature_name}: Alla värden ska vara >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0),
                           f"{feature_name}: Alla värden ska vara <= 1.0")
        
        print("✅ T027 PASSED: Comprehensive static feature normalization test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
