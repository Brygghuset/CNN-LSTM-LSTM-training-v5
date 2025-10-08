#!/usr/bin/env python3
"""
T025: Test Range [-1, 1]
Verifiera att alla normaliserade värden ligger inom [-1, 1]
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

class TestT025RangeMinusOneToOne(unittest.TestCase):
    """T025: Test Range [-1, 1]"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t025_range_minus_one_to_one_basic(self):
        """
        T025: Test Range [-1, 1]
        Verifiera att alla normaliserade värden ligger inom [-1, 1]
        """
        # Arrange
        # Testa med olika features och värden
        test_cases = [
            ('HR', np.array([20, 60, 110, 200])),
            ('SPO2', np.array([70, 85, 100])),
            ('ETCO2', np.array([2.0, 5.0, 8.0])),
            ('BIS', np.array([0, 50, 100])),
            ('Propofol_INF', np.array([0.0, 6.0, 12.0])),
        ]
        
        # Act & Assert
        for feature_name, values in test_cases:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0), 
                           f"{feature_name}: Alla värden ska vara >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0), 
                           f"{feature_name}: Alla värden ska vara <= 1.0")
            
            # Verifiera att min och max värden ger -1 och 1
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{feature_name}: Min värde ska ge -1.0")
            self.assertAlmostEqual(normalized[-1], 1.0, places=3,
                                 msg=f"{feature_name}: Max värde ska ge 1.0")
        
        print("✅ T025 PASSED: Basic range [-1, 1] fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_edge_cases(self):
        """
        Verifiera range [-1, 1] med edge cases
        """
        # Arrange
        # Testa med värden utanför kliniska ranges
        test_cases = [
            ('HR', np.array([10, 300])),  # Utanför range 20-200
            ('SPO2', np.array([50, 120])),  # Utanför range 70-100
            ('ETCO2', np.array([0.5, 15.0])),  # Utanför range 2.0-8.0
            ('BIS', np.array([-10, 150])),  # Utanför range 0-100
        ]
        
        # Act & Assert
        for feature_name, values in test_cases:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1] (clamped)
            self.assertTrue(np.all(normalized >= -1.0), 
                           f"{feature_name}: Värden utanför range ska clampas till >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0), 
                           f"{feature_name}: Värden utanför range ska clampas till <= 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med edge cases fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_extreme_values(self):
        """
        Verifiera range [-1, 1] med extrema värden
        """
        # Arrange
        # Testa med mycket extrema värden
        extreme_values = np.array([-1000, 0, 1000])
        
        # Act & Assert
        for feature_name in ['HR', 'SPO2', 'ETCO2', 'BIS']:
            normalized = self.normalizer.normalize_feature(extreme_values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0), 
                           f"{feature_name}: Extrema värden ska clampas till >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0), 
                           f"{feature_name}: Extrema värden ska clampas till <= 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med extrema värden fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_static_features(self):
        """
        Verifiera range [-1, 1] med static features
        """
        # Arrange
        test_cases = [
            ('age', np.array([0, 60, 120])),
            ('height', np.array([100, 165, 230])),
            ('weight', np.array([20, 70, 200])),
            ('bmi', np.array([10, 25, 50])),
            ('asa', np.array([1, 3, 6])),
        ]
        
        # Act & Assert
        for feature_name, values in test_cases:
            normalized = self.normalizer.normalize_feature(values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0), 
                           f"{feature_name}: Static features ska vara >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0), 
                           f"{feature_name}: Static features ska vara <= 1.0")
            
            # Verifiera att min och max värden ger -1 och 1
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{feature_name}: Min värde ska ge -1.0")
            self.assertAlmostEqual(normalized[-1], 1.0, places=3,
                                 msg=f"{feature_name}: Max värde ska ge 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med static features fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_sex_encoding(self):
        """
        Verifiera range [-1, 1] med sex encoding
        """
        # Arrange
        # Sex ska vara -1 (Female) och 1 (Male)
        sex_values = np.array([-1, 1])
        
        # Act
        normalized = self.normalizer.normalize_feature(sex_values, 'sex')
        
        # Assert
        # Verifiera att sex encoding fungerar korrekt
        self.assertAlmostEqual(normalized[0], -1.0, places=3,
                             msg="Female (-1) ska normaliseras till -1.0")
        self.assertAlmostEqual(normalized[1], 1.0, places=3,
                             msg="Male (1) ska normaliseras till 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med sex encoding fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_dataframe(self):
        """
        Verifiera range [-1, 1] med DataFrame normalization
        """
        # Arrange
        # Skapa DataFrame med olika features (alla med samma längd)
        df = pd.DataFrame({
            'HR': [20, 60, 110, 200],
            'SPO2': [70, 85, 100, 100],  # Lägg till extra värde
            'ETCO2': [2.0, 5.0, 8.0, 8.0],  # Lägg till extra värde
            'BIS': [0, 50, 100, 100],  # Lägg till extra värde
            'age': [0, 60, 120, 120],  # Lägg till extra värde
            'sex': [-1, 1, 1, 1]  # Lägg till extra värden
        })
        
        # Act
        normalized_df = self.normalizer.normalize_dataframe(df)
        
        # Assert
        # Verifiera att alla kolumner ligger inom [-1, 1]
        for col in normalized_df.columns:
            if col in self.normalizer.clinical_ranges:
                values = normalized_df[col].dropna()
                if len(values) > 0:
                    self.assertTrue(np.all(values >= -1.0), 
                                   f"{col}: DataFrame värden ska vara >= -1.0")
                    self.assertTrue(np.all(values <= 1.0), 
                                   f"{col}: DataFrame värden ska vara <= 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med DataFrame normalization fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_nan_handling(self):
        """
        Verifiera range [-1, 1] med NaN värden
        """
        # Arrange
        # Testa med NaN värden
        test_values = np.array([50.0, np.nan, 150.0, np.nan])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'HR')
        
        # Assert
        # Verifiera att NaN värden förblir NaN
        self.assertTrue(np.isnan(normalized[1]), "NaN ska förbli NaN")
        self.assertTrue(np.isnan(normalized[3]), "NaN ska förbli NaN")
        
        # Verifiera att giltiga värden ligger inom [-1, 1]
        valid_values = normalized[~np.isnan(normalized)]
        self.assertTrue(np.all(valid_values >= -1.0), 
                       "Giltiga värden ska vara >= -1.0")
        self.assertTrue(np.all(valid_values <= 1.0), 
                       "Giltiga värden ska vara <= 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] med NaN hantering fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_precision(self):
        """
        Verifiera precision av range [-1, 1]
        """
        # Arrange
        # Testa med värden som ska ge exakta resultat
        test_values = np.array([0.0, 6.0, 12.0])  # Propofol_INF range
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'Propofol_INF')
        
        # Assert
        # Verifiera att precision är korrekt
        self.assertAlmostEqual(normalized[0], -1.0, places=6,
                             msg="Min värde ska ge exakt -1.0")
        self.assertAlmostEqual(normalized[-1], 1.0, places=6,
                             msg="Max värde ska ge exakt 1.0")
        
        # Verifiera att alla värden ligger inom [-1, 1]
        self.assertTrue(np.all(normalized >= -1.0), 
                       "Precision test ska vara >= -1.0")
        self.assertTrue(np.all(normalized <= 1.0), 
                       "Precision test ska vara <= 1.0")
        
        print("✅ T025 PASSED: Range [-1, 1] precision fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_comprehensive(self):
        """
        Omfattande test av range [-1, 1] med alla features
        """
        # Arrange
        # Testa alla features med olika värden
        all_features = self.normalizer.get_all_features()
        
        # Act & Assert
        for feature_name in all_features:
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            
            # Testa med min, max och midpoint värden
            test_values = np.array([min_clinical, (min_clinical + max_clinical) / 2, max_clinical])
            normalized = self.normalizer.normalize_feature(test_values, feature_name)
            
            # Verifiera att alla värden ligger inom [-1, 1]
            self.assertTrue(np.all(normalized >= -1.0), 
                           f"{feature_name}: Alla värden ska vara >= -1.0")
            self.assertTrue(np.all(normalized <= 1.0), 
                           f"{feature_name}: Alla värden ska vara <= 1.0")
            
            # Verifiera att min och max värden ger -1 och 1
            self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                 msg=f"{feature_name}: Min värde ska ge -1.0")
            self.assertAlmostEqual(normalized[-1], 1.0, places=3,
                                 msg=f"{feature_name}: Max värde ska ge 1.0")
        
        print("✅ T025 PASSED: Comprehensive range [-1, 1] test fungerar korrekt")
    
    def test_t025_range_minus_one_to_one_target_range_verification(self):
        """
        Verifiera att target range är korrekt satt till [-1, 1]
        """
        # Arrange & Act
        target_range = self.normalizer.target_range
        
        # Assert
        # Verifiera att target range är [-1, 1]
        self.assertEqual(target_range, (-1, 1), 
                        f"Target range ska vara (-1, 1), fick {target_range}")
        
        # Verifiera att target_min och target_max är korrekta
        self.assertEqual(self.normalizer.target_min, -1.0, 
                        "Target min ska vara -1.0")
        self.assertEqual(self.normalizer.target_max, 1.0, 
                        "Target max ska vara 1.0")
        
        print("✅ T025 PASSED: Target range [-1, 1] verifiering fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
