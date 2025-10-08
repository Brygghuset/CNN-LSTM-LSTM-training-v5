#!/usr/bin/env python3
"""
T028: Test Sex Encoding
Verifiera att sex kodas som -1 (Female) och 1 (Male)
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

class TestT028SexEncoding(unittest.TestCase):
    """T028: Test Sex Encoding"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t028_sex_encoding_basic(self):
        """
        T028: Test Sex Encoding
        Verifiera att sex kodas som -1 (Female) och 1 (Male)
        """
        # Arrange
        # Testa sex encoding med olika värden
        test_cases = [
            # (sex_value, expected_normalized, description)
            (-1, -1.0, "Female encoding"),
            (1, 1.0, "Male encoding"),
            (0, 0.0, "Zero ska normaliseras till 0.0"),  # Edge case
        ]
        
        # Act & Assert
        for sex_value, expected_normalized, description in test_cases:
            normalized = self.normalizer.normalize_feature(sex_value, 'sex')
            
            # Verifiera att sex encoding fungerar korrekt
            self.assertAlmostEqual(normalized[0], expected_normalized, places=3,
                                 msg=f"{description}: Sex {sex_value} ska normaliseras till {expected_normalized}")
            
            # Verifiera att värden ligger inom [-1, 1]
            self.assertTrue(-1.0 <= normalized[0] <= 1.0,
                           f"{description}: Normaliserat värde ska ligga inom [-1, 1]")
        
        print("✅ T028 PASSED: Basic sex encoding fungerar korrekt")
    
    def test_t028_sex_encoding_female_male(self):
        """
        Verifiera Female (-1) och Male (1) encoding specifikt
        """
        # Arrange
        female_value = -1
        male_value = 1
        
        # Act
        female_normalized = self.normalizer.normalize_feature(female_value, 'sex')
        male_normalized = self.normalizer.normalize_feature(male_value, 'sex')
        
        # Assert
        # Verifiera att Female (-1) normaliseras till -1.0
        self.assertAlmostEqual(female_normalized[0], -1.0, places=3,
                             msg="Female (-1) ska normaliseras till -1.0")
        
        # Verifiera att Male (1) normaliseras till 1.0
        self.assertAlmostEqual(male_normalized[0], 1.0, places=3,
                             msg="Male (1) ska normaliseras till 1.0")
        
        # Verifiera att Female och Male är olika
        self.assertNotEqual(female_normalized[0], male_normalized[0],
                           msg="Female och Male ska ha olika normaliserade värden")
        
        print("✅ T028 PASSED: Female och Male sex encoding fungerar korrekt")
    
    def test_t028_sex_encoding_range(self):
        """
        Verifiera sex encoding range (-1, 1)
        """
        # Arrange
        # Sex range ska vara (-1, 1) enligt Master POC spec
        sex_range = self.normalizer.clinical_ranges['sex']
        
        # Act & Assert
        # Verifiera att sex range är (-1, 1)
        self.assertEqual(sex_range, (-1, 1),
                        f"Sex range ska vara (-1, 1), fick {sex_range}")
        
        # Verifiera att min och max värden ger korrekt normalization
        min_normalized = self.normalizer.normalize_feature(sex_range[0], 'sex')
        max_normalized = self.normalizer.normalize_feature(sex_range[1], 'sex')
        
        self.assertAlmostEqual(min_normalized[0], -1.0, places=3,
                             msg="Sex min värde (-1) ska normaliseras till -1.0")
        self.assertAlmostEqual(max_normalized[0], 1.0, places=3,
                             msg="Sex max värde (1) ska normaliseras till 1.0")
        
        print("✅ T028 PASSED: Sex encoding range fungerar korrekt")
    
    def test_t028_sex_encoding_array(self):
        """
        Verifiera sex encoding med array av värden
        """
        # Arrange
        sex_values = np.array([-1, 1, -1, 1, -1])
        expected_normalized = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(sex_values, 'sex')
        
        # Assert
        # Verifiera att array normalization fungerar korrekt
        np.testing.assert_array_almost_equal(normalized, expected_normalized, decimal=3,
                                            err_msg="Sex array normalization ska fungera korrekt")
        
        # Verifiera att alla värden ligger inom [-1, 1]
        self.assertTrue(np.all(normalized >= -1.0),
                       "Alla sex värden ska vara >= -1.0")
        self.assertTrue(np.all(normalized <= 1.0),
                       "Alla sex värden ska vara <= 1.0")
        
        print("✅ T028 PASSED: Sex encoding med array fungerar korrekt")
    
    def test_t028_sex_encoding_pandas_series(self):
        """
        Verifiera sex encoding med pandas Series
        """
        # Arrange
        sex_series = pd.Series([-1, 1, -1, 1, -1], name='sex')
        expected_normalized = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        
        # Act
        normalized = self.normalizer.normalize_feature(sex_series, 'sex')
        
        # Assert
        # Verifiera att pandas Series normalization fungerar korrekt
        np.testing.assert_array_almost_equal(normalized, expected_normalized, decimal=3,
                                            err_msg="Sex pandas Series normalization ska fungera korrekt")
        
        # Verifiera att längden bevaras
        self.assertEqual(len(normalized), len(sex_series),
                       "Sex Series längd ska bevaras")
        
        print("✅ T028 PASSED: Sex encoding med pandas Series fungerar korrekt")
    
    def test_t028_sex_encoding_dataframe(self):
        """
        Verifiera sex encoding med DataFrame
        """
        # Arrange
        df = pd.DataFrame({
            'sex': [-1, 1, -1, 1, -1],
            'age': [25, 30, 35, 40, 45],
            'height': [160, 175, 165, 180, 170]
        })
        
        # Act
        normalized_df = self.normalizer.normalize_dataframe(df)
        
        # Assert
        # Verifiera att sex kolumn normaliseras korrekt
        sex_normalized = normalized_df['sex'].values
        expected_sex = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        
        np.testing.assert_array_almost_equal(sex_normalized, expected_sex, decimal=3,
                                            err_msg="Sex DataFrame normalization ska fungera korrekt")
        
        # Verifiera att andra kolumner också normaliseras
        self.assertIn('age', normalized_df.columns,
                     "Age kolumn ska finnas i normalized DataFrame")
        self.assertIn('height', normalized_df.columns,
                     "Height kolumn ska finnas i normalized DataFrame")
        
        print("✅ T028 PASSED: Sex encoding med DataFrame fungerar korrekt")
    
    def test_t028_sex_encoding_edge_cases(self):
        """
        Verifiera sex encoding med edge cases
        """
        # Arrange
        edge_cases = [
            # (sex_value, description)
            (-1, "Female (-1)"),
            (1, "Male (1)"),
            (0, "Zero (edge case)"),
            (-2, "Under min (edge case)"),
            (2, "Over max (edge case)"),
        ]
        
        # Act & Assert
        for sex_value, description in edge_cases:
            normalized = self.normalizer.normalize_feature(sex_value, 'sex')
            
            # Verifiera att alla värden ligger inom [-1, 1] (clamped)
            self.assertTrue(-1.0 <= normalized[0] <= 1.0,
                           f"{description}: Sex värde ska ligga inom [-1, 1]")
            
            # Verifiera att edge cases hanteras korrekt
            if sex_value == -1:
                self.assertAlmostEqual(normalized[0], -1.0, places=3,
                                     msg=f"{description}: Ska normaliseras till -1.0")
            elif sex_value == 1:
                self.assertAlmostEqual(normalized[0], 1.0, places=3,
                                     msg=f"{description}: Ska normaliseras till 1.0")
            elif sex_value == 0:
                # Zero ska normaliseras till 0.0
                self.assertAlmostEqual(normalized[0], 0.0, places=3,
                                     msg=f"{description}: Zero ska normaliseras till 0.0")
        
        print("✅ T028 PASSED: Sex encoding med edge cases fungerar korrekt")
    
    def test_t028_sex_encoding_reverse_normalization(self):
        """
        Verifiera reverse normalization för sex encoding
        """
        # Arrange
        sex_values = np.array([-1, 1, -1, 1])
        
        # Act
        normalized = self.normalizer.normalize_feature(sex_values, 'sex')
        denormalized = self.normalizer.denormalize_feature(normalized, 'sex')
        
        # Assert
        # Verifiera att reverse normalization fungerar korrekt
        np.testing.assert_array_almost_equal(denormalized, sex_values, decimal=3,
                                            err_msg="Sex reverse normalization ska fungera korrekt")
        
        # Verifiera att Female och Male bevaras
        self.assertAlmostEqual(denormalized[0], -1.0, places=3,
                             msg="Female ska bevaras efter reverse normalization")
        self.assertAlmostEqual(denormalized[1], 1.0, places=3,
                             msg="Male ska bevaras efter reverse normalization")
        
        print("✅ T028 PASSED: Sex encoding reverse normalization fungerar korrekt")
    
    def test_t028_sex_encoding_clinical_interpretation(self):
        """
        Verifiera klinisk tolkning av sex encoding
        """
        # Arrange
        # Simulera patientdata med olika kön
        patient_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003', 'P004'],
            'sex': [-1, 1, -1, 1],  # Female, Male, Female, Male
            'age': [25, 30, 35, 40]
        })
        
        # Act
        normalized_df = self.normalizer.normalize_dataframe(patient_data)
        
        # Assert
        # Verifiera att sex encoding bevarar klinisk information
        sex_normalized = normalized_df['sex'].values
        
        # Verifiera att Female patients har -1.0
        female_patients = sex_normalized[patient_data['sex'] == -1]
        self.assertTrue(np.all(female_patients == -1.0),
                       "Female patients ska ha sex = -1.0")
        
        # Verifiera att Male patients har 1.0
        male_patients = sex_normalized[patient_data['sex'] == 1]
        self.assertTrue(np.all(male_patients == 1.0),
                       "Male patients ska ha sex = 1.0")
        
        # Verifiera att kön kan särskiljas
        unique_sex_values = np.unique(sex_normalized)
        self.assertEqual(len(unique_sex_values), 2,
                       "Sex ska ha exakt 2 unika värden")
        self.assertIn(-1.0, unique_sex_values,
                     "Sex ska innehålla -1.0 (Female)")
        self.assertIn(1.0, unique_sex_values,
                     "Sex ska innehålla 1.0 (Male)")
        
        print("✅ T028 PASSED: Sex encoding klinisk tolkning fungerar korrekt")
    
    def test_t028_sex_encoding_comprehensive(self):
        """
        Omfattande test av sex encoding
        """
        # Arrange
        # Testa alla möjliga sex värden inom och utanför range
        test_values = np.array([-2, -1, 0, 1, 2])
        
        # Act
        normalized = self.normalizer.normalize_feature(test_values, 'sex')
        
        # Assert
        # Verifiera att alla värden ligger inom [-1, 1]
        self.assertTrue(np.all(normalized >= -1.0),
                       "Alla sex värden ska vara >= -1.0")
        self.assertTrue(np.all(normalized <= 1.0),
                       "Alla sex värden ska vara <= 1.0")
        
        # Verifiera att Female (-1) och Male (1) bevaras
        self.assertAlmostEqual(normalized[1], -1.0, places=3,
                             msg="Female (-1) ska bevaras")
        self.assertAlmostEqual(normalized[3], 1.0, places=3,
                             msg="Male (1) ska bevaras")
        
        # Verifiera att sex encoding är deterministisk
        normalized2 = self.normalizer.normalize_feature(test_values, 'sex')
        np.testing.assert_array_almost_equal(normalized, normalized2, decimal=6,
                                            err_msg="Sex encoding ska vara deterministisk")
        
        print("✅ T028 PASSED: Comprehensive sex encoding test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
