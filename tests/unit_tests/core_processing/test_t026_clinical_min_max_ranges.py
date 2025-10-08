#!/usr/bin/env python3
"""
T026: Test Clinical Min/Max Ranges
Verifiera att clinical ranges följer Master POC spec
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

class TestT026ClinicalMinMaxRanges(unittest.TestCase):
    """T026: Test Clinical Min/Max Ranges"""
    
    def setUp(self):
        """Setup för varje test."""
        self.normalizer = create_master_poc_unified_normalizer()
    
    def test_t026_clinical_min_max_ranges_basic(self):
        """
        T026: Test Clinical Min/Max Ranges
        Verifiera att clinical ranges följer Master POC spec
        """
        # Arrange
        # Expected clinical ranges enligt Master POC specifikationen
        expected_ranges = {
            # Vital Signs (7 features)
            'HR': (20, 200),
            'BP_SYS': (60, 250),
            'BP_DIA': (30, 150),
            'BP_MAP': (40, 180),
            'SPO2': (70, 100),
            'ETCO2': (2.0, 8.0),
            'BIS': (0, 100),
            
            # Drug Infusions (3 features)
            'Propofol_INF': (0, 12),
            'Remifentanil_INF': (0, 0.8),
            'Noradrenalin_INF': (0, 0.5),
            
            # Ventilator Settings (6 features)
            'TV': (0, 12),
            'PEEP': (0, 30),
            'FIO2': (21, 100),
            'RR': (6, 30),
            'etSEV': (0, 6),
            'inSev': (0, 8),
            
            # Static Patient Features (6 features)
            'age': (0, 120),
            'sex': (-1, 1),
            'height': (100, 230),
            'weight': (20, 200),
            'bmi': (10, 50),
            'asa': (1, 6)
        }
        
        # Act & Assert
        for feature_name, expected_range in expected_ranges.items():
            # Verifiera att feature finns i clinical_ranges
            self.assertIn(feature_name, self.normalizer.clinical_ranges,
                         f"{feature_name} ska finnas i clinical_ranges")
            
            # Verifiera att range är korrekt
            actual_range = self.normalizer.clinical_ranges[feature_name]
            self.assertEqual(actual_range, expected_range,
                           f"{feature_name} ska ha range {expected_range}, fick {actual_range}")
        
        print("✅ T026 PASSED: Basic clinical min/max ranges fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_vital_signs(self):
        """
        Verifiera clinical ranges för vital signs
        """
        # Arrange
        vital_signs_ranges = {
            'HR': (20, 200),      # Heart Rate BPM
            'BP_SYS': (60, 250),  # Systolic Blood Pressure mmHg
            'BP_DIA': (30, 150),  # Diastolic Blood Pressure mmHg
            'BP_MAP': (40, 180),  # Mean Arterial Pressure mmHg
            'SPO2': (70, 100),    # Oxygen Saturation %
            'ETCO2': (2.0, 8.0),  # End-Tidal CO2 kPa
            'BIS': (0, 100),      # Bispectral Index
        }
        
        # Act & Assert
        for feature_name, expected_range in vital_signs_ranges.items():
            actual_range = self.normalizer.clinical_ranges[feature_name]
            self.assertEqual(actual_range, expected_range,
                           f"Vital sign {feature_name} ska ha range {expected_range}")
            
            # Verifiera att min < max
            min_val, max_val = actual_range
            self.assertLess(min_val, max_val,
                          f"{feature_name}: Min ({min_val}) ska vara mindre än max ({max_val})")
        
        print("✅ T026 PASSED: Clinical ranges för vital signs fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_drug_infusions(self):
        """
        Verifiera clinical ranges för drug infusions
        """
        # Arrange
        drug_infusions_ranges = {
            'Propofol_INF': (0, 12),      # mg/kg/h
            'Remifentanil_INF': (0, 0.8), # mcg/kg/min
            'Noradrenalin_INF': (0, 0.5), # mcg/kg/min
        }
        
        # Act & Assert
        for feature_name, expected_range in drug_infusions_ranges.items():
            actual_range = self.normalizer.clinical_ranges[feature_name]
            self.assertEqual(actual_range, expected_range,
                           f"Drug infusion {feature_name} ska ha range {expected_range}")
            
            # Verifiera att alla drug infusions börjar på 0
            min_val, max_val = actual_range
            self.assertEqual(min_val, 0.0,
                          f"{feature_name}: Drug infusion ska börja på 0")
            self.assertGreater(max_val, 0.0,
                             f"{feature_name}: Max värde ska vara större än 0")
        
        print("✅ T026 PASSED: Clinical ranges för drug infusions fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_ventilator_settings(self):
        """
        Verifiera clinical ranges för ventilator settings
        """
        # Arrange
        ventilator_ranges = {
            'TV': (0, 12),        # ml/kg IBW
            'PEEP': (0, 30),      # cmH2O
            'FIO2': (21, 100),    # %
            'RR': (6, 30),        # breaths/min
            'etSEV': (0, 6),      # kPa
            'inSev': (0, 8),      # kPa
        }
        
        # Act & Assert
        for feature_name, expected_range in ventilator_ranges.items():
            actual_range = self.normalizer.clinical_ranges[feature_name]
            self.assertEqual(actual_range, expected_range,
                           f"Ventilator setting {feature_name} ska ha range {expected_range}")
            
            # Verifiera att ranges är kliniskt rimliga
            min_val, max_val = actual_range
            self.assertGreaterEqual(min_val, 0.0,
                                  f"{feature_name}: Min värde ska vara >= 0")
            self.assertGreater(max_val, min_val,
                             f"{feature_name}: Max värde ska vara större än min")
        
        print("✅ T026 PASSED: Clinical ranges för ventilator settings fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_static_features(self):
        """
        Verifiera clinical ranges för static features
        """
        # Arrange
        static_features_ranges = {
            'age': (0, 120),      # years
            'sex': (-1, 1),       # -1=Female, 1=Male
            'height': (100, 230), # cm
            'weight': (20, 200),  # kg
            'bmi': (10, 50),      # kg/m²
            'asa': (1, 6),        # ASA Score
        }
        
        # Act & Assert
        for feature_name, expected_range in static_features_ranges.items():
            actual_range = self.normalizer.clinical_ranges[feature_name]
            self.assertEqual(actual_range, expected_range,
                           f"Static feature {feature_name} ska ha range {expected_range}")
            
            # Verifiera att ranges är kliniskt rimliga
            min_val, max_val = actual_range
            self.assertGreater(max_val, min_val,
                             f"{feature_name}: Max värde ska vara större än min")
        
        print("✅ T026 PASSED: Clinical ranges för static features fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_sex_encoding(self):
        """
        Verifiera sex encoding range
        """
        # Arrange
        sex_range = self.normalizer.clinical_ranges['sex']
        
        # Act & Assert
        # Verifiera att sex range är (-1, 1)
        self.assertEqual(sex_range, (-1, 1),
                        f"Sex encoding ska vara (-1, 1), fick {sex_range}")
        
        # Verifiera att sex encoding fungerar korrekt
        female_normalized = self.normalizer.normalize_feature(-1, 'sex')
        male_normalized = self.normalizer.normalize_feature(1, 'sex')
        
        self.assertAlmostEqual(female_normalized[0], -1.0, places=3,
                             msg="Female (-1) ska normaliseras till -1.0")
        self.assertAlmostEqual(male_normalized[0], 1.0, places=3,
                             msg="Male (1) ska normaliseras till 1.0")
        
        print("✅ T026 PASSED: Sex encoding range fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_completeness(self):
        """
        Verifiera att alla features från Master POC spec finns
        """
        # Arrange
        # Expected features enligt Master POC specifikationen
        expected_features = {
            # Vital Signs (7 features)
            'HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS',
            # Drug Infusions (3 features)
            'Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF',
            # Ventilator Settings (6 features)
            'TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev',
            # Static Patient Features (6 features)
            'age', 'sex', 'height', 'weight', 'bmi', 'asa'
        }
        
        # Act
        actual_features = set(self.normalizer.clinical_ranges.keys())
        
        # Assert
        # Verifiera att alla expected features finns
        missing_features = expected_features - actual_features
        self.assertEqual(len(missing_features), 0,
                        f"Saknade features: {missing_features}")
        
        # Verifiera att det inte finns extra features
        extra_features = actual_features - expected_features
        self.assertEqual(len(extra_features), 0,
                        f"Extra features: {extra_features}")
        
        # Verifiera att vi har exakt 22 features (7+3+6+6)
        self.assertEqual(len(actual_features), 22,
                        f"Ska ha exakt 22 features, fick {len(actual_features)}")
        
        print("✅ T026 PASSED: Clinical ranges completeness fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_validation(self):
        """
        Verifiera att clinical ranges kan användas för validering
        """
        # Arrange
        # Skapa test data med värden inom och utanför ranges
        test_data = pd.DataFrame({
            'HR': [20, 110, 200, 10, 250],  # 10 och 250 är utanför range
            'SPO2': [70, 85, 100, 50, 120],  # 50 och 120 är utanför range
            'ETCO2': [2.0, 5.0, 8.0, 1.0, 10.0],  # 1.0 och 10.0 är utanför range
        })
        
        # Act
        validation_results = self.normalizer.validate_ranges(test_data)
        
        # Assert
        # Verifiera att validering fungerar
        for feature_name in ['HR', 'SPO2', 'ETCO2']:
            self.assertIn(feature_name, validation_results,
                         f"{feature_name} ska finnas i valideringsresultat")
            
            result = validation_results[feature_name]
            self.assertIn('values_below_min', result,
                         f"{feature_name} ska ha values_below_min")
            self.assertIn('values_above_max', result,
                         f"{feature_name} ska ha values_above_max")
            
            # Verifiera att värden utanför range detekteras
            self.assertGreater(result['values_below_min'], 0,
                             f"{feature_name} ska ha värden under min")
            self.assertGreater(result['values_above_max'], 0,
                             f"{feature_name} ska ha värden över max")
        
        print("✅ T026 PASSED: Clinical ranges validation fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_edge_values(self):
        """
        Verifiera edge values för clinical ranges
        """
        # Arrange
        edge_test_cases = [
            # (feature_name, min_value, max_value, description)
            ('HR', 20, 200, "Heart Rate edge values"),
            ('SPO2', 70, 100, "Oxygen Saturation edge values"),
            ('ETCO2', 2.0, 8.0, "End-Tidal CO2 edge values"),
            ('BIS', 0, 100, "Bispectral Index edge values"),
            ('age', 0, 120, "Age edge values"),
            ('height', 100, 230, "Height edge values"),
        ]
        
        # Act & Assert
        for feature_name, min_val, max_val, description in edge_test_cases:
            actual_range = self.normalizer.clinical_ranges[feature_name]
            
            # Verifiera att edge values är korrekta
            self.assertEqual(actual_range[0], min_val,
                           f"{description}: Min värde ska vara {min_val}")
            self.assertEqual(actual_range[1], max_val,
                           f"{description}: Max värde ska vara {max_val}")
            
            # Verifiera att normalization fungerar med edge values
            min_normalized = self.normalizer.normalize_feature(min_val, feature_name)
            max_normalized = self.normalizer.normalize_feature(max_val, feature_name)
            
            self.assertAlmostEqual(min_normalized[0], -1.0, places=3,
                                 msg=f"{description}: Min värde ska normaliseras till -1.0")
            self.assertAlmostEqual(max_normalized[0], 1.0, places=3,
                                 msg=f"{description}: Max värde ska normaliseras till 1.0")
        
        print("✅ T026 PASSED: Clinical ranges edge values fungerar korrekt")
    
    def test_t026_clinical_min_max_ranges_comprehensive(self):
        """
        Omfattande test av clinical ranges med alla features
        """
        # Arrange
        all_features = self.normalizer.get_all_features()
        
        # Act & Assert
        for feature_name in all_features:
            # Verifiera att feature har range
            self.assertIn(feature_name, self.normalizer.clinical_ranges,
                         f"{feature_name} ska ha clinical range")
            
            min_clinical, max_clinical = self.normalizer.clinical_ranges[feature_name]
            
            # Verifiera att range är giltig
            self.assertIsInstance(min_clinical, (int, float),
                                f"{feature_name}: Min värde ska vara numeriskt")
            self.assertIsInstance(max_clinical, (int, float),
                                f"{feature_name}: Max värde ska vara numeriskt")
            self.assertLess(min_clinical, max_clinical,
                          f"{feature_name}: Min ska vara mindre än max")
            
            # Verifiera att normalization fungerar med range
            min_normalized = self.normalizer.normalize_feature(min_clinical, feature_name)
            max_normalized = self.normalizer.normalize_feature(max_clinical, feature_name)
            
            self.assertAlmostEqual(min_normalized[0], -1.0, places=3,
                                 msg=f"{feature_name}: Min värde ska normaliseras till -1.0")
            self.assertAlmostEqual(max_normalized[0], 1.0, places=3,
                                 msg=f"{feature_name}: Max värde ska normaliseras till 1.0")
        
        print("✅ T026 PASSED: Comprehensive clinical ranges test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
