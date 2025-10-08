#!/usr/bin/env python3
"""
T008: Test 6 Static Features
Verifiera att age, sex, height, weight, bmi, asa mappas korrekt
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår feature mapping modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_feature_mapping", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_feature_mapping.py')
)
master_poc_feature_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_feature_mapping)

# Använd modulen
MASTER_POC_STATIC_FEATURES = master_poc_feature_mapping.MASTER_POC_STATIC_FEATURES
get_static_feature_order = master_poc_feature_mapping.get_static_feature_order

class TestT008StaticFeatures(unittest.TestCase):
    """T008: Test 6 Static Features"""
    
    def test_t008_static_features_count(self):
        """
        T008: Test 6 Static Features
        Verifiera att exakt 6 static features mappas korrekt
        """
        # Arrange
        expected_count = 6
        
        # Act
        actual_count = len(MASTER_POC_STATIC_FEATURES)
        
        # Assert
        self.assertEqual(actual_count, expected_count, 
                        f"Förväntade {expected_count} static features, fick {actual_count}")
        
        print("✅ T008 PASSED: Exakt 6 static features mappas korrekt")
    
    def test_t008_static_feature_order(self):
        """
        Verifiera att static features har korrekt ordning enligt Master POC spec
        """
        # Arrange
        expected_order = ['age', 'sex', 'height', 'weight', 'bmi', 'asa']
        
        # Act
        actual_order = get_static_feature_order()
        
        # Assert
        self.assertEqual(actual_order, expected_order,
                        f"Static feature ordning ska följa Master POC spec")
        
        print("✅ T008 PASSED: Static features har korrekt ordning")
    
    def test_t008_age_feature(self):
        """
        Verifiera att age feature är korrekt definierad
        """
        # Arrange
        feature = 'age'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], 'years', f"Age ska ha enhet 'years'")
        self.assertEqual(feature_config['default_value'], 50, f"Age ska ha default värde 50")
        self.assertEqual(feature_config['range'], (0, 120), f"Age ska ha range (0, 120)")
        self.assertEqual(feature_config['normalization_formula'], 'age/120 × 2 - 1',
                        f"Age ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: Age feature är korrekt definierad")
    
    def test_t008_sex_feature(self):
        """
        Verifiera att sex feature är korrekt definierad
        """
        # Arrange
        feature = 'sex'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], 'F/M', f"Sex ska ha enhet 'F/M'")
        self.assertEqual(feature_config['default_value'], -1, f"Sex ska ha default värde -1 (Female)")
        self.assertEqual(feature_config['range'], (-1, 1), f"Sex ska ha range (-1, 1)")
        self.assertEqual(feature_config['normalization_formula'], '1.0 eller -1.0',
                        f"Sex ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: Sex feature är korrekt definierad")
    
    def test_t008_height_feature(self):
        """
        Verifiera att height feature är korrekt definierad
        """
        # Arrange
        feature = 'height'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], 'cm', f"Height ska ha enhet 'cm'")
        self.assertEqual(feature_config['default_value'], 170, f"Height ska ha default värde 170")
        self.assertEqual(feature_config['range'], (100, 230), f"Height ska ha range (100, 230)")
        self.assertEqual(feature_config['normalization_formula'], '(height-100)/130 × 2 - 1',
                        f"Height ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: Height feature är korrekt definierad")
    
    def test_t008_weight_feature(self):
        """
        Verifiera att weight feature är korrekt definierad
        """
        # Arrange
        feature = 'weight'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], 'kg', f"Weight ska ha enhet 'kg'")
        self.assertEqual(feature_config['default_value'], 70, f"Weight ska ha default värde 70")
        self.assertEqual(feature_config['range'], (20, 200), f"Weight ska ha range (20, 200)")
        self.assertEqual(feature_config['normalization_formula'], '(weight-20)/180 × 2 - 1',
                        f"Weight ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: Weight feature är korrekt definierad")
    
    def test_t008_bmi_feature(self):
        """
        Verifiera att bmi feature är korrekt definierad
        """
        # Arrange
        feature = 'bmi'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], 'kg/m²', f"BMI ska ha enhet 'kg/m²'")
        self.assertEqual(feature_config['default_value'], 24.2, f"BMI ska ha default värde 24.2")
        self.assertEqual(feature_config['range'], (10, 50), f"BMI ska ha range (10, 50)")
        self.assertEqual(feature_config['normalization_formula'], '(bmi-10)/40 × 2 - 1',
                        f"BMI ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: BMI feature är korrekt definierad")
    
    def test_t008_asa_feature(self):
        """
        Verifiera att asa feature är korrekt definierad
        """
        # Arrange
        feature = 'asa'
        
        # Act
        feature_config = MASTER_POC_STATIC_FEATURES[feature]
        
        # Assert
        self.assertEqual(feature_config['unit'], '1-6', f"ASA ska ha enhet '1-6'")
        self.assertEqual(feature_config['default_value'], 2, f"ASA ska ha default värde 2")
        self.assertEqual(feature_config['range'], (1, 6), f"ASA ska ha range (1, 6)")
        self.assertEqual(feature_config['normalization_formula'], '(asa-1)/5 × 2 - 1',
                        f"ASA ska ha korrekt normalization formel")
        
        print("✅ T008 PASSED: ASA feature är korrekt definierad")
    
    def test_t008_all_static_features_have_required_properties(self):
        """
        Verifiera att alla static features har required properties
        """
        # Arrange
        required_properties = ['order', 'unit', 'default_value', 'range', 'normalization_formula']
        
        # Act & Assert
        for feature_name, feature_config in MASTER_POC_STATIC_FEATURES.items():
            for prop in required_properties:
                self.assertIn(prop, feature_config, 
                            f"Static feature {feature_name} saknar property '{prop}'")
        
        print("✅ T008 PASSED: Alla static features har required properties")

if __name__ == '__main__':
    unittest.main()
