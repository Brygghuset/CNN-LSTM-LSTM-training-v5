#!/usr/bin/env python3
"""
T007: Test 16 Timeseries Features
Verifiera att exakt 16 timeseries features mappas enligt Master POC spec
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår feature mapping modul (undvik __init__.py problem)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_feature_mapping", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_feature_mapping.py')
)
master_poc_feature_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_feature_mapping)

# Använd modulen
MASTER_POC_TIMESERIES_FEATURES = master_poc_feature_mapping.MASTER_POC_TIMESERIES_FEATURES
get_master_poc_feature_count = master_poc_feature_mapping.get_master_poc_feature_count
get_timeseries_feature_order = master_poc_feature_mapping.get_timeseries_feature_order
validate_master_poc_feature_mapping = master_poc_feature_mapping.validate_master_poc_feature_mapping

class TestT007TimeseriesFeatures(unittest.TestCase):
    """T007: Test 16 Timeseries Features"""
    
    def test_t007_timeseries_features_count(self):
        """
        T007: Test 16 Timeseries Features
        Verifiera att exakt 16 timeseries features mappas enligt Master POC spec
        """
        # Arrange
        expected_count = 16
        
        # Act
        actual_count = len(MASTER_POC_TIMESERIES_FEATURES)
        feature_counts = get_master_poc_feature_count()
        
        # Assert
        self.assertEqual(actual_count, expected_count, 
                        f"Förväntade {expected_count} timeseries features, fick {actual_count}")
        self.assertEqual(feature_counts['timeseries_features'], expected_count,
                        f"Feature count dict ska visa {expected_count} timeseries features")
        
        print("✅ T007 PASSED: Exakt 16 timeseries features mappas korrekt")
    
    def test_t007_timeseries_feature_order(self):
        """
        Verifiera att timeseries features har korrekt ordning enligt Master POC spec
        """
        # Arrange
        expected_order = [
            'HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS',
            'Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF',
            'TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev'
        ]
        
        # Act
        actual_order = get_timeseries_feature_order()
        
        # Assert
        self.assertEqual(actual_order, expected_order,
                        f"Feature ordning ska följa Master POC spec")
        
        print("✅ T007 PASSED: Timeseries features har korrekt ordning")
    
    def test_t007_vital_signs_features(self):
        """
        Verifiera att vital signs features (7 st) är korrekt definierade
        """
        # Arrange
        vital_signs = ['HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS']
        expected_count = 7
        
        # Act & Assert
        self.assertEqual(len(vital_signs), expected_count,
                        f"Förväntade {expected_count} vital signs features")
        
        for feature in vital_signs:
            self.assertIn(feature, MASTER_POC_TIMESERIES_FEATURES,
                         f"Vital sign {feature} saknas i feature mapping")
            
            # Verifiera att feature har required properties
            feature_config = MASTER_POC_TIMESERIES_FEATURES[feature]
            self.assertIn('order', feature_config, f"Feature {feature} saknar 'order'")
            self.assertIn('unit', feature_config, f"Feature {feature} saknar 'unit'")
            self.assertIn('clinical_range', feature_config, f"Feature {feature} saknar 'clinical_range'")
            self.assertIn('default_value', feature_config, f"Feature {feature} saknar 'default_value'")
        
        print("✅ T007 PASSED: Vital signs features är korrekt definierade")
    
    def test_t007_drug_infusion_features(self):
        """
        Verifiera att drug infusion features (3 st) är korrekt definierade
        """
        # Arrange
        drug_features = ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF']
        expected_count = 3
        
        # Act & Assert
        self.assertEqual(len(drug_features), expected_count,
                        f"Förväntade {expected_count} drug infusion features")
        
        for feature in drug_features:
            self.assertIn(feature, MASTER_POC_TIMESERIES_FEATURES,
                         f"Drug feature {feature} saknas i feature mapping")
            
            # Verifiera att drug features har concentration info
            feature_config = MASTER_POC_TIMESERIES_FEATURES[feature]
            self.assertIn('concentration', feature_config, f"Drug feature {feature} saknar 'concentration'")
            self.assertIn('conversion_factor', feature_config, f"Drug feature {feature} saknar 'conversion_factor'")
        
        print("✅ T007 PASSED: Drug infusion features är korrekt definierade")
    
    def test_t007_ventilator_features(self):
        """
        Verifiera att ventilator features (6 st) är korrekt definierade
        """
        # Arrange
        ventilator_features = ['TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev']
        expected_count = 6
        
        # Act & Assert
        self.assertEqual(len(ventilator_features), expected_count,
                        f"Förväntade {expected_count} ventilator features")
        
        for feature in ventilator_features:
            self.assertIn(feature, MASTER_POC_TIMESERIES_FEATURES,
                         f"Ventilator feature {feature} saknas i feature mapping")
            
            # Verifiera att ventilator features har korrekt enheter
            feature_config = MASTER_POC_TIMESERIES_FEATURES[feature]
            if feature == 'TV':
                self.assertEqual(feature_config['unit'], 'ml/kg IBW',
                               f"TV ska ha enhet 'ml/kg IBW', fick {feature_config['unit']}")
            elif feature == 'PEEP':
                self.assertEqual(feature_config['unit'], 'cmH2O',
                               f"PEEP ska ha enhet 'cmH2O', fick {feature_config['unit']}")
        
        print("✅ T007 PASSED: Ventilator features är korrekt definierade")
    
    def test_t007_feature_mapping_validation(self):
        """
        Verifiera att hela feature mapping valideras korrekt
        """
        # Act & Assert
        try:
            result = validate_master_poc_feature_mapping()
            self.assertTrue(result, "Feature mapping validation ska returnera True")
            print("✅ T007 PASSED: Feature mapping validation fungerar korrekt")
        except AssertionError as e:
            self.fail(f"Feature mapping validation misslyckades: {e}")

if __name__ == '__main__':
    unittest.main()
