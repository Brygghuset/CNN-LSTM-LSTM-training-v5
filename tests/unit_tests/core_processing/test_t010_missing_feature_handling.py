#!/usr/bin/env python3
"""
T010: Test Missing Feature Handling
Verifiera hantering när required features saknas i rådata
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår feature handler modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_feature_handler", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_feature_handler.py')
)
master_poc_feature_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_feature_handler)

# Använd modulen
MasterPOCFeatureHandler = master_poc_feature_handler.MasterPOCFeatureHandler
create_feature_handler = master_poc_feature_handler.create_feature_handler

class TestT010MissingFeatureHandling(unittest.TestCase):
    """T010: Test Missing Feature Handling"""
    
    def setUp(self):
        """Setup för varje test."""
        self.feature_handler = create_feature_handler()
    
    def test_t010_missing_timeseries_features(self):
        """
        T010: Test Missing Feature Handling
        Verifiera hantering när required timeseries features saknas i rådata
        """
        # Arrange - Data med saknade timeseries features
        incomplete_data = {
            'HR': 75,
            'BP_SYS': 120,
            'BP_DIA': 80,
            # BP_MAP saknas
            'SPO2': 98,
            # ETCO2 saknas
            'BIS': 45,
            'Propofol_INF': 2.5,
            'Remifentanil_INF': 0.1,
            'Noradrenalin_INF': 0.05,
            'TV': 6.5,
            'PEEP': 5,
            'FIO2': 40,
            'RR': 12,
            'etSEV': 1.5,
            'inSev': 1.8,
            'age': 65,
            'sex': 1,
            'height': 175,
            'weight': 80,
            'bmi': 26.1,
            'asa': 3
        }
        
        # Act
        is_valid, missing_features = self.feature_handler.validate_required_features(incomplete_data)
        
        # Assert
        self.assertFalse(is_valid, "Data med missing features ska vara invalid")
        self.assertIn('BP_MAP', missing_features, "BP_MAP ska identifieras som missing")
        self.assertIn('ETCO2', missing_features, "ETCO2 ska identifieras som missing")
        
        print("✅ T010 PASSED: Missing timeseries features identifieras korrekt")
    
    def test_t010_missing_static_features(self):
        """
        Verifiera hantering när required static features saknas i rådata
        """
        # Arrange - Data med saknade static features
        incomplete_data = {
            'HR': 75, 'BP_SYS': 120, 'BP_DIA': 80, 'BP_MAP': 93, 'SPO2': 98,
            'ETCO2': 4.5, 'BIS': 45, 'Propofol_INF': 2.5, 'Remifentanil_INF': 0.1,
            'Noradrenalin_INF': 0.05, 'TV': 6.5, 'PEEP': 5, 'FIO2': 40, 'RR': 12,
            'etSEV': 1.5, 'inSev': 1.8,
            'age': 65,
            # sex saknas
            'height': 175,
            # weight saknas
            'bmi': 26.1,
            'asa': 3
        }
        
        # Act
        is_valid, missing_features = self.feature_handler.validate_required_features(incomplete_data)
        
        # Assert
        self.assertFalse(is_valid, "Data med missing static features ska vara invalid")
        self.assertIn('sex', missing_features, "sex ska identifieras som missing")
        self.assertIn('weight', missing_features, "weight ska identifieras som missing")
        
        print("✅ T010 PASSED: Missing static features identifieras korrekt")
    
    def test_t010_handle_missing_features_with_defaults(self):
        """
        Verifiera att missing features fylls med default values
        """
        # Arrange - Data med saknade features
        incomplete_data = {
            'HR': 75,
            'BP_SYS': 120,
            'BP_DIA': 80,
            # BP_MAP saknas - ska beräknas
            'SPO2': 98,
            'ETCO2': 4.5,
            'BIS': 45,
            'Propofol_INF': 2.5,
            'Remifentanil_INF': 0.1,
            'Noradrenalin_INF': 0.05,
            'TV': 6.5,
            'PEEP': 5,
            'FIO2': 40,
            'RR': 12,
            'etSEV': 1.5,
            'inSev': 1.8,
            'age': 65,
            'sex': 1,
            'height': 175,
            'weight': 80,
            'bmi': 26.1,
            'asa': 3
        }
        
        # Act
        _, missing_features = self.feature_handler.validate_required_features(incomplete_data)
        handled_data = self.feature_handler.handle_missing_features(incomplete_data, missing_features)
        
        # Assert
        self.assertIn('BP_MAP', handled_data, "BP_MAP ska läggas till")
        self.assertIsNotNone(handled_data['BP_MAP'], "BP_MAP ska ha ett värde")
        
        # Verifiera att BP_MAP beräknas korrekt från BP_SYS och BP_DIA
        expected_map = (120 + 2 * 80) / 3  # (SYS + 2*DIA) / 3
        self.assertAlmostEqual(handled_data['BP_MAP'], expected_map, places=1,
                              msg=f"BP_MAP ska beräknas som {expected_map}")
        
        print("✅ T010 PASSED: Missing features fylls med korrekta default values")
    
    def test_t010_complete_data_validation(self):
        """
        Verifiera att komplett data valideras korrekt
        """
        # Arrange - Komplett data
        complete_data = {
            'HR': 75, 'BP_SYS': 120, 'BP_DIA': 80, 'BP_MAP': 93, 'SPO2': 98,
            'ETCO2': 4.5, 'BIS': 45, 'Propofol_INF': 2.5, 'Remifentanil_INF': 0.1,
            'Noradrenalin_INF': 0.05, 'TV': 6.5, 'PEEP': 5, 'FIO2': 40, 'RR': 12,
            'etSEV': 1.5, 'inSev': 1.8, 'age': 65, 'sex': 1, 'height': 175,
            'weight': 80, 'bmi': 26.1, 'asa': 3
        }
        
        # Act
        is_valid, missing_features = self.feature_handler.validate_required_features(complete_data)
        
        # Assert
        self.assertTrue(is_valid, "Komplett data ska vara valid")
        self.assertEqual(len(missing_features), 0, "Inga features ska saknas")
        
        print("✅ T010 PASSED: Komplett data valideras korrekt")
    
    def test_t010_feature_range_validation(self):
        """
        Verifiera validering av feature ranges
        """
        # Arrange - Data med värden utanför kliniska ranges
        out_of_range_data = {
            'HR': 250,  # För högt (max 200)
            'BP_SYS': 300,  # För högt (max 250)
            'SPO2': 50,  # För lågt (min 70)
            'age': 150,  # För högt (max 120)
            'asa': 8,  # För högt (max 6)
            'sex': 2,  # För högt (max 1)
        }
        
        # Act
        is_valid, out_of_range_features = self.feature_handler.validate_feature_ranges(out_of_range_data)
        
        # Assert
        self.assertFalse(is_valid, "Data med out-of-range värden ska vara invalid")
        self.assertGreater(len(out_of_range_features), 0, "Out-of-range features ska identifieras")
        
        # Verifiera att specifika features identifieras
        out_of_range_str = ' '.join(out_of_range_features)
        self.assertIn('HR=250', out_of_range_str, "HR=250 ska identifieras som out-of-range")
        self.assertIn('age=150', out_of_range_str, "age=150 ska identifieras som out-of-range")
        
        print("✅ T010 PASSED: Feature range validation fungerar korrekt")
    
    def test_t010_process_missing_features_complete_flow(self):
        """
        Verifiera komplett flow för missing feature handling
        """
        # Arrange - Data med både missing features och out-of-range värden
        problematic_data = {
            'HR': 75,
            'BP_SYS': 120,
            'BP_DIA': 80,
            # BP_MAP saknas
            'SPO2': 50,  # För lågt
            'ETCO2': 4.5,
            'BIS': 45,
            'Propofol_INF': 2.5,
            'Remifentanil_INF': 0.1,
            'Noradrenalin_INF': 0.05,
            'TV': 6.5,
            'PEEP': 5,
            'FIO2': 40,
            'RR': 12,
            'etSEV': 1.5,
            'inSev': 1.8,
            'age': 65,
            'sex': 1,
            'height': 175,
            'weight': 80,
            'bmi': 26.1,
            'asa': 3
        }
        
        # Act
        processed_data = self.feature_handler.process_missing_features(problematic_data)
        
        # Assert
        self.assertIn('BP_MAP', processed_data, "BP_MAP ska läggas till")
        self.assertIsNotNone(processed_data['BP_MAP'], "BP_MAP ska ha ett värde")
        
        # Verifiera att SPO2 fortfarande är out-of-range (detta är förväntat)
        self.assertEqual(processed_data['SPO2'], 50, "SPO2 ska behålla sitt ursprungliga värde")
        
        print("✅ T010 PASSED: Komplett missing feature handling flow fungerar")

if __name__ == '__main__':
    unittest.main()
