#!/usr/bin/env python3
"""
T009: Test Feature Order Consistency
Verifiera att feature-ordning följer Master POC spec (HR, BP_SYS, etc.)
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
MASTER_POC_TIMESERIES_FEATURES = master_poc_feature_mapping.MASTER_POC_TIMESERIES_FEATURES
MASTER_POC_STATIC_FEATURES = master_poc_feature_mapping.MASTER_POC_STATIC_FEATURES
get_timeseries_feature_order = master_poc_feature_mapping.get_timeseries_feature_order
get_static_feature_order = master_poc_feature_mapping.get_static_feature_order

class TestT009FeatureOrderConsistency(unittest.TestCase):
    """T009: Test Feature Order Consistency"""
    
    def test_t009_timeseries_feature_order_consistency(self):
        """
        T009: Test Feature Order Consistency
        Verifiera att feature-ordning följer Master POC spec (HR, BP_SYS, etc.)
        """
        # Arrange - Expected order enligt Master POC spec
        expected_timeseries_order = [
            'HR',           # 1 - Heart Rate
            'BP_SYS',       # 2 - Systolic Blood Pressure
            'BP_DIA',       # 3 - Diastolic Blood Pressure
            'BP_MAP',       # 4 - Mean Arterial Pressure
            'SPO2',         # 5 - SpO2
            'ETCO2',        # 6 - EtCO2
            'BIS',          # 7 - BIS
            'Propofol_INF', # 8 - Propofol
            'Remifentanil_INF', # 9 - Remifentanil
            'Noradrenalin_INF', # 10 - Noradrenalin
            'TV',           # 11 - Tidal Volume
            'PEEP',         # 12 - PEEP
            'FIO2',         # 13 - FiO2
            'RR',           # 14 - Respiratory Rate
            'etSEV',        # 15 - Expiratory sevoflurane pressure
            'inSev'         # 16 - Inspiratory sevoflurane pressure
        ]
        
        # Act
        actual_order = get_timeseries_feature_order()
        
        # Assert
        self.assertEqual(actual_order, expected_timeseries_order,
                        f"Timeseries feature ordning ska följa Master POC spec exakt")
        
        # Verifiera att varje feature har rätt order-nummer
        for i, feature in enumerate(expected_timeseries_order):
            expected_order_num = i + 1
            actual_order_num = MASTER_POC_TIMESERIES_FEATURES[feature]['order']
            self.assertEqual(actual_order_num, expected_order_num,
                            f"Feature {feature} ska ha order {expected_order_num}, fick {actual_order_num}")
        
        print("✅ T009 PASSED: Timeseries feature ordning är konsistent med Master POC spec")
    
    def test_t009_static_feature_order_consistency(self):
        """
        Verifiera att static feature-ordning följer Master POC spec
        """
        # Arrange - Expected order enligt Master POC spec
        expected_static_order = [
            'age',      # 17 - Age
            'sex',      # 18 - Sex
            'height',   # 19 - Height
            'weight',   # 20 - Weight
            'bmi',      # 21 - BMI
            'asa'       # 22 - ASA Score
        ]
        
        # Act
        actual_order = get_static_feature_order()
        
        # Assert
        self.assertEqual(actual_order, expected_static_order,
                        f"Static feature ordning ska följa Master POC spec exakt")
        
        # Verifiera att varje feature har rätt order-nummer
        for i, feature in enumerate(expected_static_order):
            expected_order_num = i + 17  # Static features börjar på 17
            actual_order_num = MASTER_POC_STATIC_FEATURES[feature]['order']
            self.assertEqual(actual_order_num, expected_order_num,
                            f"Feature {feature} ska ha order {expected_order_num}, fick {actual_order_num}")
        
        print("✅ T009 PASSED: Static feature ordning är konsistent med Master POC spec")
    
    def test_t009_vital_signs_order(self):
        """
        Verifiera att vital signs är i korrekt ordning (första 7 features)
        """
        # Arrange
        expected_vital_signs = ['HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS']
        
        # Act
        timeseries_order = get_timeseries_feature_order()
        actual_vital_signs = timeseries_order[:7]  # Första 7 features
        
        # Assert
        self.assertEqual(actual_vital_signs, expected_vital_signs,
                        f"Vital signs ska vara i korrekt ordning")
        
        print("✅ T009 PASSED: Vital signs är i korrekt ordning")
    
    def test_t009_drug_infusion_order(self):
        """
        Verifiera att drug infusions är i korrekt ordning (features 8-10)
        """
        # Arrange
        expected_drug_features = ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF']
        
        # Act
        timeseries_order = get_timeseries_feature_order()
        actual_drug_features = timeseries_order[7:10]  # Features 8-10 (index 7-9)
        
        # Assert
        self.assertEqual(actual_drug_features, expected_drug_features,
                        f"Drug infusion features ska vara i korrekt ordning")
        
        print("✅ T009 PASSED: Drug infusion features är i korrekt ordning")
    
    def test_t009_ventilator_order(self):
        """
        Verifiera att ventilator features är i korrekt ordning (features 11-16)
        """
        # Arrange
        expected_ventilator_features = ['TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev']
        
        # Act
        timeseries_order = get_timeseries_feature_order()
        actual_ventilator_features = timeseries_order[10:16]  # Features 11-16 (index 10-15)
        
        # Assert
        self.assertEqual(actual_ventilator_features, expected_ventilator_features,
                        f"Ventilator features ska vara i korrekt ordning")
        
        print("✅ T009 PASSED: Ventilator features är i korrekt ordning")
    
    def test_t009_no_duplicate_orders(self):
        """
        Verifiera att det inte finns duplicerade order-nummer
        """
        # Arrange
        all_orders = []
        
        # Act - Samla alla order-nummer
        for feature_config in MASTER_POC_TIMESERIES_FEATURES.values():
            all_orders.append(feature_config['order'])
        
        for feature_config in MASTER_POC_STATIC_FEATURES.values():
            all_orders.append(feature_config['order'])
        
        # Assert
        unique_orders = set(all_orders)
        self.assertEqual(len(unique_orders), len(all_orders),
                        f"Inga duplicerade order-nummer ska finnas. Hittade: {all_orders}")
        
        print("✅ T009 PASSED: Inga duplicerade order-nummer")

if __name__ == '__main__':
    unittest.main()
