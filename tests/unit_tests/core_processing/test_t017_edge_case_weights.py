#!/usr/bin/env python3
"""
T017: Test Edge Case Weights
Verifiera unit conversion med extrema vikter (20kg, 200kg)
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår unit converter modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_unit_converter", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_unit_converter.py')
)
master_poc_unit_converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_unit_converter)

# Använd modulen
MasterPOCUnitConverter = master_poc_unit_converter.MasterPOCUnitConverter
create_unit_converter = master_poc_unit_converter.create_unit_converter

class TestT017EdgeCaseWeights(unittest.TestCase):
    """T017: Test Edge Case Weights"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t017_propofol_extreme_weights(self):
        """
        T017: Test Edge Case Weights
        Verifiera Propofol conversion med extrema vikter (20kg, 200kg)
        """
        # Arrange
        ml_h = 5.0  # 5 mL/h infusion
        extreme_weights = [20.0, 200.0]  # kg
        
        # Expected calculations:
        # 20 kg: (5 * 20) / 20 = 100 / 20 = 5.0 mg/kg/h
        # 200 kg: (5 * 20) / 200 = 100 / 200 = 0.5 mg/kg/h
        expected_results = [5.0, 0.5]
        
        # Act & Assert
        for weight_kg, expected_mg_kg_h in zip(extreme_weights, expected_results):
            result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
            self.assertEqual(result, expected_mg_kg_h,
                           f"Propofol conversion för {weight_kg} kg ska ge {expected_mg_kg_h} mg/kg/h")
        
        print("✅ T017 PASSED: Propofol conversion med extrema vikter fungerar korrekt")
    
    def test_t017_remifentanil_extreme_weights(self):
        """
        Verifiera Remifentanil conversion med extrema vikter
        """
        # Arrange
        ml_h = 2.0  # 2 mL/h infusion
        extreme_weights = [20.0, 200.0]  # kg
        
        # Expected calculations:
        # 20 kg: (2 * 20) / 60 / 20 = 40 / 60 / 20 = 0.0333 mcg/kg/min
        # 200 kg: (2 * 20) / 60 / 200 = 40 / 60 / 200 = 0.0033 mcg/kg/min
        expected_results = [0.0333, 0.0033]
        
        # Act & Assert
        for weight_kg, expected_mcg_kg_min in zip(extreme_weights, expected_results):
            result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
            self.assertEqual(result, expected_mcg_kg_min,
                           f"Remifentanil conversion för {weight_kg} kg ska ge {expected_mcg_kg_min} mcg/kg/min")
        
        print("✅ T017 PASSED: Remifentanil conversion med extrema vikter fungerar korrekt")
    
    def test_t017_noradrenalin_extreme_weights(self):
        """
        Verifiera Noradrenalin conversion med extrema vikter
        """
        # Arrange
        ml_h = 1.0  # 1 mL/h infusion
        extreme_weights = [20.0, 200.0]  # kg
        
        # Expected calculations:
        # 20 kg: (1 * 0.1) / 60 / 20 = 0.1 / 60 / 20 = 0.0001 mcg/kg/min
        # 200 kg: (1 * 0.1) / 60 / 200 = 0.1 / 60 / 200 = 0.0000 mcg/kg/min
        expected_results = [0.0001, 0.0000]
        
        # Act & Assert
        for weight_kg, expected_mcg_kg_min in zip(extreme_weights, expected_results):
            result = self.converter.convert_noradrenalin_ml_h_to_mcg_kg_min(ml_h, weight_kg)
            self.assertEqual(result, expected_mcg_kg_min,
                           f"Noradrenalin conversion för {weight_kg} kg ska ge {expected_mcg_kg_min} mcg/kg/min")
        
        print("✅ T017 PASSED: Noradrenalin conversion med extrema vikter fungerar korrekt")
    
    def test_t017_tidal_volume_extreme_weights(self):
        """
        Verifiera Tidal Volume conversion med extrema höjder (som ger extrema IBW)
        """
        # Arrange
        tv_ml = 400.0  # 400 ml tidal volume
        
        # Testa med extrema höjder som ger extrema IBW
        # Kort person (150 cm man): IBW = 50 kg (minimum)
        # Lång person (200 cm man): IBW = 93.1 kg
        extreme_cases = [
            (150.0, 1, 8.0),   # Kort man: 400 / 50 = 8.0 ml/kg IBW
            (200.0, 1, 4.3),    # Lång man: 400 / 93.1 = 4.30 ml/kg IBW
        ]
        
        # Act & Assert
        for height_cm, sex, expected_tv_ml_kg_ibw in extreme_cases:
            result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
            self.assertEqual(result, expected_tv_ml_kg_ibw,
                           f"TV conversion för {height_cm} cm ska ge {expected_tv_ml_kg_ibw} ml/kg IBW")
        
        print("✅ T017 PASSED: Tidal Volume conversion med extrema höjder fungerar korrekt")
    
    def test_t017_drug_conversion_very_low_weight(self):
        """
        Verifiera drug conversion med mycket låg vikt (10 kg)
        """
        # Arrange
        ml_h = 3.0
        very_low_weight = 10.0  # kg
        
        # Act & Assert
        # Propofol: (3 * 20) / 10 = 60 / 10 = 6.0 mg/kg/h
        propofol_result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, very_low_weight)
        self.assertEqual(propofol_result, 6.0,
                        f"Propofol för {very_low_weight} kg ska ge 6.0 mg/kg/h")
        
        # Remifentanil: (3 * 20) / 60 / 10 = 60 / 60 / 10 = 0.1 mcg/kg/min
        remifentanil_result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, very_low_weight)
        self.assertEqual(remifentanil_result, 0.1,
                        f"Remifentanil för {very_low_weight} kg ska ge 0.1 mcg/kg/min")
        
        print("✅ T017 PASSED: Drug conversion med mycket låg vikt fungerar korrekt")
    
    def test_t017_drug_conversion_very_high_weight(self):
        """
        Verifiera drug conversion med mycket hög vikt (300 kg)
        """
        # Arrange
        ml_h = 10.0
        very_high_weight = 300.0  # kg
        
        # Act & Assert
        # Propofol: (10 * 20) / 300 = 200 / 300 = 0.667 mg/kg/h
        propofol_result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, very_high_weight)
        self.assertEqual(propofol_result, 0.667,
                        f"Propofol för {very_high_weight} kg ska ge 0.667 mg/kg/h")
        
        # Remifentanil: (10 * 20) / 60 / 300 = 200 / 60 / 300 = 0.0111 mcg/kg/min
        remifentanil_result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, very_high_weight)
        self.assertEqual(remifentanil_result, 0.0111,
                        f"Remifentanil för {very_high_weight} kg ska ge 0.0111 mcg/kg/min")
        
        print("✅ T017 PASSED: Drug conversion med mycket hög vikt fungerar korrekt")
    
    def test_t017_edge_case_weight_boundaries(self):
        """
        Verifiera conversion vid viktgränser
        """
        # Arrange
        ml_h = 1.0
        boundary_weights = [0.1, 0.5, 1.0, 500.0, 1000.0]  # kg
        
        # Act & Assert
        for weight_kg in boundary_weights:
            # Testa att conversion fungerar utan fel
            try:
                propofol_result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
                remifentanil_result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
                
                # Verifiera att resultaten är positiva (förutom för ogiltiga vikter)
                if weight_kg > 0:
                    self.assertGreater(propofol_result, 0,
                                     f"Propofol för {weight_kg} kg ska ge positivt resultat")
                    self.assertGreater(remifentanil_result, 0,
                                     f"Remifentanil för {weight_kg} kg ska ge positivt resultat")
                else:
                    self.assertEqual(propofol_result, 0,
                                   f"Propofol för {weight_kg} kg ska ge 0")
                    self.assertEqual(remifentanil_result, 0,
                                   f"Remifentanil för {weight_kg} kg ska ge 0")
                
            except Exception as e:
                self.fail(f"Conversion för {weight_kg} kg gav fel: {e}")
        
        print("✅ T017 PASSED: Drug conversion vid viktgränser fungerar korrekt")
    
    def test_t017_extreme_weight_clinical_ranges(self):
        """
        Verifiera att extrema vikter ger värden inom eller utanför kliniska ranges
        """
        # Arrange
        ml_h = 5.0
        
        # Act & Assert
        # Mycket låg vikt (20 kg) - ska ge höga mg/kg/h värden
        propofol_low = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, 20.0)
        is_propofol_low_in_range = self.converter.validate_conversion_ranges(propofol_low, 'Propofol_INF')
        self.assertTrue(is_propofol_low_in_range,
                      f"Propofol {propofol_low} mg/kg/h för 20 kg ska vara inom klinisk range")
        
        # Mycket hög vikt (200 kg) - ska ge låga mg/kg/h värden
        propofol_high = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, 200.0)
        is_propofol_high_in_range = self.converter.validate_conversion_ranges(propofol_high, 'Propofol_INF')
        self.assertTrue(is_propofol_high_in_range,
                      f"Propofol {propofol_high} mg/kg/h för 200 kg ska vara inom klinisk range")
        
        print("✅ T017 PASSED: Extrema vikter ger värden inom kliniska ranges")
    
    def test_t017_edge_case_precision(self):
        """
        Verifiera att precision är korrekt även för extrema vikter
        """
        # Arrange
        ml_h = 1.0
        extreme_weights = [20.0, 200.0]
        
        # Act & Assert
        for weight_kg in extreme_weights:
            # Propofol precision (3 decimaler)
            propofol_result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
            propofol_decimal_places = len(str(propofol_result).split('.')[-1]) if '.' in str(propofol_result) else 0
            self.assertLessEqual(propofol_decimal_places, 3,
                               f"Propofol precision för {weight_kg} kg ska vara max 3 decimaler")
            
            # Remifentanil precision (4 decimaler)
            remifentanil_result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
            remifentanil_decimal_places = len(str(remifentanil_result).split('.')[-1]) if '.' in str(remifentanil_result) else 0
            self.assertLessEqual(remifentanil_decimal_places, 4,
                               f"Remifentanil precision för {weight_kg} kg ska vara max 4 decimaler")
        
        print("✅ T017 PASSED: Precision är korrekt även för extrema vikter")

if __name__ == '__main__':
    unittest.main()
