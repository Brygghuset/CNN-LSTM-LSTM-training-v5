#!/usr/bin/env python3
"""
T012: Test Propofol Conversion
Verifiera konvertering från mL/h till mg/kg/h med 20mg/ml koncentration
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

class TestT012PropofolConversion(unittest.TestCase):
    """T012: Test Propofol Conversion"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t012_propofol_conversion_basic(self):
        """
        T012: Test Propofol Conversion
        Verifiera konvertering från mL/h till mg/kg/h med 20mg/ml koncentration
        """
        # Arrange
        ml_h = 5.0  # 5 mL/h
        weight_kg = 70.0  # 70 kg patient
        concentration_mg_ml = 20.0  # 20 mg/ml enligt Master POC spec
        
        # Expected: (5 mL/h * 20 mg/ml) / 70 kg = 100 mg/h / 70 kg = 1.429 mg/kg/h
        expected_mg_kg_h = round((ml_h * concentration_mg_ml) / weight_kg, 3)
        
        # Act
        result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, expected_mg_kg_h,
                        f"Propofol conversion ska ge {expected_mg_kg_h} mg/kg/h, fick {result}")
        
        print("✅ T012 PASSED: Basic Propofol conversion fungerar korrekt")
    
    def test_t012_propofol_conversion_zero_infusion(self):
        """
        Verifiera Propofol conversion med zero infusion rate
        """
        # Arrange
        ml_h = 0.0
        weight_kg = 70.0
        
        # Act
        result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, 0.0, "Zero infusion ska ge 0 mg/kg/h")
        
        print("✅ T012 PASSED: Zero Propofol infusion hanteras korrekt")
    
    def test_t012_propofol_conversion_high_infusion(self):
        """
        Verifiera Propofol conversion med hög infusion rate
        """
        # Arrange
        ml_h = 15.0  # Hög infusion rate
        weight_kg = 70.0
        
        # Expected: (15 mL/h * 20 mg/ml) / 70 kg = 300 mg/h / 70 kg = 4.286 mg/kg/h
        expected_mg_kg_h = round((ml_h * 20.0) / weight_kg, 3)
        
        # Act
        result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, expected_mg_kg_h,
                        f"Hög Propofol infusion ska ge {expected_mg_kg_h} mg/kg/h")
        
        print("✅ T012 PASSED: Hög Propofol infusion konverteras korrekt")
    
    def test_t012_propofol_conversion_different_weights(self):
        """
        Verifiera Propofol conversion med olika patientvikter
        """
        # Arrange
        ml_h = 10.0
        test_cases = [
            (50.0, 4.0),   # 50 kg -> 4.0 mg/kg/h
            (80.0, 2.5),   # 80 kg -> 2.5 mg/kg/h
            (100.0, 2.0),  # 100 kg -> 2.0 mg/kg/h
        ]
        
        # Act & Assert
        for weight_kg, expected_mg_kg_h in test_cases:
            result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
            self.assertEqual(result, expected_mg_kg_h,
                            f"Propofol conversion för {weight_kg} kg ska ge {expected_mg_kg_h} mg/kg/h")
        
        print("✅ T012 PASSED: Propofol conversion med olika vikter fungerar korrekt")
    
    def test_t012_propofol_conversion_edge_case_weights(self):
        """
        Verifiera Propofol conversion med extrema vikter
        """
        # Arrange
        ml_h = 5.0
        
        # Act & Assert
        # Mycket låg vikt
        result_low = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, 20.0)
        expected_low = round((5.0 * 20.0) / 20.0, 3)  # 5.0 mg/kg/h
        self.assertEqual(result_low, expected_low,
                        f"Propofol conversion för 20 kg ska ge {expected_low} mg/kg/h")
        
        # Mycket hög vikt
        result_high = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, 200.0)
        expected_high = round((5.0 * 20.0) / 200.0, 3)  # 0.5 mg/kg/h
        self.assertEqual(result_high, expected_high,
                        f"Propofol conversion för 200 kg ska ge {expected_high} mg/kg/h")
        
        print("✅ T012 PASSED: Propofol conversion med extrema vikter fungerar korrekt")
    
    def test_t012_propofol_conversion_invalid_weight(self):
        """
        Verifiera Propofol conversion med ogiltig vikt
        """
        # Arrange
        ml_h = 5.0
        
        # Act & Assert
        # Negativ vikt
        result_negative = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, -10.0)
        self.assertEqual(result_negative, 0.0, "Negativ vikt ska ge 0 mg/kg/h")
        
        # Zero vikt
        result_zero = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, 0.0)
        self.assertEqual(result_zero, 0.0, "Zero vikt ska ge 0 mg/kg/h")
        
        print("✅ T012 PASSED: Propofol conversion med ogiltig vikt hanteras korrekt")
    
    def test_t012_propofol_concentration_validation(self):
        """
        Verifiera att Propofol concentration är korrekt enligt Master POC spec
        """
        # Arrange
        expected_concentration = 20.0  # mg/ml enligt Master POC spec
        
        # Act
        actual_concentration = self.converter.drug_concentrations['Propofol_INF']
        
        # Assert
        self.assertEqual(actual_concentration, expected_concentration,
                        f"Propofol concentration ska vara {expected_concentration} mg/ml enligt Master POC spec")
        
        print("✅ T012 PASSED: Propofol concentration är korrekt enligt Master POC spec")
    
    def test_t012_propofol_range_validation(self):
        """
        Verifiera att konverterade Propofol värden ligger inom klinisk range
        """
        # Arrange
        test_cases = [
            (1.0, 70.0, True),   # 0.286 mg/kg/h - inom range (0-12)
            (50.0, 70.0, False), # 14.286 mg/kg/h - över range (0-12)
            (0.0, 70.0, True),   # 0 mg/kg/h - inom range
        ]
        
        # Act & Assert
        for ml_h, weight_kg, should_be_in_range in test_cases:
            result = self.converter.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
            is_in_range = self.converter.validate_conversion_ranges(result, 'Propofol_INF')
            
            if should_be_in_range:
                self.assertTrue(is_in_range,
                               f"Propofol {result} mg/kg/h ska vara inom klinisk range")
            else:
                self.assertFalse(is_in_range,
                                f"Propofol {result} mg/kg/h ska vara utanför klinisk range")
        
        print("✅ T012 PASSED: Propofol range validation fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
