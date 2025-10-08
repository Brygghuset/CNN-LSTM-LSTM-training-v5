#!/usr/bin/env python3
"""
T013: Test Remifentanil Conversion
Verifiera konvertering från mL/h till mcg/kg/min med 20mcg/ml koncentration
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

class TestT013RemifentanilConversion(unittest.TestCase):
    """T013: Test Remifentanil Conversion"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t013_remifentanil_conversion_basic(self):
        """
        T013: Test Remifentanil Conversion
        Verifiera konvertering från mL/h till mcg/kg/min med 20mcg/ml koncentration
        """
        # Arrange
        ml_h = 2.0  # 2 mL/h
        weight_kg = 70.0  # 70 kg patient
        concentration_mcg_ml = 20.0  # 20 mcg/ml enligt Master POC spec
        
        # Expected: (2 mL/h * 20 mcg/ml) / 60 min/h / 70 kg = 40 mcg/h / 60 / 70 kg = 0.0095 mcg/kg/min
        expected_mcg_kg_min = round((ml_h * concentration_mcg_ml) / 60 / weight_kg, 4)
        
        # Act
        result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, expected_mcg_kg_min,
                        f"Remifentanil conversion ska ge {expected_mcg_kg_min} mcg/kg/min, fick {result}")
        
        print("✅ T013 PASSED: Basic Remifentanil conversion fungerar korrekt")
    
    def test_t013_remifentanil_conversion_zero_infusion(self):
        """
        Verifiera Remifentanil conversion med zero infusion rate
        """
        # Arrange
        ml_h = 0.0
        weight_kg = 70.0
        
        # Act
        result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, 0.0, "Zero infusion ska ge 0 mcg/kg/min")
        
        print("✅ T013 PASSED: Zero Remifentanil infusion hanteras korrekt")
    
    def test_t013_remifentanil_conversion_high_infusion(self):
        """
        Verifiera Remifentanil conversion med hög infusion rate
        """
        # Arrange
        ml_h = 5.0  # Hög infusion rate
        weight_kg = 70.0
        
        # Expected: (5 mL/h * 20 mcg/ml) / 60 min/h / 70 kg = 100 mcg/h / 60 / 70 kg = 0.0238 mcg/kg/min
        expected_mcg_kg_min = round((ml_h * 20.0) / 60 / weight_kg, 4)
        
        # Act
        result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        
        # Assert
        self.assertEqual(result, expected_mcg_kg_min,
                        f"Hög Remifentanil infusion ska ge {expected_mcg_kg_min} mcg/kg/min")
        
        print("✅ T013 PASSED: Hög Remifentanil infusion konverteras korrekt")
    
    def test_t013_remifentanil_conversion_different_weights(self):
        """
        Verifiera Remifentanil conversion med olika patientvikter
        """
        # Arrange
        ml_h = 3.0
        test_cases = [
            (50.0, 0.0200),   # 50 kg -> 0.0200 mcg/kg/min
            (80.0, 0.0125),   # 80 kg -> 0.0125 mcg/kg/min
            (100.0, 0.0100),  # 100 kg -> 0.0100 mcg/kg/min
        ]
        
        # Act & Assert
        for weight_kg, expected_mcg_kg_min in test_cases:
            result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
            self.assertEqual(result, expected_mcg_kg_min,
                            f"Remifentanil conversion för {weight_kg} kg ska ge {expected_mcg_kg_min} mcg/kg/min")
        
        print("✅ T013 PASSED: Remifentanil conversion med olika vikter fungerar korrekt")
    
    def test_t013_remifentanil_conversion_edge_case_weights(self):
        """
        Verifiera Remifentanil conversion med extrema vikter
        """
        # Arrange
        ml_h = 2.0
        
        # Act & Assert
        # Mycket låg vikt
        result_low = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, 20.0)
        expected_low = round((2.0 * 20.0) / 60 / 20.0, 4)  # 0.0333 mcg/kg/min
        self.assertEqual(result_low, expected_low,
                        f"Remifentanil conversion för 20 kg ska ge {expected_low} mcg/kg/min")
        
        # Mycket hög vikt
        result_high = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, 200.0)
        expected_high = round((2.0 * 20.0) / 60 / 200.0, 4)  # 0.0033 mcg/kg/min
        self.assertEqual(result_high, expected_high,
                        f"Remifentanil conversion för 200 kg ska ge {expected_high} mcg/kg/min")
        
        print("✅ T013 PASSED: Remifentanil conversion med extrema vikter fungerar korrekt")
    
    def test_t013_remifentanil_conversion_invalid_weight(self):
        """
        Verifiera Remifentanil conversion med ogiltig vikt
        """
        # Arrange
        ml_h = 2.0
        
        # Act & Assert
        # Negativ vikt
        result_negative = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, -10.0)
        self.assertEqual(result_negative, 0.0, "Negativ vikt ska ge 0 mcg/kg/min")
        
        # Zero vikt
        result_zero = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, 0.0)
        self.assertEqual(result_zero, 0.0, "Zero vikt ska ge 0 mcg/kg/min")
        
        print("✅ T013 PASSED: Remifentanil conversion med ogiltig vikt hanteras korrekt")
    
    def test_t013_remifentanil_concentration_validation(self):
        """
        Verifiera att Remifentanil concentration är korrekt enligt Master POC spec
        """
        # Arrange
        expected_concentration = 20.0  # mcg/ml enligt Master POC spec
        
        # Act
        actual_concentration = self.converter.drug_concentrations['Remifentanil_INF']
        
        # Assert
        self.assertEqual(actual_concentration, expected_concentration,
                        f"Remifentanil concentration ska vara {expected_concentration} mcg/ml enligt Master POC spec")
        
        print("✅ T013 PASSED: Remifentanil concentration är korrekt enligt Master POC spec")
    
    def test_t013_remifentanil_range_validation(self):
        """
        Verifiera att konverterade Remifentanil värden ligger inom klinisk range
        """
        # Arrange
        test_cases = [
            (1.0, 70.0, True),   # 0.0048 mcg/kg/min - inom range (0-0.8)
            (20.0, 70.0, True),  # 0.0952 mcg/kg/min - inom range (0-0.8)
            (0.0, 70.0, True),   # 0 mcg/kg/min - inom range
            (200.0, 50.0, False), # 1.3333 mcg/kg/min - över range (0-0.8)
        ]
        
        # Act & Assert
        for ml_h, weight_kg, should_be_in_range in test_cases:
            result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
            is_in_range = self.converter.validate_conversion_ranges(result, 'Remifentanil_INF')
            
            if should_be_in_range:
                self.assertTrue(is_in_range,
                               f"Remifentanil {result} mcg/kg/min ska vara inom klinisk range")
            else:
                self.assertFalse(is_in_range,
                                f"Remifentanil {result} mcg/kg/min ska vara utanför klinisk range")
        
        print("✅ T013 PASSED: Remifentanil range validation fungerar korrekt")
    
    def test_t013_remifentanil_precision(self):
        """
        Verifiera att Remifentanil conversion har korrekt precision (4 decimaler)
        """
        # Arrange
        ml_h = 1.5
        weight_kg = 75.0
        
        # Act
        result = self.converter.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        
        # Assert
        # Verifiera att resultatet har korrekt precision
        decimal_places = len(str(result).split('.')[-1]) if '.' in str(result) else 0
        self.assertLessEqual(decimal_places, 4, 
                            f"Remifentanil conversion ska ha max 4 decimaler, fick {decimal_places}")
        
        print("✅ T013 PASSED: Remifentanil conversion har korrekt precision")

if __name__ == '__main__':
    unittest.main()
