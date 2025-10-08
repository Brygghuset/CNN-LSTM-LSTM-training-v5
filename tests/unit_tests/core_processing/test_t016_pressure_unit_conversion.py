#!/usr/bin/env python3
"""
T016: Test Pressure Unit Conversion
Verifiera konvertering mellan kPa och cmH2O
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

class TestT016PressureUnitConversion(unittest.TestCase):
    """T016: Test Pressure Unit Conversion"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t016_kpa_to_cmh2o_conversion(self):
        """
        T016: Test Pressure Unit Conversion
        Verifiera konvertering från kPa till cmH2O
        """
        # Arrange
        kpa = 1.0  # 1 kPa
        
        # Expected: 1 kPa = 10.197 cmH2O (enligt conversion factor)
        expected_cmh2o = round(kpa * 10.197, 2)
        
        # Act
        result = self.converter.convert_kpa_to_cmh2o(kpa)
        
        # Assert
        self.assertEqual(result, expected_cmh2o,
                        f"kPa till cmH2O conversion ska ge {expected_cmh2o} cmH2O, fick {result}")
        
        print("✅ T016 PASSED: kPa till cmH2O conversion fungerar korrekt")
    
    def test_t016_cmh2o_to_kpa_conversion(self):
        """
        Verifiera konvertering från cmH2O till kPa
        """
        # Arrange
        cmh2o = 10.197  # 10.197 cmH2O
        
        # Expected: 10.197 cmH2O = 1.0 kPa (enligt conversion factor)
        expected_kpa = round(cmh2o * 0.0981, 3)
        
        # Act
        result = self.converter.convert_cmh2o_to_kpa(cmh2o)
        
        # Assert
        self.assertEqual(result, expected_kpa,
                        f"cmH2O till kPa conversion ska ge {expected_kpa} kPa, fick {result}")
        
        print("✅ T016 PASSED: cmH2O till kPa conversion fungerar korrekt")
    
    def test_t016_pressure_conversion_zero(self):
        """
        Verifiera pressure conversion med zero värden
        """
        # Arrange
        zero_kpa = 0.0
        zero_cmh2o = 0.0
        
        # Act
        result_kpa_to_cmh2o = self.converter.convert_kpa_to_cmh2o(zero_kpa)
        result_cmh2o_to_kpa = self.converter.convert_cmh2o_to_kpa(zero_cmh2o)
        
        # Assert
        self.assertEqual(result_kpa_to_cmh2o, 0.0, "Zero kPa ska ge 0 cmH2O")
        self.assertEqual(result_cmh2o_to_kpa, 0.0, "Zero cmH2O ska ge 0 kPa")
        
        print("✅ T016 PASSED: Zero pressure conversion fungerar korrekt")
    
    def test_t016_pressure_conversion_roundtrip(self):
        """
        Verifiera att konvertering fungerar båda vägar (roundtrip)
        """
        # Arrange
        original_kpa = 2.5
        
        # Act
        # kPa -> cmH2O -> kPa
        cmh2o = self.converter.convert_kpa_to_cmh2o(original_kpa)
        back_to_kpa = self.converter.convert_cmh2o_to_kpa(cmh2o)
        
        # Assert
        self.assertAlmostEqual(back_to_kpa, original_kpa, places=2,
                              msg=f"Roundtrip conversion ska ge tillbaka {original_kpa} kPa")
        
        print("✅ T016 PASSED: Pressure conversion roundtrip fungerar korrekt")
    
    def test_t016_pressure_conversion_clinical_ranges(self):
        """
        Verifiera pressure conversion med kliniska ranges
        """
        # Arrange
        test_cases = [
            # (kPa, expected_cmH2O, description)
            (0.5, 5.10, "Lågt PEEP"),      # 0.5 kPa = ~5.1 cmH2O
            (1.0, 10.20, "Normal PEEP"),   # 1.0 kPa = ~10.2 cmH2O
            (2.0, 20.39, "Högt PEEP"),     # 2.0 kPa = ~20.4 cmH2O
            (3.0, 30.59, "Mycket högt PEEP"), # 3.0 kPa = ~30.6 cmH2O
        ]
        
        # Act & Assert
        for kpa, expected_cmh2o, description in test_cases:
            result = self.converter.convert_kpa_to_cmh2o(kpa)
            self.assertAlmostEqual(result, expected_cmh2o, places=1,
                                 msg=f"{description}: {kpa} kPa ska ge ~{expected_cmh2o} cmH2O")
        
        print("✅ T016 PASSED: Pressure conversion med kliniska ranges fungerar korrekt")
    
    def test_t016_pressure_conversion_etsev_range(self):
        """
        Verifiera pressure conversion för etSEV (end-tidal sevoflurane) range
        """
        # Arrange
        # etSEV range: 0-6 kPa enligt Master POC spec
        etsev_kpa_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        # Act & Assert
        for kpa in etsev_kpa_values:
            cmh2o = self.converter.convert_kpa_to_cmh2o(kpa)
            expected_cmh2o = round(kpa * 10.197, 2)
            
            self.assertEqual(cmh2o, expected_cmh2o,
                           f"etSEV {kpa} kPa ska ge {expected_cmh2o} cmH2O")
            
            # Verifiera att värdet ligger inom klinisk range (kPa range)
            is_in_range = self.converter.validate_conversion_ranges(kpa, 'etSEV')
            self.assertTrue(is_in_range,
                          f"etSEV {kpa} kPa ska vara inom klinisk range")
        
        print("✅ T016 PASSED: Pressure conversion för etSEV range fungerar korrekt")
    
    def test_t016_pressure_conversion_insev_range(self):
        """
        Verifiera pressure conversion för inSev (inspired sevoflurane) range
        """
        # Arrange
        # inSev range: 0-8 kPa enligt Master POC spec
        insev_kpa_values = [0.0, 2.0, 4.0, 6.0, 8.0]
        
        # Act & Assert
        for kpa in insev_kpa_values:
            cmh2o = self.converter.convert_kpa_to_cmh2o(kpa)
            expected_cmh2o = round(kpa * 10.197, 2)
            
            self.assertEqual(cmh2o, expected_cmh2o,
                           f"inSev {kpa} kPa ska ge {expected_cmh2o} cmH2O")
            
            # Verifiera att värdet ligger inom klinisk range (kPa range)
            is_in_range = self.converter.validate_conversion_ranges(kpa, 'inSev')
            self.assertTrue(is_in_range,
                          f"inSev {kpa} kPa ska vara inom klinisk range")
        
        print("✅ T016 PASSED: Pressure conversion för inSev range fungerar korrekt")
    
    def test_t016_pressure_conversion_precision(self):
        """
        Verifiera att pressure conversion har korrekt precision
        """
        # Arrange
        kpa = 1.5
        cmh2o = 15.3
        
        # Act
        result_kpa_to_cmh2o = self.converter.convert_kpa_to_cmh2o(kpa)
        result_cmh2o_to_kpa = self.converter.convert_cmh2o_to_kpa(cmh2o)
        
        # Assert
        # Verifiera precision för kPa -> cmH2O (2 decimaler)
        decimal_places_kpa_to_cmh2o = len(str(result_kpa_to_cmh2o).split('.')[-1]) if '.' in str(result_kpa_to_cmh2o) else 0
        self.assertLessEqual(decimal_places_kpa_to_cmh2o, 2,
                            f"kPa->cmH2O conversion ska ha max 2 decimaler, fick {decimal_places_kpa_to_cmh2o}")
        
        # Verifiera precision för cmH2O -> kPa (3 decimaler)
        decimal_places_cmh2o_to_kpa = len(str(result_cmh2o_to_kpa).split('.')[-1]) if '.' in str(result_cmh2o_to_kpa) else 0
        self.assertLessEqual(decimal_places_cmh2o_to_kpa, 3,
                            f"cmH2O->kPa conversion ska ha max 3 decimaler, fick {decimal_places_cmh2o_to_kpa}")
        
        print("✅ T016 PASSED: Pressure conversion har korrekt precision")
    
    def test_t016_conversion_factors_validation(self):
        """
        Verifiera att conversion factors är korrekta
        """
        # Arrange
        expected_kpa_to_cmh2o = 10.197
        expected_cmh2o_to_kpa = 0.0981
        
        # Act
        actual_kpa_to_cmh2o = self.converter.conversion_factors['kpa_to_cmh2o']
        actual_cmh2o_to_kpa = self.converter.conversion_factors['cmh2o_to_kpa']
        
        # Assert
        self.assertEqual(actual_kpa_to_cmh2o, expected_kpa_to_cmh2o,
                        f"kPa->cmH2O conversion factor ska vara {expected_kpa_to_cmh2o}")
        self.assertEqual(actual_cmh2o_to_kpa, expected_cmh2o_to_kpa,
                        f"cmH2O->kPa conversion factor ska vara {expected_cmh2o_to_kpa}")
        
        print("✅ T016 PASSED: Pressure conversion factors är korrekta")

if __name__ == '__main__':
    unittest.main()
