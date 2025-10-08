#!/usr/bin/env python3
"""
T015: Test Tidal Volume Conversion
Verifiera konvertering till ml/kg IBW
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

class TestT015TidalVolumeConversion(unittest.TestCase):
    """T015: Test Tidal Volume Conversion"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t015_tidal_volume_conversion_basic(self):
        """
        T015: Test Tidal Volume Conversion
        Verifiera konvertering till ml/kg IBW
        """
        # Arrange
        tv_ml = 500.0  # 500 ml tidal volume
        height_cm = 175.0  # 175 cm man
        sex = 1  # Male
        
        # Expected IBW för 175 cm man: 50 + 2.3 * (68.9 - 60) = 70.47 kg
        # Expected TV: 500 ml / 70.47 kg = 7.09 ml/kg IBW
        expected_ibw = round(50 + 2.3 * (175.0 / 2.54 - 60), 1)
        expected_tv_ml_kg_ibw = round(500.0 / expected_ibw, 2)
        
        # Act
        result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_tv_ml_kg_ibw,
                        f"TV conversion ska ge {expected_tv_ml_kg_ibw} ml/kg IBW, fick {result}")
        
        print("✅ T015 PASSED: Basic Tidal Volume conversion fungerar korrekt")
    
    def test_t015_tidal_volume_conversion_female(self):
        """
        Verifiera Tidal Volume conversion för kvinna
        """
        # Arrange
        tv_ml = 400.0  # 400 ml tidal volume
        height_cm = 165.0  # 165 cm kvinna
        sex = -1  # Female
        
        # Expected IBW för 165 cm kvinna: 45.5 + 2.3 * (64.96 - 60) = 56.91 kg
        # Expected TV: 400 ml / 56.91 kg = 7.03 ml/kg IBW
        expected_ibw = round(45.5 + 2.3 * (165.0 / 2.54 - 60), 1)
        expected_tv_ml_kg_ibw = round(400.0 / expected_ibw, 2)
        
        # Act
        result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_tv_ml_kg_ibw,
                        f"TV conversion för kvinna ska ge {expected_tv_ml_kg_ibw} ml/kg IBW")
        
        print("✅ T015 PASSED: Tidal Volume conversion för kvinna fungerar korrekt")
    
    def test_t015_tidal_volume_conversion_zero_volume(self):
        """
        Verifiera Tidal Volume conversion med zero volume
        """
        # Arrange
        tv_ml = 0.0
        height_cm = 175.0
        sex = 1
        
        # Act
        result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
        
        # Assert
        self.assertEqual(result, 0.0, "Zero tidal volume ska ge 0 ml/kg IBW")
        
        print("✅ T015 PASSED: Zero Tidal Volume hanteras korrekt")
    
    def test_t015_tidal_volume_conversion_different_volumes(self):
        """
        Verifiera Tidal Volume conversion med olika volymer
        """
        # Arrange
        height_cm = 170.0
        sex = 1  # Male
        test_cases = [
            (300.0, 4.55),   # 300 ml -> ~4.55 ml/kg IBW
            (600.0, 9.10),   # 600 ml -> ~9.10 ml/kg IBW
            (800.0, 12.13),  # 800 ml -> ~12.13 ml/kg IBW
        ]
        
        # Act & Assert
        for tv_ml, expected_tv_ml_kg_ibw in test_cases:
            result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
            self.assertAlmostEqual(result, expected_tv_ml_kg_ibw, places=1,
                                 msg=f"TV {tv_ml} ml ska ge ~{expected_tv_ml_kg_ibw} ml/kg IBW")
        
        print("✅ T015 PASSED: Tidal Volume conversion med olika volymer fungerar korrekt")
    
    def test_t015_tidal_volume_conversion_different_heights(self):
        """
        Verifiera Tidal Volume conversion med olika höjder
        """
        # Arrange
        tv_ml = 500.0
        sex = 1  # Male
        test_cases = [
            (160.0, 8.79),   # Kort man -> högre ml/kg IBW (beräknat)
            (180.0, 6.67),   # Lång man -> lägre ml/kg IBW (beräknat)
            (200.0, 5.37),   # Mycket lång man -> ännu lägre ml/kg IBW (beräknat)
        ]
        
        # Act & Assert
        for height_cm, expected_tv_ml_kg_ibw in test_cases:
            result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
            self.assertAlmostEqual(result, expected_tv_ml_kg_ibw, places=1,
                                 msg=f"TV för {height_cm} cm man ska ge ~{expected_tv_ml_kg_ibw} ml/kg IBW")
        
        print("✅ T015 PASSED: Tidal Volume conversion med olika höjder fungerar korrekt")
    
    def test_t015_tidal_volume_conversion_edge_case_heights(self):
        """
        Verifiera Tidal Volume conversion med extrema höjder
        """
        # Arrange
        tv_ml = 400.0
        
        # Act & Assert
        # Mycket kort person (minimum IBW)
        result_short_male = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, 150.0, 1)
        expected_short_male = round(400.0 / 50.0, 2)  # 400 / 50 = 8.0 ml/kg IBW
        self.assertEqual(result_short_male, expected_short_male,
                        f"TV för kort man ska ge {expected_short_male} ml/kg IBW")
        
        # Mycket lång person
        result_tall_male = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, 200.0, 1)
        expected_tall_male = round(400.0 / 93.1, 2)  # 400 / 93.1 = 4.30 ml/kg IBW
        self.assertEqual(result_tall_male, expected_tall_male,
                        f"TV för lång man ska ge {expected_tall_male} ml/kg IBW")
        
        print("✅ T015 PASSED: Tidal Volume conversion med extrema höjder fungerar korrekt")
    
    def test_t015_tidal_volume_conversion_invalid_ibw(self):
        """
        Verifiera Tidal Volume conversion med ogiltig IBW
        """
        # Arrange
        tv_ml = 500.0
        
        # Act & Assert
        # Testa med negativ höjd som skulle ge IBW = 0
        result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, -10.0, 1)
        self.assertEqual(result, 0.0, "Negativ höjd ska ge 0 ml/kg IBW")
        
        print("✅ T015 PASSED: Tidal Volume conversion med ogiltig IBW hanteras korrekt")
    
    def test_t015_tidal_volume_range_validation(self):
        """
        Verifiera att konverterade Tidal Volume värden ligger inom klinisk range
        """
        # Arrange
        height_cm = 175.0
        sex = 1
        test_cases = [
            (300.0, True),   # ~4.26 ml/kg IBW - inom range (0-12)
            (600.0, True),   # ~8.52 ml/kg IBW - inom range (0-12)
            (1500.0, False), # ~21.3 ml/kg IBW - över range (0-12)
        ]
        
        # Act & Assert
        for tv_ml, should_be_in_range in test_cases:
            result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
            is_in_range = self.converter.validate_conversion_ranges(result, 'TV')
            
            if should_be_in_range:
                self.assertTrue(is_in_range,
                               f"TV {result} ml/kg IBW ska vara inom klinisk range")
            else:
                self.assertFalse(is_in_range,
                                f"TV {result} ml/kg IBW ska vara utanför klinisk range")
        
        print("✅ T015 PASSED: Tidal Volume range validation fungerar korrekt")
    
    def test_t015_tidal_volume_precision(self):
        """
        Verifiera att Tidal Volume conversion har korrekt precision (2 decimaler)
        """
        # Arrange
        tv_ml = 450.0
        height_cm = 175.0
        sex = 1
        
        # Act
        result = self.converter.convert_tidal_volume_ml_to_ml_kg_ibw(tv_ml, height_cm, sex)
        
        # Assert
        # Verifiera att resultatet har korrekt precision
        decimal_places = len(str(result).split('.')[-1]) if '.' in str(result) else 0
        self.assertLessEqual(decimal_places, 2, 
                            f"TV conversion ska ha max 2 decimaler, fick {decimal_places}")
        
        print("✅ T015 PASSED: Tidal Volume conversion har korrekt precision")

if __name__ == '__main__':
    unittest.main()
