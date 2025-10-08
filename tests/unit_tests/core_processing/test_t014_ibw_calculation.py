#!/usr/bin/env python3
"""
T014: Test IBW Calculation
Verifiera Devine formula för Ideal Body Weight beräkning
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

class TestT014IBWCalculation(unittest.TestCase):
    """T014: Test IBW Calculation"""
    
    def setUp(self):
        """Setup för varje test."""
        self.converter = create_unit_converter()
    
    def test_t014_ibw_calculation_male(self):
        """
        T014: Test IBW Calculation
        Verifiera Devine formula för Ideal Body Weight beräkning (Male)
        """
        # Arrange
        height_cm = 175.0  # 175 cm = 68.9 inches
        sex = 1  # Male
        
        # Expected: IBW = 50 + 2.3 * (68.9 - 60) = 50 + 2.3 * 8.9 = 50 + 20.47 = 70.47 kg
        height_inches = height_cm / 2.54
        expected_ibw = round(50 + 2.3 * (height_inches - 60), 1)
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för man (175 cm) ska vara {expected_ibw} kg, fick {result}")
        
        print("✅ T014 PASSED: IBW calculation för man fungerar korrekt")
    
    def test_t014_ibw_calculation_female(self):
        """
        Verifiera Devine formula för Ideal Body Weight beräkning (Female)
        """
        # Arrange
        height_cm = 165.0  # 165 cm = 64.96 inches
        sex = -1  # Female
        
        # Expected: IBW = 45.5 + 2.3 * (64.96 - 60) = 45.5 + 2.3 * 4.96 = 45.5 + 11.41 = 56.91 kg
        height_inches = height_cm / 2.54
        expected_ibw = round(45.5 + 2.3 * (height_inches - 60), 1)
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för kvinna (165 cm) ska vara {expected_ibw} kg, fick {result}")
        
        print("✅ T014 PASSED: IBW calculation för kvinna fungerar korrekt")
    
    def test_t014_ibw_calculation_short_male(self):
        """
        Verifiera IBW calculation för kort man (< 60 inches)
        """
        # Arrange
        height_cm = 150.0  # 150 cm = 59.06 inches (< 60 inches)
        sex = 1  # Male
        
        # Expected: För män under 60 inches ska IBW vara 50 kg (minimum)
        expected_ibw = 50.0
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för kort man (150 cm) ska vara {expected_ibw} kg (minimum)")
        
        print("✅ T014 PASSED: IBW calculation för kort man fungerar korrekt")
    
    def test_t014_ibw_calculation_short_female(self):
        """
        Verifiera IBW calculation för kort kvinna (< 60 inches)
        """
        # Arrange
        height_cm = 145.0  # 145 cm = 57.09 inches (< 60 inches)
        sex = -1  # Female
        
        # Expected: För kvinnor under 60 inches ska IBW vara 45.5 kg (minimum)
        expected_ibw = 45.5
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för kort kvinna (145 cm) ska vara {expected_ibw} kg (minimum)")
        
        print("✅ T014 PASSED: IBW calculation för kort kvinna fungerar korrekt")
    
    def test_t014_ibw_calculation_tall_male(self):
        """
        Verifiera IBW calculation för lång man (> 60 inches)
        """
        # Arrange
        height_cm = 190.0  # 190 cm = 74.8 inches
        sex = 1  # Male
        
        # Expected: IBW = 50 + 2.3 * (74.8 - 60) = 50 + 2.3 * 14.8 = 50 + 34.04 = 84.04 kg
        height_inches = height_cm / 2.54
        expected_ibw = round(50 + 2.3 * (height_inches - 60), 1)
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för lång man (190 cm) ska vara {expected_ibw} kg")
        
        print("✅ T014 PASSED: IBW calculation för lång man fungerar korrekt")
    
    def test_t014_ibw_calculation_tall_female(self):
        """
        Verifiera IBW calculation för lång kvinna (> 60 inches)
        """
        # Arrange
        height_cm = 180.0  # 180 cm = 70.87 inches
        sex = -1  # Female
        
        # Expected: IBW = 45.5 + 2.3 * (70.87 - 60) = 45.5 + 2.3 * 10.87 = 45.5 + 25.00 = 70.50 kg
        height_inches = height_cm / 2.54
        expected_ibw = round(45.5 + 2.3 * (height_inches - 60), 1)
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        self.assertEqual(result, expected_ibw,
                        f"IBW för lång kvinna (180 cm) ska vara {expected_ibw} kg")
        
        print("✅ T014 PASSED: IBW calculation för lång kvinna fungerar korrekt")
    
    def test_t014_ibw_calculation_edge_cases(self):
        """
        Verifiera IBW calculation med edge cases
        """
        # Arrange - Beräkna korrekta värden med Devine formula
        test_cases = [
            (170.0, 1, 65.9),   # Man, 170 cm -> 65.9 kg
            (160.0, -1, 52.4),  # Kvinna, 160 cm -> 52.4 kg (beräknat: 45.5 + 2.3*(62.99-60))
            (200.0, 1, 93.1),   # Man, 200 cm -> 93.1 kg (beräknat: 50 + 2.3*(78.74-60))
            (140.0, -1, 45.5),  # Kvinna, 140 cm -> 45.5 kg (minimum)
        ]
        
        # Act & Assert
        for height_cm, sex, expected_ibw in test_cases:
            result = self.converter.calculate_ideal_body_weight(height_cm, sex)
            self.assertEqual(result, expected_ibw,
                            f"IBW för {height_cm} cm, sex {sex} ska vara {expected_ibw} kg")
        
        print("✅ T014 PASSED: IBW calculation med edge cases fungerar korrekt")
    
    def test_t014_ibw_calculation_precision(self):
        """
        Verifiera att IBW calculation har korrekt precision (1 decimal)
        """
        # Arrange
        height_cm = 175.5  # Liten decimal för att testa precision
        sex = 1  # Male
        
        # Act
        result = self.converter.calculate_ideal_body_weight(height_cm, sex)
        
        # Assert
        # Verifiera att resultatet har korrekt precision (1 decimal)
        decimal_places = len(str(result).split('.')[-1]) if '.' in str(result) else 0
        self.assertLessEqual(decimal_places, 1, 
                            f"IBW calculation ska ha max 1 decimal, fick {decimal_places}")
        
        print("✅ T014 PASSED: IBW calculation har korrekt precision")
    
    def test_t014_ibw_calculation_formula_validation(self):
        """
        Verifiera att Devine formula implementeras korrekt
        """
        # Arrange - Testa med exakta värden för att verifiera formeln
        height_cm = 180.0  # 180 cm = 70.866 inches
        sex_male = 1
        sex_female = -1
        
        # Act
        ibw_male = self.converter.calculate_ideal_body_weight(height_cm, sex_male)
        ibw_female = self.converter.calculate_ideal_body_weight(height_cm, sex_female)
        
        # Assert
        # Verifiera att kvinnor har lägre IBW än män för samma höjd
        self.assertLess(ibw_female, ibw_male,
                       f"Kvinnor ska ha lägre IBW än män för samma höjd")
        
        # Verifiera att skillnaden är korrekt (50 - 45.5 = 4.5 kg)
        difference = ibw_male - ibw_female
        self.assertAlmostEqual(difference, 4.5, places=1,
                             msg=f"Skillnaden mellan man och kvinna ska vara ~4.5 kg")
        
        print("✅ T014 PASSED: Devine formula implementeras korrekt")

if __name__ == '__main__':
    unittest.main()
