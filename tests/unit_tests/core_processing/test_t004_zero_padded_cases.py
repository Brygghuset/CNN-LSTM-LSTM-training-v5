#!/usr/bin/env python3
"""
T004: Test Zero-Padded Cases
Verifiera att "0001,0022" hanteras korrekt med zero-padding
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera från vår utils-modul
from utils.case_range_parser import parse_case_range

class TestT004ZeroPaddedCases(unittest.TestCase):
    """T004: Test Zero-Padded Cases"""
    
    def test_t004_zero_padded_cases(self):
        """
        T004: Test Zero-Padded Cases
        Verifiera att "0001,0022" hanteras korrekt med zero-padding
        """
        # Arrange
        case_range = "0001,0022"
        expected_cases = ["0001", "0022"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(len(result), 2, 
                        f"Förväntade 2 cases, fick {len(result)}")
        self.assertEqual(result, expected_cases, 
                        f"Förväntade {expected_cases}, fick {result}")
        
        # Verifiera att zero-padding bevaras
        for case in result:
            self.assertEqual(len(case), 4, 
                            f"Case {case} ska ha 4 tecken (zero-padded)")
            self.assertTrue(case.isdigit(), 
                          f"Case {case} ska vara numeriskt")
        
        print("✅ T004 PASSED: Zero-padded cases hanteras korrekt")

if __name__ == '__main__':
    unittest.main()
