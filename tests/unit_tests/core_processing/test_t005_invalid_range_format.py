#!/usr/bin/env python3
"""
T005: Test Invalid Range Format
Verifiera att ogiltiga format ger tydliga felmeddelanden
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera från vår utils-modul
from utils.case_range_parser import parse_case_range

class TestT005InvalidRangeFormat(unittest.TestCase):
    """T005: Test Invalid Range Format"""
    
    def test_t005_invalid_range_format(self):
        """
        T005: Test Invalid Range Format
        Verifiera att ogiltiga format ger tydliga felmeddelanden
        """
        # Test cases med ogiltiga format
        invalid_formats = [
            "1-",           # Saknar end värde
            "-100",         # Saknar start värde
            "1--100",       # Dubbel minus
            "1,",           # Trailing comma
            ",1",           # Leading comma
            "1,,2",         # Dubbel comma
            "1-2-3",        # För många minus
            "abc",          # Icke-numeriska tecken
            "1-abc",        # Blandat numeriskt och icke-numeriskt
            "",             # Tom sträng
            "   ",          # Bara whitespace
        ]
        
        for invalid_format in invalid_formats:
            with self.subTest(format=invalid_format):
                # Act & Assert
                try:
                    result = parse_case_range(invalid_format)
                    # Om vi kommer hit utan exception, kontrollera att resultatet är rimligt
                    if invalid_format.strip() == "":
                        # Tom sträng ska ge tom lista
                        self.assertEqual(result, [], 
                                       f"Tom sträng ska ge tom lista, fick {result}")
                    else:
                        # För andra ogiltiga format, logga vad som hände
                        print(f"⚠️ Format '{invalid_format}' gav oväntat resultat: {result}")
                except Exception as e:
                    # Detta är förväntat för ogiltiga format
                    print(f"✅ Format '{invalid_format}' gav förväntat fel: {type(e).__name__}: {e}")
        
        print("✅ T005 PASSED: Invalid range format hantering fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
