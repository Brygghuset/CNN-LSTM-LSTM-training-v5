#!/usr/bin/env python3
"""
T002: Test Comma-Separated Parsing
Verifiera att "1,5,10" parsas korrekt till lista med 3 cases
"""

import unittest
import sys
import os

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår case range parser modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "case_range_parser", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'utils', 'case_range_parser.py')
)
case_range_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_range_parser)

# Använd modulen
parse_case_range = case_range_parser.parse_case_range

class TestT002CommaSeparatedParsing(unittest.TestCase):
    """T002: Test Comma-Separated Parsing"""
    
    def test_t002_comma_separated_parsing_basic(self):
        """
        T002: Test Comma-Separated Parsing
        Verifiera att "1,5,10" parsas korrekt till lista med 3 cases
        """
        # Arrange
        case_range = "1,5,10"
        expected_cases = ["0001", "0005", "0010"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Comma-separated parsing ska ge {expected_cases}, fick {result}")
        
        print("✅ T002 PASSED: Basic comma-separated parsing fungerar korrekt")
    
    def test_t002_comma_separated_parsing_mixed_numbers(self):
        """
        Verifiera comma-separated parsing med olika nummerformat
        """
        # Arrange
        test_cases = [
            ("1,2,3", ["0001", "0002", "0003"]),
            ("10,20,30", ["0010", "0020", "0030"]),
            ("100,200,300", ["0100", "0200", "0300"]),
            ("1000,2000,3000", ["1000", "2000", "3000"]),
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Comma-separated '{case_range}' ska ge {expected_cases}")
        
        print("✅ T002 PASSED: Comma-separated parsing med olika nummerformat fungerar korrekt")
    
    def test_t002_comma_separated_parsing_with_spaces(self):
        """
        Verifiera comma-separated parsing med mellanslag
        """
        # Arrange
        test_cases = [
            ("1, 5, 10", ["0001", "0005", "0010"]),
            (" 1 , 5 , 10 ", ["0001", "0005", "0010"]),
            ("1 ,5,10", ["0001", "0005", "0010"]),
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Comma-separated med spaces '{case_range}' ska ge {expected_cases}")
        
        print("✅ T002 PASSED: Comma-separated parsing med mellanslag fungerar korrekt")
    
    def test_t002_comma_separated_parsing_single_case(self):
        """
        Verifiera comma-separated parsing med enbart ett case
        """
        # Arrange
        case_range = "42"
        expected_cases = ["0042"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Single case '{case_range}' ska ge {expected_cases}")
        
        print("✅ T002 PASSED: Comma-separated parsing med enbart ett case fungerar korrekt")
    
    def test_t002_comma_separated_parsing_empty_cases(self):
        """
        Verifiera comma-separated parsing med tomma cases
        """
        # Arrange
        test_cases = [
            ("1,,10", ["0001", "0010"]),  # Tom case i mitten ignoreras
            (",1,10", ["0001", "0010"]),  # Tom case i början ignoreras
            ("1,10,", ["0001", "0010"]),  # Tom case i slutet ignoreras
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Comma-separated med tomma cases '{case_range}' ska ge {expected_cases}")
        
        print("✅ T002 PASSED: Comma-separated parsing med tomma cases fungerar korrekt")
    
    def test_t002_comma_separated_parsing_large_list(self):
        """
        Verifiera comma-separated parsing med stor lista
        """
        # Arrange
        # Skapa en lista med 100 cases
        case_numbers = list(range(1, 101))
        case_range = ",".join(map(str, case_numbers))
        expected_cases = [f"{i:04d}" for i in case_numbers]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Large comma-separated list ska ge {len(expected_cases)} cases")
        self.assertEqual(len(result), 100)
        
        print("✅ T002 PASSED: Comma-separated parsing med stor lista fungerar korrekt")
    
    def test_t002_comma_separated_parsing_duplicate_cases(self):
        """
        Verifiera comma-separated parsing med duplicerade cases
        """
        # Arrange
        case_range = "1,5,1,10,5"
        expected_cases = ["0001", "0005", "0001", "0010", "0005"]  # Behåller duplicater
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Comma-separated med duplicater ska behålla alla cases")
        
        print("✅ T002 PASSED: Comma-separated parsing med duplicerade cases fungerar korrekt")
    
    def test_t002_comma_separated_parsing_edge_cases(self):
        """
        Verifiera comma-separated parsing med edge cases
        """
        # Arrange
        test_cases = [
            ("0", ["0000"]),  # Case 0
            ("0000", ["0000"]),  # Zero-padded 0
            ("9999", ["9999"]),  # Max case number
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Edge case '{case_range}' ska ge {expected_cases}")
        
        print("✅ T002 PASSED: Comma-separated parsing med edge cases fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
