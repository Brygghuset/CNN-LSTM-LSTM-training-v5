#!/usr/bin/env python3
"""
Test Suite för Master POC CNN-LSTM-LSTM v5.0
============================================

Test Driven Development för AWS preprocessing pipeline.
Baserat på AWS_TEST-LIST_V5.0_3000_cases.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import unittest
import sys
import os
from typing import List

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera funktioner från vår utils-modul
from utils.case_range_parser import parse_case_range

class TestCaseRangeParsing(unittest.TestCase):
    """Test suite för case range parsing funktionalitet"""
    
    def test_t001_basic_range_parsing(self):
        """
        T001: Test Basic Range Parsing
        Verifiera att "1-100" parsas korrekt till lista med 100 cases
        """
        # Arrange
        case_range = "1-100"
        expected_count = 100
        expected_first = "0001"
        expected_last = "0100"
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(len(result), expected_count, 
                        f"Förväntade {expected_count} cases, fick {len(result)}")
        self.assertEqual(result[0], expected_first, 
                        f"Första case ska vara {expected_first}, fick {result[0]}")
        self.assertEqual(result[-1], expected_last, 
                        f"Sista case ska vara {expected_last}, fick {result[-1]}")
        
        # Verifiera att alla cases är sekventiella
        for i in range(len(result)):
            expected_case = f"{i+1:04d}"
            self.assertEqual(result[i], expected_case, 
                            f"Case {i} ska vara {expected_case}, fick {result[i]}")
        
        print("✅ T001 PASSED: Basic range parsing fungerar korrekt")
    
    def test_t002_comma_separated_parsing(self):
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
        self.assertEqual(len(result), 3, 
                        f"Förväntade 3 cases, fick {len(result)}")
        self.assertEqual(result, expected_cases, 
                        f"Förväntade {expected_cases}, fick {result}")
        
        print("✅ T002 PASSED: Comma-separated parsing fungerar korrekt")
    
    def test_t003_mixed_format_parsing(self):
        """
        T003: Test Mixed Format Parsing
        Verifiera att "1-10,17,0022" parsas korrekt till lista med 12 cases
        """
        # Arrange
        case_range = "1-10,17,0022"
        expected_cases = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0017", "0022"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(len(result), 12, 
                        f"Förväntade 12 cases, fick {len(result)}")
        self.assertEqual(result, expected_cases, 
                        f"Förväntade {expected_cases}, fick {result}")
        
        print("✅ T003 PASSED: Mixed format parsing fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
