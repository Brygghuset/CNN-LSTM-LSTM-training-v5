#!/usr/bin/env python3
"""
T003: Test Mixed Format Parsing
Verifiera att "1-10,17,0022" parsas korrekt till lista med 12 cases
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

class TestT003MixedFormatParsing(unittest.TestCase):
    """T003: Test Mixed Format Parsing"""
    
    def test_t003_mixed_format_parsing_basic(self):
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
        self.assertEqual(result, expected_cases,
                        f"Mixed format parsing ska ge {expected_cases}, fick {result}")
        self.assertEqual(len(result), 12, "Mixed format ska ge 12 cases")
        
        print("✅ T003 PASSED: Basic mixed format parsing fungerar korrekt")
    
    def test_t003_mixed_format_parsing_complex(self):
        """
        Verifiera mixed format parsing med komplexa kombinationer
        """
        # Arrange
        test_cases = [
            ("1-5,10,15-17", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
            ("10-12,20,30-32", ["0010", "0011", "0012", "0020", "0030", "0031", "0032"]),
            ("1,5-7,10,15-16", ["0001", "0005", "0006", "0007", "0010", "0015", "0016"]),
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Complex mixed format '{case_range}' ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Complex mixed format parsing fungerar korrekt")
    
    def test_t003_mixed_format_parsing_with_spaces(self):
        """
        Verifiera mixed format parsing med mellanslag
        """
        # Arrange
        test_cases = [
            ("1-5, 10, 15-17", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
            (" 1-5 , 10 , 15-17 ", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
            ("1 - 5,10,15 - 17", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Mixed format med spaces '{case_range}' ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Mixed format parsing med mellanslag fungerar korrekt")
    
    def test_t003_mixed_format_parsing_zero_padded(self):
        """
        Verifiera mixed format parsing med zero-padded cases
        """
        # Arrange
        test_cases = [
            ("0001-0005,0010,0015-0017", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
            ("1-5,0010,15-17", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
            ("0001-5,10,0015-17", ["0001", "0002", "0003", "0004", "0005", "0010", "0015", "0016", "0017"]),
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Mixed format med zero-padding '{case_range}' ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Mixed format parsing med zero-padding fungerar korrekt")
    
    def test_t003_mixed_format_parsing_large_ranges(self):
        """
        Verifiera mixed format parsing med stora ranges
        """
        # Arrange
        case_range = "1-100,200,300-305"
        expected_count = 100 + 1 + 6  # 1-100 (100 cases) + 200 (1 case) + 300-305 (6 cases) = 107 cases
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(len(result), expected_count,
                        f"Large mixed format ska ge {expected_count} cases")
        self.assertEqual(result[0], "0001")
        self.assertEqual(result[99], "0100")  # Slutet av första range
        self.assertEqual(result[100], "0200")  # Mitten case
        self.assertEqual(result[-1], "0305")  # Slutet av sista range
        
        print("✅ T003 PASSED: Mixed format parsing med stora ranges fungerar korrekt")
    
    def test_t003_mixed_format_parsing_overlapping_ranges(self):
        """
        Verifiera mixed format parsing med överlappande ranges
        """
        # Arrange
        case_range = "1-5,3-7,5-10"
        expected_cases = ["0001", "0002", "0003", "0004", "0005", "0003", "0004", "0005", "0006", "0007", "0005", "0006", "0007", "0008", "0009", "0010"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Overlapping ranges ska behålla alla cases inklusive duplicater")
        
        print("✅ T003 PASSED: Mixed format parsing med överlappande ranges fungerar korrekt")
    
    def test_t003_mixed_format_parsing_single_cases_mixed(self):
        """
        Verifiera mixed format parsing med enbart enskilda cases
        """
        # Arrange
        case_range = "1,5,10,15,20"
        expected_cases = ["0001", "0005", "0010", "0015", "0020"]
        
        # Act
        result = parse_case_range(case_range)
        
        # Assert
        self.assertEqual(result, expected_cases,
                        f"Single cases mixed format ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Mixed format parsing med enbart enskilda cases fungerar korrekt")
    
    def test_t003_mixed_format_parsing_empty_parts(self):
        """
        Verifiera mixed format parsing med tomma delar
        """
        # Arrange
        test_cases = [
            ("1-5,,10", ["0001", "0002", "0003", "0004", "0005", "0010"]),  # Tom del ignoreras
            (",1-5,10", ["0001", "0002", "0003", "0004", "0005", "0010"]),  # Tom del i början ignoreras
            ("1-5,10,", ["0001", "0002", "0003", "0004", "0005", "0010"]),  # Tom del i slutet ignoreras
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Mixed format med tomma delar '{case_range}' ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Mixed format parsing med tomma delar fungerar korrekt")
    
    def test_t003_mixed_format_parsing_edge_cases(self):
        """
        Verifiera mixed format parsing med edge cases
        """
        # Arrange
        test_cases = [
            ("0-2,5", ["0000", "0001", "0002", "0005"]),  # Inkluderar case 0
            ("9998-9999,1", ["9998", "9999", "0001"]),  # Stora nummer
            ("1-1,2-2", ["0001", "0002"]),  # Enkla ranges
        ]
        
        # Act & Assert
        for case_range, expected_cases in test_cases:
            result = parse_case_range(case_range)
            self.assertEqual(result, expected_cases,
                           f"Edge case '{case_range}' ska ge {expected_cases}")
        
        print("✅ T003 PASSED: Mixed format parsing med edge cases fungerar korrekt")
    
    def test_t003_mixed_format_parsing_performance(self):
        """
        Verifiera att mixed format parsing är effektiv
        """
        # Arrange
        import time
        case_range = "1-1000,2000,3000-3005"  # Stor mixed format
        
        # Act
        start_time = time.time()
        result = parse_case_range(case_range)
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0, f"Mixed format parsing ska vara snabb, tog {processing_time:.3f}s")
        self.assertEqual(len(result), 1007, "Stor mixed format ska ge 1007 cases")
        
        print("✅ T003 PASSED: Mixed format parsing prestanda är acceptabel")

if __name__ == '__main__':
    unittest.main()
