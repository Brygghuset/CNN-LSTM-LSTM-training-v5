#!/usr/bin/env python3
"""
T006: Test Large Range Parsing
Verifiera att "1-3000" parsas effektivt utan minnesöverskridning
"""

import unittest
import sys
import os
import psutil
import time

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera från vår utils-modul
from utils.case_range_parser import parse_case_range

class TestT006LargeRangeParsing(unittest.TestCase):
    """T006: Test Large Range Parsing"""
    
    def test_t006_large_range_parsing(self):
        """
        T006: Test Large Range Parsing
        Verifiera att "1-3000" parsas effektivt utan minnesöverskridning
        """
        # Arrange
        case_range = "1-3000"
        expected_count = 3000
        expected_first = "0001"
        expected_last = "3000"
        
        # Mät minnesanvändning före parsing
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Act
        start_time = time.time()
        result = parse_case_range(case_range)
        end_time = time.time()
        
        # Mät minnesanvändning efter parsing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Assert
        self.assertEqual(len(result), expected_count, 
                        f"Förväntade {expected_count} cases, fick {len(result)}")
        self.assertEqual(result[0], expected_first, 
                        f"Första case ska vara {expected_first}, fick {result[0]}")
        self.assertEqual(result[-1], expected_last, 
                        f"Sista case ska vara {expected_last}, fick {result[-1]}")
        
        # Verifiera att parsing var effektiv
        parsing_time = end_time - start_time
        self.assertLess(parsing_time, 5.0, 
                       f"Parsing ska ta mindre än 5 sekunder, tog {parsing_time:.2f}s")
        
        # Verifiera att minnesanvändning är rimlig (mindre än 100MB för 3000 cases)
        self.assertLess(memory_used, 100, 
                       f"Minnesanvändning ska vara mindre än 100MB, använde {memory_used:.2f}MB")
        
        # Verifiera att alla cases är sekventiella
        for i in range(min(10, len(result))):  # Testa första 10 cases
            expected_case = f"{i+1:04d}"
            self.assertEqual(result[i], expected_case, 
                            f"Case {i} ska vara {expected_case}, fick {result[i]}")
        
        # Verifiera att alla cases är korrekt formaterade
        for case in result:
            self.assertEqual(len(case), 4, 
                            f"Case {case} ska ha 4 tecken (zero-padded)")
            self.assertTrue(case.isdigit(), 
                          f"Case {case} ska vara numeriskt")
        
        print(f"✅ T006 PASSED: Large range parsing fungerar effektivt")
        print(f"   Parsing time: {parsing_time:.2f}s")
        print(f"   Memory used: {memory_used:.2f}MB")
        print(f"   Cases parsed: {len(result)}")

if __name__ == '__main__':
    unittest.main()
