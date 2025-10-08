#!/usr/bin/env python3
"""
T035: Test Window Count Calculation
Verifiera korrekt beräkning av förväntat antal windows
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår master POC window creator modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_window_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_window_creator.py')
)
master_poc_window_creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_window_creator)

# Använd modulen
MasterPOCWindowCreator = master_poc_window_creator.MasterPOCWindowCreator
WindowConfig = master_poc_window_creator.WindowConfig
create_master_poc_window_creator = master_poc_window_creator.create_master_poc_window_creator

class TestT035WindowCountCalculation(unittest.TestCase):
    """T035: Test Window Count Calculation"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t035_window_count_calculation_basic(self):
        """
        T035: Test Window Count Calculation
        Verifiera korrekt beräkning av förväntat antal windows
        """
        # Arrange
        # Testa med olika data storlekar
        test_cases = [
            (0, 0),      # 0 time steps
            (100, 0),    # 100 time steps (< 300)
            (299, 0),    # 299 time steps (< 300)
            (300, 1),    # 300 time steps (exakt 1 window)
            (330, 2),    # 330 time steps (2 windows)
            (600, 11),   # 600 time steps (11 windows)
            (900, 21),   # 900 time steps (21 windows)
            (1000, 24),  # 1000 time steps (24 windows)
        ]
        
        # Act & Assert
        for time_steps, expected_windows in test_cases:
            # Beräkna förväntat antal windows
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att beräkningen är korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Window count calculation för {time_steps} steg ska vara {expected_windows}")
        
        print("✅ T035 PASSED: Basic window count calculation fungerar korrekt")
    
    def test_t035_window_count_calculation_formula(self):
        """
        Verifiera att window count calculation använder korrekt formel
        """
        # Arrange
        # Formel: (time_steps - window_size) // step_size + 1
        # För time_steps < window_size: 0
        # För time_steps >= window_size: (time_steps - 300) // 30 + 1
        
        test_cases = [
            # (time_steps, expected_calculation, description)
            (0, 0, "0 steg"),
            (100, 0, "100 steg (< 300)"),
            (299, 0, "299 steg (< 300)"),
            (300, (300 - 300) // 30 + 1, "300 steg (exakt boundary)"),
            (330, (330 - 300) // 30 + 1, "330 steg"),
            (600, (600 - 300) // 30 + 1, "600 steg"),
            (900, (900 - 300) // 30 + 1, "900 steg"),
            (1000, (1000 - 300) // 30 + 1, "1000 steg"),
        ]
        
        # Act & Assert
        for time_steps, expected_calculation, description in test_cases:
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att beräkningen matchar förväntad formel
            self.assertEqual(calculated_windows, expected_calculation,
                           f"Window count för {description} ska vara {expected_calculation}")
        
        print("✅ T035 PASSED: Window count calculation formula fungerar korrekt")
    
    def test_t035_window_count_calculation_vs_actual(self):
        """
        Verifiera att window count calculation matchar faktisk window creation
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Testa med olika data storlekar
        test_cases = [300, 330, 600, 900, 1000, 1200]
        
        # Act & Assert
        for time_steps in test_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            # Beräkna förväntat antal windows
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Skapa faktiska windows
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            # Verifiera att beräkning matchar faktisk creation
            self.assertEqual(len(windows), calculated_windows,
                           f"Faktisk window creation för {time_steps} steg ska matcha beräkning {calculated_windows}")
            
            # Verifiera att metadata har rätt antal entries
            self.assertEqual(len(metadata), calculated_windows,
                           f"Metadata för {time_steps} steg ska ha {calculated_windows} entries")
        
        print("✅ T035 PASSED: Window count calculation vs actual fungerar korrekt")
    
    def test_t035_window_count_calculation_boundary_cases(self):
        """
        Verifiera window count calculation vid boundary cases
        """
        # Arrange
        # Testa boundary cases runt 300 steg
        boundary_cases = [
            (299, 0, "299 steg (just under boundary)"),
            (300, 1, "300 steg (exakt boundary)"),
            (301, 1, "301 steg (just over boundary)"),
            (329, 1, "329 steg (just under second window)"),
            (330, 2, "330 steg (exakt second window)"),
            (331, 2, "331 steg (just over second window)"),
        ]
        
        # Act & Assert
        for time_steps, expected_windows, description in boundary_cases:
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att boundary cases hanteras korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Boundary case {description} ska ge {expected_windows} windows")
        
        print("✅ T035 PASSED: Window count calculation boundary cases fungerar korrekt")
    
    def test_t035_window_count_calculation_large_datasets(self):
        """
        Verifiera window count calculation för stora datasets
        """
        # Arrange
        # Testa med stora datasets
        large_datasets = [
            (3000, (3000 - 300) // 30 + 1),  # 3000 steg
            (6000, (6000 - 300) // 30 + 1),  # 6000 steg
            (9000, (9000 - 300) // 30 + 1),  # 9000 steg
            (12000, (12000 - 300) // 30 + 1), # 12000 steg
        ]
        
        # Act & Assert
        for time_steps, expected_windows in large_datasets:
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att stora datasets hanteras korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Large dataset {time_steps} steg ska ge {expected_windows} windows")
            
            # Verifiera att beräkningen är rimlig
            self.assertGreater(calculated_windows, 0,
                             f"Large dataset {time_steps} steg ska ge > 0 windows")
            self.assertLess(calculated_windows, time_steps,
                          f"Window count för {time_steps} steg ska vara < {time_steps}")
        
        print("✅ T035 PASSED: Window count calculation large datasets fungerar korrekt")
    
    def test_t035_window_count_calculation_custom_config(self):
        """
        Verifiera window count calculation med custom configuration
        """
        # Arrange
        # Skapa custom config med samma parametrar
        config = WindowConfig(
            window_size_seconds=300,
            step_size_seconds=30,
            timeseries_features=16,
            static_features=6
        )
        
        window_creator_custom = MasterPOCWindowCreator(config)
        
        # Testa med olika data storlekar
        test_cases = [300, 600, 900, 1200]
        
        # Act & Assert
        for time_steps in test_cases:
            # Beräkna med både default och custom config
            calculated_default = self.window_creator.calculate_expected_window_count(time_steps)
            calculated_custom = window_creator_custom.calculate_expected_window_count(time_steps)
            
            # Verifiera att båda ger samma resultat
            self.assertEqual(calculated_default, calculated_custom,
                           f"Default och custom config ska ge samma resultat för {time_steps} steg")
            
            # Verifiera att custom config använder rätt parametrar
            expected_calculation = (time_steps - config.window_size_seconds) // config.step_size_seconds + 1
            if time_steps < config.window_size_seconds:
                expected_calculation = 0
            
            self.assertEqual(calculated_custom, expected_calculation,
                           f"Custom config för {time_steps} steg ska ge {expected_calculation}")
        
        print("✅ T035 PASSED: Window count calculation custom config fungerar korrekt")
    
    def test_t035_window_count_calculation_edge_cases(self):
        """
        Verifiera window count calculation med edge cases
        """
        # Arrange
        # Testa edge cases
        edge_cases = [
            (0, 0, "0 steg"),
            (1, 0, "1 steg"),
            (299, 0, "299 steg"),
            (300, 1, "300 steg"),
            (301, 1, "301 steg"),
            (330, 2, "330 steg"),
            (360, 3, "360 steg"),
            (390, 4, "390 steg"),
        ]
        
        # Act & Assert
        for time_steps, expected_windows, description in edge_cases:
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att edge cases hanteras korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Edge case {description} ska ge {expected_windows} windows")
        
        print("✅ T035 PASSED: Window count calculation edge cases fungerar korrekt")
    
    def test_t035_window_count_calculation_performance(self):
        """
        Verifiera att window count calculation är snabb
        """
        # Arrange
        import time
        
        # Testa med stora datasets för att mäta prestanda
        large_time_steps = [10000, 50000, 100000]
        
        # Act & Assert
        for time_steps in large_time_steps:
            start_time = time.time()
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            end_time = time.time()
            
            calculation_time = end_time - start_time
            
            # Verifiera att beräkningen är snabb (< 0.001 sekunder)
            self.assertLess(calculation_time, 0.001,
                           f"Window count calculation för {time_steps} steg ska vara snabb")
            
            # Verifiera att resultatet är korrekt
            expected_windows = (time_steps - 300) // 30 + 1
            self.assertEqual(calculated_windows, expected_windows,
                           f"Window count för {time_steps} steg ska vara {expected_windows}")
        
        print("✅ T035 PASSED: Window count calculation performance fungerar korrekt")
    
    def test_t035_window_count_calculation_comprehensive(self):
        """
        Omfattande test av window count calculation
        """
        # Arrange
        # Testa med många olika data storlekar
        test_cases = []
        
        # Lägg till edge cases
        test_cases.extend([(i, 0) for i in range(0, 300)])
        
        # Lägg till boundary cases
        test_cases.extend([(300, 1), (330, 2), (360, 3), (390, 4)])
        
        # Lägg till vanliga cases
        test_cases.extend([(600, 11), (900, 21), (1200, 31), (1500, 41)])
        
        # Lägg till stora cases
        test_cases.extend([(3000, 91), (6000, 191), (9000, 291)])
        
        # Act & Assert
        for time_steps, expected_windows in test_cases:
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att beräkningen är korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Window count för {time_steps} steg ska vara {expected_windows}")
            
            # Verifiera att resultatet är rimligt
            if time_steps < 300:
                self.assertEqual(calculated_windows, 0,
                               f"Window count för {time_steps} steg ska vara 0")
            else:
                self.assertGreater(calculated_windows, 0,
                                 f"Window count för {time_steps} steg ska vara > 0")
        
        print("✅ T035 PASSED: Comprehensive window count calculation test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
