#!/usr/bin/env python3
"""
T034: Test Insufficient Data Handling
Verifiera hantering när case har <300s data
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

class TestT034InsufficientDataHandling(unittest.TestCase):
    """T034: Test Insufficient Data Handling"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t034_insufficient_data_handling_basic(self):
        """
        T034: Test Insufficient Data Handling
        Verifiera hantering när case har <300s data
        """
        # Arrange
        # Skapa test data med mindre än 300 time steps
        time_steps = 200  # Mindre än 300
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att inga windows skapas från otillräcklig data
        self.assertEqual(len(windows), 0, "Ska inte skapa windows från < 300 time steps")
        self.assertEqual(len(static_windows), 0, "Ska inte skapa static windows från < 300 time steps")
        self.assertEqual(len(metadata), 0, "Ska inte ha metadata från < 300 time steps")
        
        # Verifiera att windows array har korrekt shape för tomma resultat
        self.assertEqual(windows.shape, (0, 300, 16),
                        "Empty windows ska ha shape (0, 300, 16)")
        self.assertEqual(static_windows.shape, (0, 6),
                        "Empty static windows ska ha shape (0, 6)")
        
        print("✅ T034 PASSED: Basic insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_edge_cases(self):
        """
        Verifiera insufficient data handling med edge cases
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: Exakt 299 time steps
        timeseries_data_299 = np.random.randn(299, 16)
        windows_299, _, metadata_299 = self.window_creator.create_sliding_windows(
            timeseries_data_299, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_299), 0, "Ska inte skapa windows från 299 time steps")
        self.assertEqual(windows_299.shape, (0, 300, 16),
                        "Empty windows från 299 steg ska ha shape (0, 300, 16)")
        
        # Test case 2: Exakt 1 time step
        timeseries_data_1 = np.random.randn(1, 16)
        windows_1, _, metadata_1 = self.window_creator.create_sliding_windows(
            timeseries_data_1, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_1), 0, "Ska inte skapa windows från 1 time step")
        self.assertEqual(windows_1.shape, (0, 300, 16),
                        "Empty windows från 1 steg ska ha shape (0, 300, 16)")
        
        # Test case 3: Exakt 0 time steps
        timeseries_data_0 = np.random.randn(0, 16)
        windows_0, _, metadata_0 = self.window_creator.create_sliding_windows(
            timeseries_data_0, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_0), 0, "Ska inte skapa windows från 0 time steps")
        self.assertEqual(windows_0.shape, (0, 300, 16),
                        "Empty windows från 0 steg ska ha shape (0, 300, 16)")
        
        print("✅ T034 PASSED: Edge cases insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_boundary_test(self):
        """
        Verifiera boundary mellan insufficient och sufficient data
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: 299 time steps (insufficient)
        timeseries_data_299 = np.random.randn(299, 16)
        windows_299, _, metadata_299 = self.window_creator.create_sliding_windows(
            timeseries_data_299, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_299), 0, "299 time steps ska vara insufficient")
        
        # Test case 2: 300 time steps (sufficient)
        timeseries_data_300 = np.random.randn(300, 16)
        windows_300, _, metadata_300 = self.window_creator.create_sliding_windows(
            timeseries_data_300, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_300), 1, "300 time steps ska vara sufficient")
        self.assertEqual(windows_300.shape, (1, 300, 16),
                        "300 time steps ska ge shape (1, 300, 16)")
        
        # Test case 3: 301 time steps (sufficient)
        timeseries_data_301 = np.random.randn(301, 16)
        windows_301, _, metadata_301 = self.window_creator.create_sliding_windows(
            timeseries_data_301, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_301), 1, "301 time steps ska vara sufficient")
        self.assertEqual(windows_301.shape, (1, 300, 16),
                        "301 time steps ska ge shape (1, 300, 16)")
        
        print("✅ T034 PASSED: Boundary test insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_window_count_calculation(self):
        """
        Verifiera window count calculation för insufficient data
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Testa olika data storlekar
        test_cases = [
            (0, 0),     # 0 time steps
            (1, 0),     # 1 time step
            (100, 0),   # 100 time steps
            (200, 0),   # 200 time steps
            (299, 0),   # 299 time steps
            (300, 1),   # 300 time steps
            (301, 1),   # 301 time steps
            (330, 2),    # 330 time steps
        ]
        
        # Act & Assert
        for time_steps, expected_windows in test_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            # Beräkna förväntat antal windows
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            
            # Verifiera att beräkningen är korrekt
            self.assertEqual(calculated_windows, expected_windows,
                           f"Window count calculation för {time_steps} steg ska vara {expected_windows}")
            
            # Verifiera att faktisk window creation matchar beräkningen
            windows, _, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            self.assertEqual(len(windows), expected_windows,
                           f"Faktisk window creation för {time_steps} steg ska ge {expected_windows} windows")
        
        print("✅ T034 PASSED: Window count calculation insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_dataframe(self):
        """
        Verifiera insufficient data handling med pandas DataFrame
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: DataFrame med 200 time steps
        time_steps_200 = 200
        timeseries_columns = [f'feature_{i}' for i in range(16)]
        
        df_200 = pd.DataFrame(
            np.random.randn(time_steps_200, 16),
            columns=timeseries_columns
        )
        
        # Act
        windows_200, static_windows_200, metadata_200 = self.window_creator.create_windows_from_dataframe(
            df_200, timeseries_columns, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_200), 0, "DataFrame med 200 steg ska inte skapa windows")
        self.assertEqual(windows_200.shape, (0, 300, 16),
                        "Empty windows från DataFrame ska ha shape (0, 300, 16)")
        
        # Test case 2: DataFrame med 300 time steps
        time_steps_300 = 300
        df_300 = pd.DataFrame(
            np.random.randn(time_steps_300, 16),
            columns=timeseries_columns
        )
        
        # Act
        windows_300, static_windows_300, metadata_300 = self.window_creator.create_windows_from_dataframe(
            df_300, timeseries_columns, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_300), 1, "DataFrame med 300 steg ska skapa 1 window")
        self.assertEqual(windows_300.shape, (1, 300, 16),
                        "Windows från DataFrame med 300 steg ska ha shape (1, 300, 16)")
        
        print("✅ T034 PASSED: DataFrame insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_error_handling(self):
        """
        Verifiera error handling för insufficient data
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: Försök med 200 time steps
        timeseries_data_200 = np.random.randn(200, 16)
        
        # Act
        windows_200, static_windows_200, metadata_200 = self.window_creator.create_sliding_windows(
            timeseries_data_200, static_data
        )
        
        # Assert
        # Verifiera att ingen exception kastas
        self.assertIsNotNone(windows_200, "Ska inte kasta exception för insufficient data")
        self.assertIsNotNone(static_windows_200, "Ska inte kasta exception för insufficient data")
        self.assertIsNotNone(metadata_200, "Ska inte kasta exception för insufficient data")
        
        # Verifiera att resultatet är tomt men korrekt formaterat
        self.assertEqual(len(windows_200), 0, "Ska returnera tom lista för insufficient data")
        self.assertEqual(windows_200.shape, (0, 300, 16),
                        "Ska returnera korrekt shape för insufficient data")
        
        print("✅ T034 PASSED: Error handling insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_custom_config(self):
        """
        Verifiera insufficient data handling med custom configuration
        """
        # Arrange
        # Skapa custom config med 300s window size
        config = WindowConfig(
            window_size_seconds=300,
            step_size_seconds=30,
            timeseries_features=16,
            static_features=6
        )
        
        window_creator_custom = MasterPOCWindowCreator(config)
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: 200 time steps med custom config
        timeseries_data_200 = np.random.randn(200, 16)
        
        # Act
        windows_200, static_windows_200, metadata_200 = window_creator_custom.create_sliding_windows(
            timeseries_data_200, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_200), 0, "Custom config ska inte skapa windows från 200 steg")
        self.assertEqual(windows_200.shape, (0, 300, 16),
                        "Custom config empty windows ska ha shape (0, 300, 16)")
        
        # Verifiera att window count calculation fungerar med custom config
        calculated_windows = window_creator_custom.calculate_expected_window_count(200)
        self.assertEqual(calculated_windows, 0,
                        "Custom config window count calculation ska vara 0 för 200 steg")
        
        print("✅ T034 PASSED: Custom config insufficient data handling fungerar korrekt")
    
    def test_t034_insufficient_data_handling_comprehensive(self):
        """
        Omfattande test av insufficient data handling
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Testa med olika data storlekar under 300
        insufficient_cases = [0, 1, 50, 100, 150, 200, 250, 299]
        
        # Act & Assert
        for time_steps in insufficient_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            # Verifiera window count calculation
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            self.assertEqual(calculated_windows, 0,
                           f"Window count calculation för {time_steps} steg ska vara 0")
            
            # Verifiera faktisk window creation
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            self.assertEqual(len(windows), 0,
                           f"Ska inte skapa windows från {time_steps} time steps")
            self.assertEqual(windows.shape, (0, 300, 16),
                           f"Empty windows från {time_steps} steg ska ha shape (0, 300, 16)")
            self.assertEqual(len(metadata), 0,
                           f"Ska inte ha metadata från {time_steps} time steps")
        
        # Testa med sufficient data för kontrast
        sufficient_cases = [300, 301, 330, 600]
        
        for time_steps in sufficient_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            # Verifiera window count calculation
            calculated_windows = self.window_creator.calculate_expected_window_count(time_steps)
            self.assertGreater(calculated_windows, 0,
                             f"Window count calculation för {time_steps} steg ska vara > 0")
            
            # Verifiera faktisk window creation
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            self.assertGreater(len(windows), 0,
                             f"Ska skapa windows från {time_steps} time steps")
            self.assertGreater(len(metadata), 0,
                             f"Ska ha metadata från {time_steps} time steps")
        
        print("✅ T034 PASSED: Comprehensive insufficient data handling test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
