#!/usr/bin/env python3
"""
T030: Test Window Size 300s
Verifiera att windows är exakt 300 sekunder
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

class TestT030WindowSize300s(unittest.TestCase):
    """T030: Test Window Size 300s"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t030_window_size_300s_basic(self):
        """
        T030: Test Window Size 300s
        Verifiera att windows är exakt 300 sekunder
        """
        # Arrange
        # Skapa test data med 1000 time steps (1000 sekunder)
        time_steps = 1000
        timeseries_data = np.random.randn(time_steps, 16)  # 16 timeseries features
        static_data = np.array([25, 1, 170, 70, 24, 2])  # 6 static features
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att windows har korrekt shape
        self.assertEqual(len(windows.shape), 3, "Windows ska ha 3 dimensioner")
        self.assertEqual(windows.shape[1], 300, "Window size ska vara exakt 300 sekunder")
        self.assertEqual(windows.shape[2], 16, "Timeseries features ska vara 16")
        
        # Verifiera att varje window är exakt 300 steg
        for i, window in enumerate(windows):
            self.assertEqual(len(window), 300, f"Window {i} ska vara exakt 300 steg")
        
        # Verifiera att metadata anger korrekt duration
        for meta in metadata:
            self.assertEqual(meta['duration_seconds'], 300, 
                           f"Window {meta['window_index']} ska ha duration 300 sekunder")
        
        print("✅ T030 PASSED: Basic window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_exact_boundary(self):
        """
        Verifiera window size med exakt 300 time steps
        """
        # Arrange
        # Skapa test data med exakt 300 time steps
        time_steps = 300
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att exakt 1 window skapas
        self.assertEqual(len(windows), 1, "Ska skapa exakt 1 window från 300 time steps")
        self.assertEqual(windows.shape[1], 300, "Window size ska vara exakt 300")
        self.assertEqual(windows.shape[2], 16, "Timeseries features ska vara 16")
        
        # Verifiera att window använder alla time steps
        self.assertEqual(metadata[0]['start_time_step'], 0, "Window ska börja på step 0")
        self.assertEqual(metadata[0]['end_time_step'], 300, "Window ska sluta på step 300")
        
        print("✅ T030 PASSED: Exact boundary window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_multiple_windows(self):
        """
        Verifiera window size med flera windows
        """
        # Arrange
        # Skapa test data med 1000 time steps för att få flera windows
        time_steps = 1000
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att alla windows har samma size
        for i, window in enumerate(windows):
            self.assertEqual(len(window), 300, f"Window {i} ska vara exakt 300 steg")
            self.assertEqual(window.shape, (300, 16), f"Window {i} ska ha shape (300, 16)")
        
        # Verifiera att antal windows är korrekt
        expected_windows = (time_steps - 300) // 30 + 1  # (1000 - 300) // 30 + 1 = 24
        self.assertEqual(len(windows), expected_windows, 
                       f"Ska skapa {expected_windows} windows från {time_steps} time steps")
        
        print("✅ T030 PASSED: Multiple windows window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_validation(self):
        """
        Verifiera window shape validation
        """
        # Arrange
        # Skapa test data
        time_steps = 1000
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att window shape validation fungerar
        self.assertTrue(self.window_creator.validate_window_shape(windows),
                       "Window shape validation ska passera")
        
        # Testa med felaktig shape
        wrong_windows = np.random.randn(5, 200, 16)  # Fel window size
        self.assertFalse(self.window_creator.validate_window_shape(wrong_windows),
                        "Window shape validation ska misslyckas med fel window size")
        
        wrong_windows2 = np.random.randn(5, 300, 14)  # Fel antal features
        self.assertFalse(self.window_creator.validate_window_shape(wrong_windows2),
                        "Window shape validation ska misslyckas med fel antal features")
        
        print("✅ T030 PASSED: Window shape validation fungerar korrekt")
    
    def test_t030_window_size_300s_edge_cases(self):
        """
        Verifiera window size med edge cases
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: Mindre än 300 time steps
        timeseries_data_short = np.random.randn(200, 16)
        windows_short, _, metadata_short = self.window_creator.create_sliding_windows(
            timeseries_data_short, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_short), 0, "Ska inte skapa windows från < 300 time steps")
        self.assertEqual(len(metadata_short), 0, "Ska inte ha metadata från < 300 time steps")
        
        # Test case 2: Exakt 301 time steps
        timeseries_data_301 = np.random.randn(301, 16)
        windows_301, _, metadata_301 = self.window_creator.create_sliding_windows(
            timeseries_data_301, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_301), 1, "Ska skapa 1 window från 301 time steps")
        self.assertEqual(windows_301.shape[1], 300, "Window ska vara exakt 300 steg")
        
        print("✅ T030 PASSED: Edge cases window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_dataframe(self):
        """
        Verifiera window size med pandas DataFrame
        """
        # Arrange
        # Skapa DataFrame med timeseries data
        time_steps = 1000
        timeseries_columns = [f'feature_{i}' for i in range(16)]
        
        df = pd.DataFrame(
            np.random.randn(time_steps, 16),
            columns=timeseries_columns
        )
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_windows_from_dataframe(
            df, timeseries_columns, static_data
        )
        
        # Assert
        # Verifiera att windows har korrekt shape
        self.assertEqual(windows.shape[1], 300, "Window size ska vara exakt 300 från DataFrame")
        self.assertEqual(windows.shape[2], 16, "Timeseries features ska vara 16 från DataFrame")
        
        # Verifiera att antal windows är korrekt
        expected_windows = (time_steps - 300) // 30 + 1
        self.assertEqual(len(windows), expected_windows, 
                       f"Ska skapa {expected_windows} windows från DataFrame")
        
        print("✅ T030 PASSED: DataFrame window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_configuration(self):
        """
        Verifiera window size configuration
        """
        # Arrange
        # Skapa custom config med 300s window
        config = WindowConfig(
            window_size_seconds=300,
            step_size_seconds=30,
            timeseries_features=16,
            static_features=6
        )
        
        window_creator_custom = MasterPOCWindowCreator(config)
        
        # Skapa test data
        time_steps = 1000
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = window_creator_custom.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att configuration används korrekt
        self.assertEqual(windows.shape[1], 300, "Custom config window size ska vara 300")
        self.assertEqual(windows.shape[2], 16, "Custom config timeseries features ska vara 16")
        
        # Verifiera att config värden är korrekta
        self.assertEqual(window_creator_custom.config.window_size_seconds, 300,
                        "Config window_size_seconds ska vara 300")
        
        print("✅ T030 PASSED: Configuration window size 300s fungerar korrekt")
    
    def test_t030_window_size_300s_comprehensive(self):
        """
        Omfattande test av window size 300s
        """
        # Arrange
        # Testa med olika data storlekar
        test_cases = [
            (300, 1),   # Exakt 300 steg
            (600, 11),  # 600 steg
            (1000, 24), # 1000 steg
            (1500, 41), # 1500 steg
        ]
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act & Assert
        for time_steps, expected_windows in test_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            # Verifiera antal windows
            self.assertEqual(len(windows), expected_windows,
                           f"Ska skapa {expected_windows} windows från {time_steps} time steps")
            
            # Verifiera att alla windows har size 300
            for i, window in enumerate(windows):
                self.assertEqual(len(window), 300,
                               f"Window {i} från {time_steps} steg ska vara 300 steg")
                self.assertEqual(window.shape, (300, 16),
                               f"Window {i} från {time_steps} steg ska ha shape (300, 16)")
            
            # Verifiera metadata
            for meta in metadata:
                self.assertEqual(meta['duration_seconds'], 300,
                               f"Window duration ska vara 300 sekunder")
        
        print("✅ T030 PASSED: Comprehensive window size 300s test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
