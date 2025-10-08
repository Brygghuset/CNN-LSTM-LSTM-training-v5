#!/usr/bin/env python3
"""
T033: Test Window Shape [300, 16]
Verifiera att window shape är [300, 16] för timeseries
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

class TestT033WindowShape300x16(unittest.TestCase):
    """T033: Test Window Shape [300, 16]"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t033_window_shape_300x16_basic(self):
        """
        T033: Test Window Shape [300, 16]
        Verifiera att window shape är [300, 16] för timeseries
        """
        # Arrange
        # Skapa test data med 16 timeseries features
        time_steps = 1000
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att windows har korrekt shape [n_windows, 300, 16]
        self.assertEqual(len(windows.shape), 3, "Windows ska ha 3 dimensioner")
        self.assertEqual(windows.shape[1], 300, "Window size ska vara 300")
        self.assertEqual(windows.shape[2], 16, "Timeseries features ska vara 16")
        
        # Verifiera att varje window har shape [300, 16]
        for i, window in enumerate(windows):
            self.assertEqual(window.shape, (300, 16),
                           f"Window {i} ska ha shape (300, 16)")
        
        # Verifiera att window shape validation fungerar
        self.assertTrue(self.window_creator.validate_window_shape(windows),
                       "Window shape validation ska passera")
        
        print("✅ T033 PASSED: Basic window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_exact_boundary(self):
        """
        Verifiera window shape med exakt 300 time steps
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
        # Verifiera att exakt 1 window skapas med shape [1, 300, 16]
        self.assertEqual(windows.shape, (1, 300, 16),
                        "Windows ska ha shape (1, 300, 16) från 300 time steps")
        
        # Verifiera att window har korrekt shape
        self.assertEqual(windows[0].shape, (300, 16),
                        "Enskild window ska ha shape (300, 16)")
        
        # Verifiera att alla 16 features finns
        self.assertEqual(windows[0].shape[1], 16,
                        "Window ska ha 16 timeseries features")
        
        print("✅ T033 PASSED: Exact boundary window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_multiple_windows(self):
        """
        Verifiera window shape med flera windows
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
        # Verifiera att alla windows har samma shape
        expected_windows = (time_steps - 300) // 30 + 1  # (1000 - 300) // 30 + 1 = 24
        self.assertEqual(windows.shape, (expected_windows, 300, 16),
                        f"Windows ska ha shape ({expected_windows}, 300, 16)")
        
        # Verifiera att varje window har shape [300, 16]
        for i, window in enumerate(windows):
            self.assertEqual(window.shape, (300, 16),
                           f"Window {i} ska ha shape (300, 16)")
        
        # Verifiera att alla 16 features finns i varje window
        for i, window in enumerate(windows):
            self.assertEqual(window.shape[1], 16,
                           f"Window {i} ska ha 16 timeseries features")
        
        print("✅ T033 PASSED: Multiple windows shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_feature_consistency(self):
        """
        Verifiera att alla 16 timeseries features är konsistenta
        """
        # Arrange
        # Skapa test data med kända värden för varje feature
        time_steps = 600
        timeseries_data = np.zeros((time_steps, 16))
        
        # Sätt unika värden för varje feature
        for feature_idx in range(16):
            timeseries_data[:, feature_idx] = feature_idx * 100
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att alla windows har 16 features
        for i, window in enumerate(windows):
            self.assertEqual(window.shape[1], 16,
                           f"Window {i} ska ha 16 features")
        
        # Verifiera att varje feature har korrekt värden
        for window_idx, window in enumerate(windows):
            for feature_idx in range(16):
                expected_value = feature_idx * 100
                # Alla värden i denna feature ska vara samma
                self.assertTrue(np.all(window[:, feature_idx] == expected_value),
                               f"Window {window_idx}, feature {feature_idx} ska ha värde {expected_value}")
        
        print("✅ T033 PASSED: Feature consistency window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_validation(self):
        """
        Verifiera window shape validation
        """
        # Arrange
        # Skapa korrekt test data
        time_steps = 600
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att korrekt shape valideras
        self.assertTrue(self.window_creator.validate_window_shape(windows),
                       "Korrekt window shape ska valideras")
        
        # Testa med felaktiga shapes
        wrong_windows_1 = np.random.randn(5, 200, 16)  # Fel window size
        self.assertFalse(self.window_creator.validate_window_shape(wrong_windows_1),
                        "Fel window size ska inte valideras")
        
        wrong_windows_2 = np.random.randn(5, 300, 14)  # Fel antal features
        self.assertFalse(self.window_creator.validate_window_shape(wrong_windows_2),
                        "Fel antal features ska inte valideras")
        
        wrong_windows_3 = np.random.randn(5, 300)  # Fel antal dimensioner
        self.assertFalse(self.window_creator.validate_window_shape(wrong_windows_3),
                        "Fel antal dimensioner ska inte valideras")
        
        print("✅ T033 PASSED: Window shape validation fungerar korrekt")
    
    def test_t033_window_shape_300x16_edge_cases(self):
        """
        Verifiera window shape med edge cases
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
        self.assertEqual(windows_short.shape, (0, 300, 16),
                        "Empty windows ska ha shape (0, 300, 16)")
        
        # Test case 2: Exakt 301 time steps
        timeseries_data_301 = np.random.randn(301, 16)
        windows_301, _, metadata_301 = self.window_creator.create_sliding_windows(
            timeseries_data_301, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_301), 1, "Ska skapa 1 window från 301 time steps")
        self.assertEqual(windows_301.shape, (1, 300, 16),
                        "Windows ska ha shape (1, 300, 16)")
        self.assertEqual(windows_301[0].shape, (300, 16),
                        "Enskild window ska ha shape (300, 16)")
        
        print("✅ T033 PASSED: Edge cases window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_dataframe(self):
        """
        Verifiera window shape med pandas DataFrame
        """
        # Arrange
        # Skapa DataFrame med 16 timeseries kolumner
        time_steps = 600
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
        # Verifiera att windows har korrekt shape från DataFrame
        expected_windows = (time_steps - 300) // 30 + 1
        self.assertEqual(windows.shape, (expected_windows, 300, 16),
                        f"DataFrame windows ska ha shape ({expected_windows}, 300, 16)")
        
        # Verifiera att varje window har shape [300, 16]
        for i, window in enumerate(windows):
            self.assertEqual(window.shape, (300, 16),
                           f"DataFrame window {i} ska ha shape (300, 16)")
        
        # Verifiera att alla 16 features finns
        for i, window in enumerate(windows):
            self.assertEqual(window.shape[1], 16,
                           f"DataFrame window {i} ska ha 16 features")
        
        print("✅ T033 PASSED: DataFrame window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_custom_config(self):
        """
        Verifiera window shape med custom configuration
        """
        # Arrange
        # Skapa custom config med 16 timeseries features
        config = WindowConfig(
            window_size_seconds=300,
            step_size_seconds=30,
            timeseries_features=16,
            static_features=6
        )
        
        window_creator_custom = MasterPOCWindowCreator(config)
        
        # Skapa test data
        time_steps = 600
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = window_creator_custom.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att custom config ger korrekt shape
        expected_windows = (time_steps - 300) // 30 + 1
        self.assertEqual(windows.shape, (expected_windows, 300, 16),
                        f"Custom config windows ska ha shape ({expected_windows}, 300, 16)")
        
        # Verifiera att config värden är korrekta
        self.assertEqual(window_creator_custom.config.timeseries_features, 16,
                        "Config timeseries_features ska vara 16")
        self.assertEqual(window_creator_custom.config.window_size_seconds, 300,
                        "Config window_size_seconds ska vara 300")
        
        # Verifiera att validation fungerar
        self.assertTrue(window_creator_custom.validate_window_shape(windows),
                       "Custom config window shape validation ska passera")
        
        print("✅ T033 PASSED: Custom config window shape [300, 16] fungerar korrekt")
    
    def test_t033_window_shape_300x16_comprehensive(self):
        """
        Omfattande test av window shape [300, 16]
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
            
            # Verifiera att alla windows har shape [300, 16]
            for i, window in enumerate(windows):
                self.assertEqual(window.shape, (300, 16),
                               f"Window {i} från {time_steps} steg ska ha shape (300, 16)")
                
                # Verifiera att alla 16 features finns
                self.assertEqual(window.shape[1], 16,
                               f"Window {i} från {time_steps} steg ska ha 16 features")
            
            # Verifiera att total shape är korrekt
            self.assertEqual(windows.shape, (expected_windows, 300, 16),
                           f"Windows från {time_steps} steg ska ha shape ({expected_windows}, 300, 16)")
            
            # Verifiera att validation fungerar
            self.assertTrue(self.window_creator.validate_window_shape(windows),
                           f"Window shape validation ska passera för {time_steps} time steps")
        
        print("✅ T033 PASSED: Comprehensive window shape [300, 16] test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
