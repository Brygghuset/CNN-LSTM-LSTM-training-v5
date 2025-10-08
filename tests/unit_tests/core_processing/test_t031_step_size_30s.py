#!/usr/bin/env python3
"""
T031: Test Step Size 30s
Verifiera att step size är 30 sekunder (10% overlap)
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

class TestT031StepSize30s(unittest.TestCase):
    """T031: Test Step Size 30s"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t031_step_size_30s_basic(self):
        """
        T031: Test Step Size 30s
        Verifiera att step size är 30 sekunder (10% overlap)
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
        # Verifiera att step size validation fungerar
        self.assertTrue(self.window_creator.validate_step_size(metadata),
                       "Step size validation ska passera")
        
        # Verifiera att step size är 30 mellan windows
        if len(metadata) >= 2:
            for i in range(1, len(metadata)):
                step_diff = metadata[i]['start_time_step'] - metadata[i-1]['start_time_step']
                self.assertEqual(step_diff, 30, 
                               f"Step size mellan window {i-1} och {i} ska vara 30")
        
        # Verifiera att overlap är 10% (270 steg overlap av 300)
        overlap_percentage = self.window_creator.get_window_overlap_percentage()
        self.assertAlmostEqual(overlap_percentage, 90.0, places=1,
                             msg=f"Overlap ska vara 90% (270/300), fick {overlap_percentage}%")
        
        print("✅ T031 PASSED: Basic step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_overlap_calculation(self):
        """
        Verifiera overlap beräkning för 30s step size
        """
        # Arrange
        window_size = 300  # 300 sekunder
        step_size = 30     # 30 sekunder
        
        # Act
        overlap_percentage = self.window_creator.get_window_overlap_percentage()
        
        # Assert
        # Beräkna förväntad overlap
        overlap_steps = window_size - step_size  # 300 - 30 = 270
        expected_overlap = (overlap_steps / window_size) * 100  # (270/300) * 100 = 90%
        
        self.assertAlmostEqual(overlap_percentage, expected_overlap, places=1,
                             msg=f"Overlap ska vara {expected_overlap}%, fick {overlap_percentage}%")
        
        # Verifiera att overlap är 90%
        self.assertAlmostEqual(overlap_percentage, 90.0, places=1,
                             msg="Overlap ska vara 90% för 30s step size")
        
        print("✅ T031 PASSED: Overlap calculation step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_window_positions(self):
        """
        Verifiera att windows positioneras korrekt med 30s step size
        """
        # Arrange
        # Skapa test data med 600 time steps för att få exakt 11 windows
        time_steps = 600
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att antal windows är korrekt
        expected_windows = (time_steps - 300) // 30 + 1  # (600 - 300) // 30 + 1 = 11
        self.assertEqual(len(windows), expected_windows,
                       f"Ska skapa {expected_windows} windows från {time_steps} time steps")
        
        # Verifiera att window positioner är korrekta
        expected_positions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
        for i, meta in enumerate(metadata):
            self.assertEqual(meta['start_time_step'], expected_positions[i],
                           f"Window {i} ska börja på position {expected_positions[i]}")
            self.assertEqual(meta['end_time_step'], expected_positions[i] + 300,
                           f"Window {i} ska sluta på position {expected_positions[i] + 300}")
        
        print("✅ T031 PASSED: Window positions step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_consecutive_windows(self):
        """
        Verifiera att konsekutiva windows har korrekt step size
        """
        # Arrange
        # Skapa test data med 900 time steps
        time_steps = 900
        timeseries_data = np.random.randn(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att alla konsekutiva windows har step size 30
        for i in range(1, len(metadata)):
            prev_start = metadata[i-1]['start_time_step']
            curr_start = metadata[i]['start_time_step']
            step_diff = curr_start - prev_start
            
            self.assertEqual(step_diff, 30,
                           f"Step size mellan window {i-1} och {i} ska vara 30, fick {step_diff}")
        
        # Verifiera att step size validation fungerar
        self.assertTrue(self.window_creator.validate_step_size(metadata),
                       "Step size validation ska passera för konsekutiva windows")
        
        print("✅ T031 PASSED: Consecutive windows step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_edge_cases(self):
        """
        Verifiera step size med edge cases
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: Exakt 300 time steps (1 window)
        timeseries_data_300 = np.random.randn(300, 16)
        windows_300, _, metadata_300 = self.window_creator.create_sliding_windows(
            timeseries_data_300, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_300), 1, "Ska skapa 1 window från 300 time steps")
        self.assertEqual(len(metadata_300), 1, "Ska ha 1 metadata entry")
        # Ingen step size validering möjlig med endast 1 window
        self.assertTrue(self.window_creator.validate_step_size(metadata_300),
                       "Step size validation ska passera med 1 window")
        
        # Test case 2: 330 time steps (2 windows)
        timeseries_data_330 = np.random.randn(330, 16)
        windows_330, _, metadata_330 = self.window_creator.create_sliding_windows(
            timeseries_data_330, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_330), 2, "Ska skapa 2 windows från 330 time steps")
        self.assertEqual(metadata_330[0]['start_time_step'], 0, "Första window ska börja på 0")
        self.assertEqual(metadata_330[1]['start_time_step'], 30, "Andra window ska börja på 30")
        
        # Verifiera step size mellan de två windows
        step_diff = metadata_330[1]['start_time_step'] - metadata_330[0]['start_time_step']
        self.assertEqual(step_diff, 30, "Step size mellan windows ska vara 30")
        
        print("✅ T031 PASSED: Edge cases step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_custom_config(self):
        """
        Verifiera step size med custom configuration
        """
        # Arrange
        # Skapa custom config med 30s step size
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
        # Verifiera att custom config används
        self.assertEqual(window_creator_custom.config.step_size_seconds, 30,
                        "Custom config step_size_seconds ska vara 30")
        
        # Verifiera att step size är korrekt
        self.assertTrue(window_creator_custom.validate_step_size(metadata),
                       "Custom config step size validation ska passera")
        
        # Verifiera overlap procent
        overlap_percentage = window_creator_custom.get_window_overlap_percentage()
        self.assertAlmostEqual(overlap_percentage, 90.0, places=1,
                             msg="Custom config overlap ska vara 90%")
        
        print("✅ T031 PASSED: Custom config step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_dataframe(self):
        """
        Verifiera step size med pandas DataFrame
        """
        # Arrange
        # Skapa DataFrame med timeseries data
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
        # Verifiera att step size är korrekt från DataFrame
        self.assertTrue(self.window_creator.validate_step_size(metadata),
                       "Step size validation ska passera från DataFrame")
        
        # Verifiera att antal windows är korrekt
        expected_windows = (time_steps - 300) // 30 + 1
        self.assertEqual(len(windows), expected_windows,
                       f"Ska skapa {expected_windows} windows från DataFrame")
        
        print("✅ T031 PASSED: DataFrame step size 30s fungerar korrekt")
    
    def test_t031_step_size_30s_comprehensive(self):
        """
        Omfattande test av step size 30s
        """
        # Arrange
        # Testa med olika data storlekar
        test_cases = [
            (300, 1, []),      # Exakt 300 steg - 1 window
            (330, 2, [0, 30]), # 330 steg - 2 windows
            (600, 11, list(range(0, 301, 30))),  # 600 steg - 11 windows
            (900, 21, list(range(0, 601, 30))),  # 900 steg - 21 windows
        ]
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act & Assert
        for time_steps, expected_windows, expected_positions in test_cases:
            timeseries_data = np.random.randn(time_steps, 16)
            
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            # Verifiera antal windows
            self.assertEqual(len(windows), expected_windows,
                           f"Ska skapa {expected_windows} windows från {time_steps} time steps")
            
            # Verifiera window positioner
            for i, expected_pos in enumerate(expected_positions):
                self.assertEqual(metadata[i]['start_time_step'], expected_pos,
                               f"Window {i} från {time_steps} steg ska börja på {expected_pos}")
            
            # Verifiera step size mellan windows
            if len(metadata) >= 2:
                self.assertTrue(self.window_creator.validate_step_size(metadata),
                               f"Step size validation ska passera för {time_steps} time steps")
        
        print("✅ T031 PASSED: Comprehensive step size 30s test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
