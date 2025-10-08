#!/usr/bin/env python3
"""
T032: Test Sliding Window Logic
Verifiera korrekt sliding window implementation
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

class TestT032SlidingWindowLogic(unittest.TestCase):
    """T032: Test Sliding Window Logic"""
    
    def setUp(self):
        """Setup för varje test."""
        self.window_creator = create_master_poc_window_creator()
    
    def test_t032_sliding_window_logic_basic(self):
        """
        T032: Test Sliding Window Logic
        Verifiera korrekt sliding window implementation
        """
        # Arrange
        # Skapa deterministisk test data för att verifiera sliding window logic
        time_steps = 600
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att sliding window logic fungerar korrekt
        expected_windows = (time_steps - 300) // 30 + 1  # (600 - 300) // 30 + 1 = 11
        self.assertEqual(len(windows), expected_windows,
                       f"Ska skapa {expected_windows} windows från {time_steps} time steps")
        
        # Verifiera att första window börjar på rätt position
        self.assertEqual(metadata[0]['start_time_step'], 0,
                        "Första window ska börja på position 0")
        self.assertEqual(metadata[0]['end_time_step'], 300,
                        "Första window ska sluta på position 300")
        
        # Verifiera att andra window börjar på rätt position
        self.assertEqual(metadata[1]['start_time_step'], 30,
                        "Andra window ska börja på position 30")
        self.assertEqual(metadata[1]['end_time_step'], 330,
                        "Andra window ska sluta på position 330")
        
        # Verifiera att data i windows är korrekt
        # Första window ska innehålla data från position 0-299
        first_window_data = windows[0]
        expected_first_data = np.arange(0, 300 * 16).reshape(300, 16)
        np.testing.assert_array_equal(first_window_data, expected_first_data,
                                     err_msg="Första window ska innehålla korrekt data")
        
        print("✅ T032 PASSED: Basic sliding window logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_overlap(self):
        """
        Verifiera att sliding windows har korrekt overlap
        """
        # Arrange
        # Skapa test data med unika värden för varje position
        time_steps = 600
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera overlap mellan första och andra window
        if len(windows) >= 2:
            first_window = windows[0]  # Position 0-299
            second_window = windows[1]  # Position 30-329
            
            # Overlap ska vara position 30-299 (270 steg)
            overlap_first = first_window[30:300]  # Position 30-299 från första window
            overlap_second = second_window[0:270]  # Position 0-269 från andra window
            
            np.testing.assert_array_equal(overlap_first, overlap_second,
                                         err_msg="Overlap mellan windows ska vara identisk")
            
            # Verifiera att overlap är 270 steg (90% av 300)
            self.assertEqual(len(overlap_first), 270,
                           "Overlap ska vara 270 steg")
            self.assertEqual(len(overlap_second), 270,
                           "Overlap ska vara 270 steg")
        
        print("✅ T032 PASSED: Sliding window overlap fungerar korrekt")
    
    def test_t032_sliding_window_logic_consecutive_windows(self):
        """
        Verifiera att konsekutiva windows följer sliding window logic
        """
        # Arrange
        # Skapa test data med 900 time steps för att få flera windows
        time_steps = 900
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att alla konsekutiva windows följer sliding window logic
        for i in range(1, len(windows)):
            prev_window = windows[i-1]
            curr_window = windows[i]
            
            # Verifiera att windows är separerade med 30 steg
            prev_start = metadata[i-1]['start_time_step']
            curr_start = metadata[i]['start_time_step']
            self.assertEqual(curr_start - prev_start, 30,
                           f"Windows {i-1} och {i} ska vara separerade med 30 steg")
            
            # Verifiera att overlap är korrekt
            prev_end = metadata[i-1]['end_time_step']
            curr_start = metadata[i]['start_time_step']
            overlap_steps = prev_end - curr_start  # 300 - 30 = 270
            self.assertEqual(overlap_steps, 270,
                           f"Overlap mellan windows {i-1} och {i} ska vara 270 steg")
            
            # Verifiera att overlap data är identisk
            prev_overlap = prev_window[30:300]  # Position 30-299 från föregående window
            curr_overlap = curr_window[0:270]   # Position 0-269 från nuvarande window
            
            np.testing.assert_array_equal(prev_overlap, curr_overlap,
                                         err_msg=f"Overlap data mellan windows {i-1} och {i} ska vara identisk")
        
        print("✅ T032 PASSED: Consecutive windows sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_boundary_conditions(self):
        """
        Verifiera sliding window logic vid boundary conditions
        """
        # Arrange
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Test case 1: Exakt 300 time steps
        timeseries_data_300 = np.arange(300 * 16).reshape(300, 16)
        windows_300, _, metadata_300 = self.window_creator.create_sliding_windows(
            timeseries_data_300, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_300), 1, "Ska skapa 1 window från 300 time steps")
        self.assertEqual(metadata_300[0]['start_time_step'], 0,
                        "Window ska börja på position 0")
        self.assertEqual(metadata_300[0]['end_time_step'], 300,
                        "Window ska sluta på position 300")
        
        # Test case 2: 330 time steps
        timeseries_data_330 = np.arange(330 * 16).reshape(330, 16)
        windows_330, _, metadata_330 = self.window_creator.create_sliding_windows(
            timeseries_data_330, static_data
        )
        
        # Assert
        self.assertEqual(len(windows_330), 2, "Ska skapa 2 windows från 330 time steps")
        self.assertEqual(metadata_330[0]['start_time_step'], 0,
                        "Första window ska börja på position 0")
        self.assertEqual(metadata_330[0]['end_time_step'], 300,
                        "Första window ska sluta på position 300")
        self.assertEqual(metadata_330[1]['start_time_step'], 30,
                        "Andra window ska börja på position 30")
        self.assertEqual(metadata_330[1]['end_time_step'], 330,
                        "Andra window ska sluta på position 330")
        
        print("✅ T032 PASSED: Boundary conditions sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_data_integrity(self):
        """
        Verifiera att data integritet bevaras i sliding windows
        """
        # Arrange
        # Skapa test data med kända värden
        time_steps = 600
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att data i windows matchar original data
        for i, window in enumerate(windows):
            start_pos = metadata[i]['start_time_step']
            end_pos = metadata[i]['end_time_step']
            
            # Extrahera motsvarande data från original
            original_data = timeseries_data[start_pos:end_pos]
            
            # Verifiera att window data matchar original data
            np.testing.assert_array_equal(window, original_data,
                                        err_msg=f"Window {i} data ska matcha original data")
        
        # Verifiera att static data replikeras korrekt
        for static_window in static_windows:
            np.testing.assert_array_equal(static_window, static_data,
                                        err_msg="Static data ska replikeras korrekt")
        
        print("✅ T032 PASSED: Data integrity sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_step_progression(self):
        """
        Verifiera att step progression följer sliding window logic
        """
        # Arrange
        # Skapa test data med 1200 time steps för att få många windows
        time_steps = 1200
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att step progression är korrekt
        expected_steps = list(range(0, time_steps - 300 + 1, 30))
        actual_steps = [meta['start_time_step'] for meta in metadata]
        
        self.assertEqual(actual_steps, expected_steps,
                       f"Step progression ska vara {expected_steps}, fick {actual_steps}")
        
        # Verifiera att varje step är 30 steg från föregående
        for i in range(1, len(metadata)):
            step_diff = metadata[i]['start_time_step'] - metadata[i-1]['start_time_step']
            self.assertEqual(step_diff, 30,
                           f"Step progression ska vara 30 mellan windows {i-1} och {i}")
        
        print("✅ T032 PASSED: Step progression sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_custom_config(self):
        """
        Verifiera sliding window logic med custom configuration
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
        
        # Skapa test data
        time_steps = 600
        timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = window_creator_custom.create_sliding_windows(
            timeseries_data, static_data
        )
        
        # Assert
        # Verifiera att custom config ger samma resultat som default
        windows_default, _, metadata_default = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        self.assertEqual(len(windows), len(windows_default),
                       "Custom config ska ge samma antal windows som default")
        
        # Verifiera att window positions är identiska
        for i, (meta_custom, meta_default) in enumerate(zip(metadata, metadata_default)):
            self.assertEqual(meta_custom['start_time_step'], meta_default['start_time_step'],
                           f"Window {i} start position ska vara identisk")
            self.assertEqual(meta_custom['end_time_step'], meta_default['end_time_step'],
                           f"Window {i} end position ska vara identisk")
        
        print("✅ T032 PASSED: Custom config sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_dataframe(self):
        """
        Verifiera sliding window logic med pandas DataFrame
        """
        # Arrange
        # Skapa DataFrame med deterministisk data
        time_steps = 600
        timeseries_columns = [f'feature_{i}' for i in range(16)]
        
        df = pd.DataFrame(
            np.arange(time_steps * 16).reshape(time_steps, 16),
            columns=timeseries_columns
        )
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act
        windows, static_windows, metadata = self.window_creator.create_windows_from_dataframe(
            df, timeseries_columns, static_data
        )
        
        # Assert
        # Verifiera att DataFrame ger samma resultat som numpy array
        timeseries_data = df[timeseries_columns].values
        windows_numpy, _, metadata_numpy = self.window_creator.create_sliding_windows(
            timeseries_data, static_data
        )
        
        self.assertEqual(len(windows), len(windows_numpy),
                       "DataFrame ska ge samma antal windows som numpy array")
        
        # Verifiera att window data är identisk
        np.testing.assert_array_equal(windows, windows_numpy,
                                     err_msg="DataFrame windows ska vara identiska med numpy windows")
        
        # Verifiera att metadata är identisk
        for i, (meta_df, meta_numpy) in enumerate(zip(metadata, metadata_numpy)):
            self.assertEqual(meta_df['start_time_step'], meta_numpy['start_time_step'],
                           f"DataFrame window {i} start position ska vara identisk")
            self.assertEqual(meta_df['end_time_step'], meta_numpy['end_time_step'],
                           f"DataFrame window {i} end position ska vara identisk")
        
        print("✅ T032 PASSED: DataFrame sliding logic fungerar korrekt")
    
    def test_t032_sliding_window_logic_comprehensive(self):
        """
        Omfattande test av sliding window logic
        """
        # Arrange
        # Testa med olika data storlekar
        test_cases = [
            (300, 1, [0]),           # Exakt 300 steg
            (330, 2, [0, 30]),       # 330 steg
            (600, 11, list(range(0, 301, 30))),  # 600 steg
            (900, 21, list(range(0, 601, 30))),  # 900 steg
        ]
        
        static_data = np.array([25, 1, 170, 70, 24, 2])
        
        # Act & Assert
        for time_steps, expected_windows, expected_steps in test_cases:
            timeseries_data = np.arange(time_steps * 16).reshape(time_steps, 16)
            
            windows, static_windows, metadata = self.window_creator.create_sliding_windows(
                timeseries_data, static_data
            )
            
            # Verifiera antal windows
            self.assertEqual(len(windows), expected_windows,
                           f"Ska skapa {expected_windows} windows från {time_steps} time steps")
            
            # Verifiera step positions
            actual_steps = [meta['start_time_step'] for meta in metadata]
            self.assertEqual(actual_steps, expected_steps,
                           f"Step positions ska vara {expected_steps} för {time_steps} time steps")
            
            # Verifiera att varje window har korrekt data
            for i, window in enumerate(windows):
                start_pos = metadata[i]['start_time_step']
                end_pos = metadata[i]['end_time_step']
                expected_data = timeseries_data[start_pos:end_pos]
                
                np.testing.assert_array_equal(window, expected_data,
                                            err_msg=f"Window {i} från {time_steps} steg ska ha korrekt data")
            
            # Verifiera sliding window logic mellan konsekutiva windows
            if len(windows) >= 2:
                for i in range(1, len(windows)):
                    prev_window = windows[i-1]
                    curr_window = windows[i]
                    
                    # Verifiera overlap
                    overlap_first = prev_window[30:300]
                    overlap_second = curr_window[0:270]
                    
                    np.testing.assert_array_equal(overlap_first, overlap_second,
                                                err_msg=f"Overlap mellan windows {i-1} och {i} ska vara identisk")
        
        print("✅ T032 PASSED: Comprehensive sliding window logic test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
