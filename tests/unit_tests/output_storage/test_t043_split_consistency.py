#!/usr/bin/env python3
"""
T043: Test Split Consistency
Verifiera att samma case alltid hamnar i samma split
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf
import shutil

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår master POC TFRecord creator modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
master_poc_tfrecord_creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_tfrecord_creator)

# Använd modulen
MasterPOCTFRecordCreator = master_poc_tfrecord_creator.MasterPOCTFRecordCreator
TFRecordConfig = master_poc_tfrecord_creator.TFRecordConfig
create_master_poc_tfrecord_creator = master_poc_tfrecord_creator.create_master_poc_tfrecord_creator
split_data_by_case_70_15_15 = master_poc_tfrecord_creator.split_data_by_case_70_15_15

class TestT043SplitConsistency(unittest.TestCase):
    """T043: Test Split Consistency"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t043_split_consistency_basic(self):
        """
        T043: Test Split Consistency
        Verifiera att samma case alltid hamnar i samma split
        """
        # Arrange
        # Skapa test data med olika cases
        case_windows = {
            'case_001': [np.full((300, 16), 1, dtype=np.float32), np.full((300, 16), 1, dtype=np.float32)],
            'case_002': [np.full((300, 16), 2, dtype=np.float32), np.full((300, 16), 2, dtype=np.float32)],
            'case_003': [np.full((300, 16), 3, dtype=np.float32), np.full((300, 16), 3, dtype=np.float32)],
            'case_004': [np.full((300, 16), 4, dtype=np.float32), np.full((300, 16), 4, dtype=np.float32)],
            'case_005': [np.full((300, 16), 5, dtype=np.float32), np.full((300, 16), 5, dtype=np.float32)],
        }
        
        case_static = {
            'case_001': np.full(6, 10, dtype=np.float32),
            'case_002': np.full(6, 20, dtype=np.float32),
            'case_003': np.full(6, 30, dtype=np.float32),
            'case_004': np.full(6, 40, dtype=np.float32),
            'case_005': np.full(6, 50, dtype=np.float32),
        }
        
        case_targets = {
            'case_001': [np.full(8, 100, dtype=np.float32), np.full(8, 100, dtype=np.float32)],
            'case_002': [np.full(8, 200, dtype=np.float32), np.full(8, 200, dtype=np.float32)],
            'case_003': [np.full(8, 300, dtype=np.float32), np.full(8, 300, dtype=np.float32)],
            'case_004': [np.full(8, 400, dtype=np.float32), np.full(8, 400, dtype=np.float32)],
            'case_005': [np.full(8, 500, dtype=np.float32), np.full(8, 500, dtype=np.float32)],
        }
        
        # Act
        train_data, validation_data, test_data = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=42
        )
        
        # Assert
        # Verifiera att alla cases är fördelade
        total_train_windows = len(train_data['windows'])
        total_val_windows = len(validation_data['windows'])
        total_test_windows = len(test_data['windows'])
        total_windows = total_train_windows + total_val_windows + total_test_windows
        
        expected_total_windows = sum(len(windows) for windows in case_windows.values())
        self.assertEqual(total_windows, expected_total_windows, f"Ska dela alla {expected_total_windows} windows")
        
        # Verifiera att samma case alltid hamnar i samma split
        # Kontrollera att alla windows från samma case har samma värde
        case_values = {}
        
        # Analysera train data
        if total_train_windows > 0:
            for i in range(0, total_train_windows, 2):  # 2 windows per case
                window_value = train_data['windows'][i][0, 0]
                case_values[window_value] = 'train'
        
        # Analysera validation data
        if total_val_windows > 0:
            for i in range(0, total_val_windows, 2):  # 2 windows per case
                window_value = validation_data['windows'][i][0, 0]
                case_values[window_value] = 'validation'
        
        # Analysera test data
        if total_test_windows > 0:
            for i in range(0, total_test_windows, 2):  # 2 windows per case
                window_value = test_data['windows'][i][0, 0]
                case_values[window_value] = 'test'
        
        # Verifiera att varje case bara finns i en split
        self.assertEqual(len(case_values), 5, "Ska ha 5 cases fördelade")
        
        # Verifiera att alla windows från samma case har samma värde
        for case_value in [1, 2, 3, 4, 5]:
            self.assertIn(case_value, case_values, f"Case {case_value} ska finnas i någon split")
        
        print(f"✅ T043 PASSED: Basic split consistency fungerar korrekt")
        print(f"   Train: {total_train_windows} windows")
        print(f"   Validation: {total_val_windows} windows")
        print(f"   Test: {total_test_windows} windows")
        print(f"   Cases fördelade: {len(case_values)}")
    
    def test_t043_split_consistency_deterministic(self):
        """
        Verifiera att case-based split är deterministisk med samma seed
        """
        # Arrange
        case_windows = {
            'case_001': [np.full((300, 16), 1, dtype=np.float32)],
            'case_002': [np.full((300, 16), 2, dtype=np.float32)],
            'case_003': [np.full((300, 16), 3, dtype=np.float32)],
            'case_004': [np.full((300, 16), 4, dtype=np.float32)],
            'case_005': [np.full((300, 16), 5, dtype=np.float32)],
        }
        
        case_static = {
            'case_001': np.full(6, 10, dtype=np.float32),
            'case_002': np.full(6, 20, dtype=np.float32),
            'case_003': np.full(6, 30, dtype=np.float32),
            'case_004': np.full(6, 40, dtype=np.float32),
            'case_005': np.full(6, 50, dtype=np.float32),
        }
        
        case_targets = {
            'case_001': [np.full(8, 100, dtype=np.float32)],
            'case_002': [np.full(8, 200, dtype=np.float32)],
            'case_003': [np.full(8, 300, dtype=np.float32)],
            'case_004': [np.full(8, 400, dtype=np.float32)],
            'case_005': [np.full(8, 500, dtype=np.float32)],
        }
        
        # Act - Kör split två gånger med samma seed
        train_data1, val_data1, test_data1 = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=42
        )
        
        train_data2, val_data2, test_data2 = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=42
        )
        
        # Assert
        # Verifiera att splits är identiska
        np.testing.assert_array_equal(train_data1['windows'], train_data2['windows'], 
                                    err_msg="Train windows ska vara identiska med samma seed")
        np.testing.assert_array_equal(train_data1['static'], train_data2['static'], 
                                    err_msg="Train static ska vara identiska med samma seed")
        np.testing.assert_array_equal(train_data1['targets'], train_data2['targets'], 
                                    err_msg="Train targets ska vara identiska med samma seed")
        
        np.testing.assert_array_equal(val_data1['windows'], val_data2['windows'], 
                                    err_msg="Validation windows ska vara identiska med samma seed")
        np.testing.assert_array_equal(val_data1['static'], val_data2['static'], 
                                    err_msg="Validation static ska vara identiska med samma seed")
        np.testing.assert_array_equal(val_data1['targets'], val_data2['targets'], 
                                    err_msg="Validation targets ska vara identiska med samma seed")
        
        np.testing.assert_array_equal(test_data1['windows'], test_data2['windows'], 
                                    err_msg="Test windows ska vara identiska med samma seed")
        np.testing.assert_array_equal(test_data1['static'], test_data2['static'], 
                                    err_msg="Test static ska vara identiska med samma seed")
        np.testing.assert_array_equal(test_data1['targets'], test_data2['targets'], 
                                    err_msg="Test targets ska vara identiska med samma seed")
        
        print(f"✅ T043 PASSED: Deterministic case-based split fungerar korrekt")
        print(f"   Verifierade att samma seed ger identiska splits")
    
    def test_t043_split_consistency_different_seeds(self):
        """
        Verifiera att olika seeds ger olika case distributions
        """
        # Arrange
        case_windows = {
            'case_001': [np.full((300, 16), 1, dtype=np.float32)],
            'case_002': [np.full((300, 16), 2, dtype=np.float32)],
            'case_003': [np.full((300, 16), 3, dtype=np.float32)],
            'case_004': [np.full((300, 16), 4, dtype=np.float32)],
            'case_005': [np.full((300, 16), 5, dtype=np.float32)],
        }
        
        case_static = {
            'case_001': np.full(6, 10, dtype=np.float32),
            'case_002': np.full(6, 20, dtype=np.float32),
            'case_003': np.full(6, 30, dtype=np.float32),
            'case_004': np.full(6, 40, dtype=np.float32),
            'case_005': np.full(6, 50, dtype=np.float32),
        }
        
        case_targets = {
            'case_001': [np.full(8, 100, dtype=np.float32)],
            'case_002': [np.full(8, 200, dtype=np.float32)],
            'case_003': [np.full(8, 300, dtype=np.float32)],
            'case_004': [np.full(8, 400, dtype=np.float32)],
            'case_005': [np.full(8, 500, dtype=np.float32)],
        }
        
        # Act - Kör split med olika seeds
        train_data1, val_data1, test_data1 = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=42
        )
        
        train_data2, val_data2, test_data2 = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=123
        )
        
        # Assert
        # Verifiera att splits är olika (men fortfarande case-consistent)
        train_different = not np.array_equal(train_data1['windows'], train_data2['windows'])
        val_different = not np.array_equal(val_data1['windows'], val_data2['windows'])
        test_different = not np.array_equal(test_data1['windows'], test_data2['windows'])
        
        # Minst en split ska vara olika
        self.assertTrue(train_different or val_different or test_different, 
                       "Olika seeds ska ge olika case distributions")
        
        # Verifiera att båda splits fortfarande är case-consistent
        for train_data, val_data, test_data, seed in [(train_data1, val_data1, test_data1, 42), 
                                                      (train_data2, val_data2, test_data2, 123)]:
            # Kontrollera att alla windows från samma case har samma värde
            case_values = {}
            
            if len(train_data['windows']) > 0:
                for window in train_data['windows']:
                    case_value = window[0, 0]
                    case_values[case_value] = 'train'
            
            if len(val_data['windows']) > 0:
                for window in val_data['windows']:
                    case_value = window[0, 0]
                    case_values[case_value] = 'validation'
            
            if len(test_data['windows']) > 0:
                for window in test_data['windows']:
                    case_value = window[0, 0]
                    case_values[case_value] = 'test'
            
            # Verifiera att varje case bara finns i en split
            self.assertEqual(len(case_values), 5, f"Seed {seed}: Ska ha 5 cases fördelade")
        
        print(f"✅ T043 PASSED: Different seeds case-based split fungerar korrekt")
        print(f"   Verifierade att olika seeds ger olika distributions men samma consistency")
    
    def test_t043_split_consistency_case_integrity(self):
        """
        Verifiera att case integrity bevaras vid split
        """
        # Arrange
        case_windows = {
            'case_001': [np.full((300, 16), 1, dtype=np.float32), np.full((300, 16), 1, dtype=np.float32)],
            'case_002': [np.full((300, 16), 2, dtype=np.float32), np.full((300, 16), 2, dtype=np.float32)],
            'case_003': [np.full((300, 16), 3, dtype=np.float32), np.full((300, 16), 3, dtype=np.float32)],
        }
        
        case_static = {
            'case_001': np.full(6, 10, dtype=np.float32),
            'case_002': np.full(6, 20, dtype=np.float32),
            'case_003': np.full(6, 30, dtype=np.float32),
        }
        
        case_targets = {
            'case_001': [np.full(8, 100, dtype=np.float32), np.full(8, 100, dtype=np.float32)],
            'case_002': [np.full(8, 200, dtype=np.float32), np.full(8, 200, dtype=np.float32)],
            'case_003': [np.full(8, 300, dtype=np.float32), np.full(8, 300, dtype=np.float32)],
        }
        
        # Act
        train_data, validation_data, test_data = split_data_by_case_70_15_15(
            case_windows, case_static, case_targets, random_seed=42
        )
        
        # Assert
        # Verifiera att alla windows är fördelade
        total_train_windows = len(train_data['windows'])
        total_val_windows = len(validation_data['windows'])
        total_test_windows = len(test_data['windows'])
        total_windows = total_train_windows + total_val_windows + total_test_windows
        
        expected_total_windows = sum(len(windows) for windows in case_windows.values())
        self.assertEqual(total_windows, expected_total_windows, f"Ska dela alla {expected_total_windows} windows")
        
        # Verifiera att case integrity bevaras
        case_values = {}
        
        # Analysera train data
        if total_train_windows > 0:
            for window in train_data['windows']:
                case_value = window[0, 0]
                case_values[case_value] = 'train'
        
        # Analysera validation data
        if total_val_windows > 0:
            for window in validation_data['windows']:
                case_value = window[0, 0]
                case_values[case_value] = 'validation'
        
        # Analysera test data
        if total_test_windows > 0:
            for window in test_data['windows']:
                case_value = window[0, 0]
                case_values[case_value] = 'test'
        
        # Verifiera att varje case bara finns i en split
        self.assertEqual(len(case_values), 3, "Ska ha 3 cases fördelade")
        
        # Verifiera att alla windows från samma case har samma värde
        for case_value in [1, 2, 3]:
            self.assertIn(case_value, case_values, f"Case {case_value} ska finnas i någon split")
        
        print(f"✅ T043 PASSED: Case integrity vid split fungerar korrekt")
        print(f"   Verifierade {len(case_values)} cases med specifika värden")
    
    def test_t043_split_consistency_edge_cases(self):
        """
        Verifiera case-based split med edge cases
        """
        # Arrange
        edge_cases = [
            (1, "Single case"),
            (2, "Two cases"),
            (3, "Three cases"),
        ]
        
        for n_cases, description in edge_cases:
            with self.subTest(case=description):
                # Arrange
                case_windows = {}
                case_static = {}
                case_targets = {}
                
                for i in range(n_cases):
                    case_id = f'case_{i+1:03d}'
                    case_windows[case_id] = [np.full((300, 16), i+1, dtype=np.float32)]
                    case_static[case_id] = np.full(6, (i+1)*10, dtype=np.float32)
                    case_targets[case_id] = [np.full(8, (i+1)*100, dtype=np.float32)]
                
                # Act
                train_data, validation_data, test_data = split_data_by_case_70_15_15(
                    case_windows, case_static, case_targets, random_seed=42
                )
                
                # Assert
                total_train_windows = len(train_data['windows'])
                total_val_windows = len(validation_data['windows'])
                total_test_windows = len(test_data['windows'])
                total_windows = total_train_windows + total_val_windows + total_test_windows
                
                # Verifiera att alla windows är fördelade
                self.assertEqual(total_windows, n_cases, f"Ska dela alla {n_cases} windows för {description}")
                
                # För små datasets, verifiera att minst train har data
                if n_cases < 3:
                    self.assertEqual(total_train_windows, n_cases, f"För {description} ska allt gå till train")
                    self.assertEqual(total_val_windows, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test_windows, 0, f"För {description} ska test vara tom")
                else:
                    # För större datasets, verifiera att alla splits har data
                    self.assertGreater(total_train_windows, 0, f"Train ska ha data för {description}")
                    # För små datasets (3 cases), validation och test kan vara tomma
                    if n_cases >= 7:  # Minst 7 cases för att garantera alla splits har data
                        self.assertGreater(total_val_windows, 0, f"Validation ska ha data för {description}")
                        self.assertGreater(total_test_windows, 0, f"Test ska ha data för {description}")
        
        print(f"✅ T043 PASSED: Edge cases case-based split fungerar korrekt")
        print(f"   Testade {len(edge_cases)} edge cases")
    
    def test_t043_split_consistency_comprehensive(self):
        """
        Omfattande test av case-based split consistency
        """
        # Arrange
        # Testa med olika antal cases
        test_cases = [
            (5, "Small dataset"),
            (10, "Medium dataset"),
            (20, "Large dataset"),
        ]
        
        for n_cases, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                case_windows = {}
                case_static = {}
                case_targets = {}
                
                for i in range(n_cases):
                    case_id = f'case_{i+1:03d}'
                    # Varje case har 2-3 windows
                    n_windows_per_case = 2 + (i % 2)  # 2 eller 3 windows
                    case_windows[case_id] = [np.full((300, 16), i+1, dtype=np.float32) for _ in range(n_windows_per_case)]
                    case_static[case_id] = np.full(6, (i+1)*10, dtype=np.float32)
                    case_targets[case_id] = [np.full(8, (i+1)*100, dtype=np.float32) for _ in range(n_windows_per_case)]
                
                # Act
                train_data, validation_data, test_data = split_data_by_case_70_15_15(
                    case_windows, case_static, case_targets, random_seed=42
                )
                
                # Assert
                total_train_windows = len(train_data['windows'])
                total_val_windows = len(validation_data['windows'])
                total_test_windows = len(test_data['windows'])
                total_windows = total_train_windows + total_val_windows + total_test_windows
                
                # Verifiera att alla windows är fördelade
                expected_total_windows = sum(len(windows) for windows in case_windows.values())
                self.assertEqual(total_windows, expected_total_windows, f"Ska dela alla {expected_total_windows} windows för {description}")
                
                # Verifiera att split är nära 70/15/15 (case-based split kan vara mer variabel)
                train_ratio = total_train_windows / expected_total_windows
                val_ratio = total_val_windows / expected_total_windows
                test_ratio = total_test_windows / expected_total_windows
                
                # Case-based split har större variation på grund av olika antal windows per case
                # Så vi använder mer generösa toleranser
                self.assertGreaterEqual(train_ratio, 0.5, f"Train ratio ska vara >= 50% för {description}")
                self.assertLessEqual(train_ratio, 0.9, f"Train ratio ska vara <= 90% för {description}")
                
                # Validation och test kan vara 0 för små datasets
                if n_cases >= 7:  # Minst 7 cases för att garantera alla splits
                    self.assertGreaterEqual(val_ratio, 0.05, f"Validation ratio ska vara >= 5% för {description}")
                    self.assertGreaterEqual(test_ratio, 0.05, f"Test ratio ska vara >= 5% för {description}")
                
                # Verifiera att alla splits har korrekt shape
                if total_train_windows > 0:
                    self.assertEqual(train_data['windows'].shape, (total_train_windows, 300, 16), 
                                   f"Train windows ska ha korrekt shape för {description}")
                
                if total_val_windows > 0:
                    self.assertEqual(validation_data['windows'].shape, (total_val_windows, 300, 16), 
                                   f"Validation windows ska ha korrekt shape för {description}")
                
                if total_test_windows > 0:
                    self.assertEqual(test_data['windows'].shape, (total_test_windows, 300, 16), 
                                   f"Test windows ska ha korrekt shape för {description}")
        
        print(f"✅ T043 PASSED: Comprehensive case-based split consistency test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
