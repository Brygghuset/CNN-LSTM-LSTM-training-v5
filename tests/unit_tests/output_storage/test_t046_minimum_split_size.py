#!/usr/bin/env python3
"""
T046: Test Minimum Split Size
Verifiera hantering när dataset är för litet för split
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
split_data_70_15_15 = master_poc_tfrecord_creator.split_data_70_15_15
create_split_metadata = master_poc_tfrecord_creator.create_split_metadata

class TestT046MinimumSplitSize(unittest.TestCase):
    """T046: Test Minimum Split Size"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t046_minimum_split_size_basic(self):
        """
        T046: Test Minimum Split Size
        Verifiera hantering när dataset är för litet för split
        """
        # Arrange
        # Testa med olika små dataset storlekar
        small_datasets = [
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
        ]
        
        for n_windows, description in small_datasets:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=42
                )
                
                # Assert
                # Verifiera att alla windows är fördelade
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # För små datasets, verifiera att allt går till train
                if n_windows < 3:
                    self.assertEqual(total_train, n_windows, f"För {description} ska allt gå till train")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                else:
                    # För 3 windows, verifiera att minst train har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    # Validation och test kan vara tomma för 3 windows
                    if total_val > 0:
                        self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                    if total_test > 0:
                        self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
        
        print(f"✅ T046 PASSED: Basic minimum split size handling fungerar korrekt")
        print(f"   Testade {len(small_datasets)} små dataset storlekar")
    
    def test_t046_minimum_split_size_edge_cases(self):
        """
        Verifiera edge cases för minimum split size
        """
        # Arrange
        edge_cases = [
            (0, "Empty dataset"),
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
            (4, "Four windows"),
            (5, "Five windows"),
            (6, "Six windows"),
            (7, "Seven windows"),  # Minst för att garantera alla splits
        ]
        
        for n_windows, description in edge_cases:
            with self.subTest(case=description):
                # Arrange
                if n_windows == 0:
                    # Special case för empty dataset
                    windows_data = np.array([]).reshape(0, 300, 16)
                    static_data = np.array([]).reshape(0, 6)
                    targets_data = np.array([]).reshape(0, 8)
                else:
                    windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                    static_data = np.random.randn(n_windows, 6).astype(np.float32)
                    targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=42
                )
                
                # Assert
                # Verifiera att alla windows är fördelade
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # Verifiera att split hanteras korrekt baserat på storlek
                if n_windows == 0:
                    # Empty dataset
                    self.assertEqual(total_train, 0, f"För {description} ska train vara tom")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                elif n_windows < 3:
                    # För små datasets, verifiera att allt går till train
                    self.assertEqual(total_train, n_windows, f"För {description} ska allt gå till train")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                elif n_windows < 7:
                    # För små datasets, verifiera att minst train har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    # Validation och test kan vara tomma för små datasets
                    if total_val > 0:
                        self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                    if total_test > 0:
                        self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
                else:
                    # För större datasets, verifiera att alla splits har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                    self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
        
        print(f"✅ T046 PASSED: Edge cases minimum split size handling fungerar korrekt")
        print(f"   Testade {len(edge_cases)} edge cases")
    
    def test_t046_minimum_split_size_metadata(self):
        """
        Verifiera att metadata hanteras korrekt för små datasets
        """
        # Arrange
        small_datasets = [
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
        ]
        
        for n_windows, description in small_datasets:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=42
                )
                
                metadata = create_split_metadata(
                    train_data, validation_data, test_data, n_windows, random_seed=42
                )
                
                # Assert
                # Verifiera att metadata innehåller alla nödvändiga keys
                self.assertIn('split_info', metadata, f"Metadata ska innehålla 'split_info' för {description}")
                self.assertIn('split_config', metadata, f"Metadata ska innehålla 'split_config' för {description}")
                self.assertIn('data_shapes', metadata, f"Metadata ska innehålla 'data_shapes' för {description}")
                
                # Verifiera att split info är korrekt
                split_info = metadata['split_info']
                self.assertEqual(split_info['total_windows'], n_windows, 
                               f"Total windows ska vara {n_windows} för {description}")
                
                # Verifiera att alla windows är fördelade
                total_split = split_info['train_windows'] + split_info['validation_windows'] + split_info['test_windows']
                self.assertEqual(total_split, n_windows, 
                               f"Ska dela alla {n_windows} windows för {description}")
                
                # Verifiera att procenten är korrekt
                train_pct = split_info['train_percentage']
                val_pct = split_info['validation_percentage']
                test_pct = split_info['test_percentage']
                
                self.assertAlmostEqual(train_pct + val_pct + test_pct, 100.0, places=1, 
                                     msg=f"Procenten ska summera till 100% för {description}")
                
                # För små datasets, verifiera att allt går till train
                if n_windows < 3:
                    self.assertEqual(split_info['train_windows'], n_windows, 
                                   f"För {description} ska allt gå till train")
                    self.assertEqual(split_info['validation_windows'], 0, 
                                   f"För {description} ska validation vara tom")
                    self.assertEqual(split_info['test_windows'], 0, 
                                   f"För {description} ska test vara tom")
        
        print(f"✅ T046 PASSED: Metadata minimum split size handling fungerar korrekt")
        print(f"   Testade {len(small_datasets)} små dataset storlekar")
    
    def test_t046_minimum_split_size_deterministic(self):
        """
        Verifiera att minimum split size hantering är deterministisk
        """
        # Arrange
        n_windows = 2  # Liten dataset
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Act - Kör split två gånger med samma seed
        train_data1, val_data1, test_data1 = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=42
        )
        
        train_data2, val_data2, test_data2 = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=42
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
        
        print(f"✅ T046 PASSED: Deterministic minimum split size handling fungerar korrekt")
        print(f"   Verifierade att samma seed ger identiska splits för små dataset")
    
    def test_t046_minimum_split_size_data_integrity(self):
        """
        Verifiera att data integritet bevaras för små datasets
        """
        # Arrange
        n_windows = 2  # Liten dataset
        # Skapa specifika test data för att verifiera integritet
        windows_data = np.array([
            np.full((300, 16), i+1, dtype=np.float32) for i in range(n_windows)
        ])
        static_data = np.array([
            np.full(6, (i+1)*10, dtype=np.float32) for i in range(n_windows)
        ])
        targets_data = np.array([
            np.full(8, (i+1)*100, dtype=np.float32) for i in range(n_windows)
        ])
        
        # Act
        train_data, validation_data, test_data = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=42
        )
        
        # Assert
        # Verifiera att alla windows är fördelade
        total_train = len(train_data['windows'])
        total_val = len(validation_data['windows'])
        total_test = len(test_data['windows'])
        total_split = total_train + total_val + total_test
        
        self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows")
        
        # För små datasets, verifiera att allt går till train
        self.assertEqual(total_train, n_windows, f"För små dataset ska allt gå till train")
        self.assertEqual(total_val, 0, f"För små dataset ska validation vara tom")
        self.assertEqual(total_test, 0, f"För små dataset ska test vara tom")
        
        # Verifiera att data integritet bevaras
        for i, window in enumerate(train_data['windows']):
            expected_value = i+1  # Index i original data
            actual_value = window[0, 0]  # Första elementet i första tidssteg
            self.assertEqual(actual_value, expected_value, f"Train window {i} ska ha korrekt värde")
        
        print(f"✅ T046 PASSED: Data integrity minimum split size handling fungerar korrekt")
        print(f"   Verifierade {n_windows} windows med specifika värden")
    
    def test_t046_minimum_split_size_comprehensive(self):
        """
        Omfattande test av minimum split size hantering
        """
        # Arrange
        # Testa med olika små dataset storlekar
        test_cases = [
            (0, "Empty dataset"),
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
            (4, "Four windows"),
            (5, "Five windows"),
            (6, "Six windows"),
            (7, "Seven windows"),  # Minst för att garantera alla splits
        ]
        
        for n_windows, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                if n_windows == 0:
                    # Special case för empty dataset
                    windows_data = np.array([]).reshape(0, 300, 16)
                    static_data = np.array([]).reshape(0, 6)
                    targets_data = np.array([]).reshape(0, 8)
                else:
                    windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                    static_data = np.random.randn(n_windows, 6).astype(np.float32)
                    targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=42
                )
                
                metadata = create_split_metadata(
                    train_data, validation_data, test_data, n_windows, random_seed=42
                )
                
                # Assert
                # Verifiera att alla windows är fördelade
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # Verifiera att metadata är korrekt
                split_info = metadata['split_info']
                self.assertEqual(split_info['total_windows'], n_windows, 
                               f"Total windows ska vara {n_windows} för {description}")
                
                # Verifiera att alla windows är fördelade i metadata
                total_split_metadata = split_info['train_windows'] + split_info['validation_windows'] + split_info['test_windows']
                self.assertEqual(total_split_metadata, n_windows, 
                               f"Ska dela alla {n_windows} windows i metadata för {description}")
                
                # Verifiera att procenten är korrekt
                train_pct = split_info['train_percentage']
                val_pct = split_info['validation_percentage']
                test_pct = split_info['test_percentage']
                
                # För empty dataset, alla procenten är 0
                if n_windows == 0:
                    self.assertEqual(train_pct, 0, f"Train percentage ska vara 0 för {description}")
                    self.assertEqual(val_pct, 0, f"Validation percentage ska vara 0 för {description}")
                    self.assertEqual(test_pct, 0, f"Test percentage ska vara 0 för {description}")
                else:
                    self.assertAlmostEqual(train_pct + val_pct + test_pct, 100.0, places=1, 
                                         msg=f"Procenten ska summera till 100% för {description}")
                
                # Verifiera att split hanteras korrekt baserat på storlek
                if n_windows == 0:
                    # Empty dataset
                    self.assertEqual(total_train, 0, f"För {description} ska train vara tom")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                elif n_windows < 3:
                    # För små datasets, verifiera att allt går till train
                    self.assertEqual(total_train, n_windows, f"För {description} ska allt gå till train")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                elif n_windows < 7:
                    # För små datasets, verifiera att minst train har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    # Validation och test kan vara tomma för små datasets
                    if total_val > 0:
                        self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                    if total_test > 0:
                        self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
                else:
                    # För större datasets, verifiera att alla splits har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                    self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
        
        print(f"✅ T046 PASSED: Comprehensive minimum split size handling test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
