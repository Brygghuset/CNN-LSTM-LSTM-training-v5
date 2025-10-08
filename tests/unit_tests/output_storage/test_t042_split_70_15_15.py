#!/usr/bin/env python3
"""
T042: Test 70/15/15 Split
Verifiera korrekt 70/15/15 split av windows
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

class TestT042Split701515(unittest.TestCase):
    """T042: Test 70/15/15 Split"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t042_split_70_15_15_basic(self):
        """
        T042: Test 70/15/15 Split
        Verifiera korrekt 70/15/15 split av windows
        """
        # Arrange
        # Skapa test data med exakt 100 windows för enkel verifiering
        n_windows = 100
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
        
        self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows")
        
        # Verifiera 70/15/15 split (med tolerans för avrundning)
        expected_train = int(0.7 * n_windows)  # 70 windows
        expected_val = int(0.15 * n_windows)   # 15 windows
        expected_test = n_windows - expected_train - expected_val  # 15 windows
        
        self.assertEqual(total_train, expected_train, f"Train ska ha {expected_train} windows, fick {total_train}")
        self.assertEqual(total_val, expected_val, f"Validation ska ha {expected_val} windows, fick {total_val}")
        self.assertEqual(total_test, expected_test, f"Test ska ha {expected_test} windows, fick {total_test}")
        
        # Verifiera att alla splits har korrekt shape
        if total_train > 0:
            self.assertEqual(train_data['windows'].shape, (total_train, 300, 16), "Train windows ska ha korrekt shape")
            self.assertEqual(train_data['static'].shape, (total_train, 6), "Train static ska ha korrekt shape")
            self.assertEqual(train_data['targets'].shape, (total_train, 8), "Train targets ska ha korrekt shape")
        
        if total_val > 0:
            self.assertEqual(validation_data['windows'].shape, (total_val, 300, 16), "Validation windows ska ha korrekt shape")
            self.assertEqual(validation_data['static'].shape, (total_val, 6), "Validation static ska ha korrekt shape")
            self.assertEqual(validation_data['targets'].shape, (total_val, 8), "Validation targets ska ha korrekt shape")
        
        if total_test > 0:
            self.assertEqual(test_data['windows'].shape, (total_test, 300, 16), "Test windows ska ha korrekt shape")
            self.assertEqual(test_data['static'].shape, (total_test, 6), "Test static ska ha korrekt shape")
            self.assertEqual(test_data['targets'].shape, (total_test, 8), "Test targets ska ha korrekt shape")
        
        print(f"✅ T042 PASSED: Basic 70/15/15 split fungerar korrekt")
        print(f"   Train: {total_train} windows ({total_train/n_windows*100:.1f}%)")
        print(f"   Validation: {total_val} windows ({total_val/n_windows*100:.1f}%)")
        print(f"   Test: {total_test} windows ({total_test/n_windows*100:.1f}%)")
    
    def test_t042_split_70_15_15_different_sizes(self):
        """
        Verifiera 70/15/15 split med olika dataset storlekar
        """
        # Arrange
        test_cases = [
            (10, "Small dataset"),
            (50, "Medium dataset"),
            (100, "Large dataset"),
            (1000, "Very large dataset"),
        ]
        
        for n_windows, description in test_cases:
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
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                # Verifiera att alla windows är fördelade
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # Verifiera att split är nära 70/15/15 (med tolerans för avrundning)
                train_ratio = total_train / n_windows
                val_ratio = total_val / n_windows
                test_ratio = total_test / n_windows
                
                # Tolerans för avrundning: ±5%
                self.assertGreaterEqual(train_ratio, 0.65, f"Train ratio ska vara >= 65% för {description}, fick {train_ratio:.3f}")
                self.assertLessEqual(train_ratio, 0.75, f"Train ratio ska vara <= 75% för {description}, fick {train_ratio:.3f}")
                
                self.assertGreaterEqual(val_ratio, 0.10, f"Validation ratio ska vara >= 10% för {description}, fick {val_ratio:.3f}")
                self.assertLessEqual(val_ratio, 0.20, f"Validation ratio ska vara <= 20% för {description}, fick {val_ratio:.3f}")
                
                self.assertGreaterEqual(test_ratio, 0.10, f"Test ratio ska vara >= 10% för {description}, fick {test_ratio:.3f}")
                self.assertLessEqual(test_ratio, 0.20, f"Test ratio ska vara <= 20% för {description}, fick {test_ratio:.3f}")
        
        print(f"✅ T042 PASSED: Different sizes 70/15/15 split fungerar korrekt")
        print(f"   Testade {len(test_cases)} olika dataset storlekar")
    
    def test_t042_split_70_15_15_deterministic(self):
        """
        Verifiera att split är deterministisk med samma seed
        """
        # Arrange
        n_windows = 100
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
        
        print(f"✅ T042 PASSED: Deterministic 70/15/15 split fungerar korrekt")
        print(f"   Verifierade att samma seed ger identiska splits")
    
    def test_t042_split_70_15_15_different_seeds(self):
        """
        Verifiera att olika seeds ger olika splits
        """
        # Arrange
        n_windows = 100
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Act - Kör split med olika seeds
        train_data1, val_data1, test_data1 = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=42
        )
        
        train_data2, val_data2, test_data2 = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=123
        )
        
        # Assert
        # Verifiera att splits är olika (men fortfarande 70/15/15)
        train_different = not np.array_equal(train_data1['windows'], train_data2['windows'])
        val_different = not np.array_equal(val_data1['windows'], val_data2['windows'])
        test_different = not np.array_equal(test_data1['windows'], test_data2['windows'])
        
        # Minst en split ska vara olika
        self.assertTrue(train_different or val_different or test_different, 
                       "Olika seeds ska ge olika splits")
        
        # Verifiera att båda splits fortfarande är 70/15/15
        for train_data, val_data, test_data, seed in [(train_data1, val_data1, test_data1, 42), 
                                                      (train_data2, val_data2, test_data2, 123)]:
            total_train = len(train_data['windows'])
            total_val = len(val_data['windows'])
            total_test = len(test_data['windows'])
            
            expected_train = int(0.7 * n_windows)
            expected_val = int(0.15 * n_windows)
            expected_test = n_windows - expected_train - expected_val
            
            self.assertEqual(total_train, expected_train, f"Seed {seed}: Train ska ha {expected_train} windows")
            self.assertEqual(total_val, expected_val, f"Seed {seed}: Validation ska ha {expected_val} windows")
            self.assertEqual(total_test, expected_test, f"Seed {seed}: Test ska ha {expected_test} windows")
        
        print(f"✅ T042 PASSED: Different seeds 70/15/15 split fungerar korrekt")
        print(f"   Verifierade att olika seeds ger olika splits men samma fördelning")
    
    def test_t042_split_70_15_15_edge_cases(self):
        """
        Verifiera 70/15/15 split med edge cases
        """
        # Arrange
        edge_cases = [
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
            (4, "Four windows"),
        ]
        
        for n_windows, description in edge_cases:
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
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                # Verifiera att alla windows är fördelade
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # För små datasets, verifiera att minst train har data
                if n_windows < 3:
                    self.assertEqual(total_train, n_windows, f"För {description} ska allt gå till train")
                    self.assertEqual(total_val, 0, f"För {description} ska validation vara tom")
                    self.assertEqual(total_test, 0, f"För {description} ska test vara tom")
                else:
                    # För större datasets, verifiera att alla splits har data
                    self.assertGreater(total_train, 0, f"Train ska ha data för {description}")
                    # För små datasets (3-4 windows), validation och test kan vara tomma
                    if n_windows >= 7:  # Minst 7 windows för att garantera alla splits har data
                        self.assertGreater(total_val, 0, f"Validation ska ha data för {description}")
                        self.assertGreater(total_test, 0, f"Test ska ha data för {description}")
        
        print(f"✅ T042 PASSED: Edge cases 70/15/15 split fungerar korrekt")
        print(f"   Testade {len(edge_cases)} edge cases")
    
    def test_t042_split_70_15_15_data_integrity(self):
        """
        Verifiera att data integritet bevaras vid split
        """
        # Arrange
        n_windows = 50
        # Skapa specifika test data för att verifiera integritet
        windows_data = np.array([
            np.full((300, 16), i, dtype=np.float32) for i in range(n_windows)
        ])
        static_data = np.array([
            np.full(6, i * 10, dtype=np.float32) for i in range(n_windows)
        ])
        targets_data = np.array([
            np.full(8, i * 100, dtype=np.float32) for i in range(n_windows)
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
        
        # Verifiera att data integritet bevaras
        all_train_windows = train_data['windows']
        all_val_windows = validation_data['windows']
        all_test_windows = test_data['windows']
        
        # Kontrollera att varje window har korrekt värde (baserat på original data)
        for i, window in enumerate(all_train_windows):
            actual_value = window[0, 0]  # Första elementet i första tidssteg
            # Verifiera att värdet finns i original data
            self.assertIn(actual_value, range(n_windows), f"Train window {i} ska ha värde från original data")
        
        for i, window in enumerate(all_val_windows):
            actual_value = window[0, 0]  # Första elementet i första tidssteg
            # Verifiera att värdet finns i original data
            self.assertIn(actual_value, range(n_windows), f"Validation window {i} ska ha värde från original data")
        
        for i, window in enumerate(all_test_windows):
            actual_value = window[0, 0]  # Första elementet i första tidssteg
            # Verifiera att värdet finns i original data
            self.assertIn(actual_value, range(n_windows), f"Test window {i} ska ha värde från original data")
        
        print(f"✅ T042 PASSED: Data integrity vid 70/15/15 split fungerar korrekt")
        print(f"   Verifierade {n_windows} windows med specifika värden")
    
    def test_t042_split_70_15_15_comprehensive(self):
        """
        Omfattande test av 70/15/15 split
        """
        # Arrange
        # Testa med olika dataset storlekar och seeds
        test_cases = [
            (10, 42, "Small dataset, seed 42"),
            (50, 123, "Medium dataset, seed 123"),
            (100, 456, "Large dataset, seed 456"),
            (500, 789, "Very large dataset, seed 789"),
        ]
        
        for n_windows, seed, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=seed
                )
                
                # Assert
                total_train = len(train_data['windows'])
                total_val = len(validation_data['windows'])
                total_test = len(test_data['windows'])
                total_split = total_train + total_val + total_test
                
                # Verifiera att alla windows är fördelade
                self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows för {description}")
                
                # Verifiera att split är nära 70/15/15
                train_ratio = total_train / n_windows
                val_ratio = total_val / n_windows
                test_ratio = total_test / n_windows
                
                # Tolerans för avrundning: ±5%
                self.assertGreaterEqual(train_ratio, 0.65, f"Train ratio ska vara >= 65% för {description}")
                self.assertLessEqual(train_ratio, 0.75, f"Train ratio ska vara <= 75% för {description}")
                
                self.assertGreaterEqual(val_ratio, 0.10, f"Validation ratio ska vara >= 10% för {description}")
                self.assertLessEqual(val_ratio, 0.20, f"Validation ratio ska vara <= 20% för {description}")
                
                self.assertGreaterEqual(test_ratio, 0.10, f"Test ratio ska vara >= 10% för {description}")
                self.assertLessEqual(test_ratio, 0.20, f"Test ratio ska vara <= 20% för {description}")
                
                # Verifiera att alla splits har korrekt shape
                if total_train > 0:
                    self.assertEqual(train_data['windows'].shape, (total_train, 300, 16), 
                                   f"Train windows ska ha korrekt shape för {description}")
                
                if total_val > 0:
                    self.assertEqual(validation_data['windows'].shape, (total_val, 300, 16), 
                                   f"Validation windows ska ha korrekt shape för {description}")
                
                if total_test > 0:
                    self.assertEqual(test_data['windows'].shape, (total_test, 300, 16), 
                                   f"Test windows ska ha korrekt shape för {description}")
        
        print(f"✅ T042 PASSED: Comprehensive 70/15/15 split test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar och seeds")

if __name__ == '__main__':
    unittest.main()
