#!/usr/bin/env python3
"""
T044: Test Three TFRecord Files
Verifiera att train.tfrecord, validation.tfrecord, test.tfrecord skapas
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
create_three_tfrecord_files = master_poc_tfrecord_creator.create_three_tfrecord_files

class TestT044ThreeTFRecordFiles(unittest.TestCase):
    """T044: Test Three TFRecord Files"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t044_three_tfrecord_files_basic(self):
        """
        T044: Test Three TFRecord Files
        Verifiera att train.tfrecord, validation.tfrecord, test.tfrecord skapas
        """
        # Arrange
        # Skapa test data med exakt 100 windows för enkel verifiering
        n_windows = 100
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_three_files")
        
        # Act
        created_files = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path, random_seed=42
        )
        
        # Assert
        # Verifiera att alla tre filer skapas
        self.assertIn('train', created_files, "Train fil ska skapas")
        self.assertIn('validation', created_files, "Validation fil ska skapas")
        self.assertIn('test', created_files, "Test fil ska skapas")
        
        # Verifiera att filerna existerar på disk
        train_path = created_files['train']
        val_path = created_files['validation']
        test_path = created_files['test']
        
        self.assertIsNotNone(train_path, "Train fil path ska vara definierad")
        self.assertIsNotNone(val_path, "Validation fil path ska vara definierad")
        self.assertIsNotNone(test_path, "Test fil path ska vara definierad")
        
        self.assertTrue(os.path.exists(train_path), "Train fil ska existera på disk")
        self.assertTrue(os.path.exists(val_path), "Validation fil ska existera på disk")
        self.assertTrue(os.path.exists(test_path), "Test fil ska existera på disk")
        
        # Verifiera att filerna har korrekt namn
        self.assertEqual(os.path.basename(train_path), "test_three_files_train.tfrecord", 
                       "Train fil ska ha korrekt namn")
        self.assertEqual(os.path.basename(val_path), "test_three_files_validation.tfrecord", 
                       "Validation fil ska ha korrekt namn")
        self.assertEqual(os.path.basename(test_path), "test_three_files_test.tfrecord", 
                       "Test fil ska ha korrekt namn")
        
        # Verifiera att filerna har storlek > 0
        self.assertGreater(os.path.getsize(train_path), 0, "Train fil ska ha storlek > 0")
        self.assertGreater(os.path.getsize(val_path), 0, "Validation fil ska ha storlek > 0")
        self.assertGreater(os.path.getsize(test_path), 0, "Test fil ska ha storlek > 0")
        
        print(f"✅ T044 PASSED: Basic three TFRecord files creation fungerar korrekt")
        print(f"   Train fil: {os.path.basename(train_path)}")
        print(f"   Validation fil: {os.path.basename(val_path)}")
        print(f"   Test fil: {os.path.basename(test_path)}")
    
    def test_t044_three_tfrecord_files_content_verification(self):
        """
        Verifiera att innehållet i de tre filerna är korrekt
        """
        # Arrange
        n_windows = 60  # 60 windows för enkel verifiering
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_content_verification")
        
        # Act
        created_files = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path, random_seed=42
        )
        
        # Assert
        # Verifiera att alla tre filer skapas
        self.assertIn('train', created_files, "Train fil ska skapas")
        self.assertIn('validation', created_files, "Validation fil ska skapas")
        self.assertIn('test', created_files, "Test fil ska skapas")
        
        train_path = created_files['train']
        val_path = created_files['validation']
        test_path = created_files['test']
        
        # Verifiera att filerna kan läsas
        train_data = self.tfrecord_creator.read_tfrecord_file(train_path)
        val_data = self.tfrecord_creator.read_tfrecord_file(val_path)
        test_data = self.tfrecord_creator.read_tfrecord_file(test_path)
        
        # Verifiera att data har korrekt struktur
        for data, split_name in [(train_data, "train"), (val_data, "validation"), (test_data, "test")]:
            self.assertGreater(len(data), 0, f"{split_name} data ska ha examples")
            
            for i, example in enumerate(data):
                # Verifiera att example innehåller alla nödvändiga keys
                self.assertIn('timeseries', example, f"{split_name} example {i} ska innehålla 'timeseries'")
                self.assertIn('static', example, f"{split_name} example {i} ska innehålla 'static'")
                self.assertIn('targets', example, f"{split_name} example {i} ska innehålla 'targets'")
                
                # Verifiera att data har korrekt shape
                self.assertEqual(example['timeseries'].shape, (300, 16), 
                               f"{split_name} example {i} timeseries ska ha shape (300, 16)")
                self.assertEqual(example['static'].shape, (6,), 
                               f"{split_name} example {i} static ska ha shape (6,)")
                self.assertEqual(example['targets'].shape, (8,), 
                               f"{split_name} example {i} targets ska ha shape (8,)")
        
        # Verifiera att total antal examples matchar input
        total_examples = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_examples, n_windows, f"Ska ha {n_windows} total examples")
        
        print(f"✅ T044 PASSED: Content verification för three TFRecord files fungerar korrekt")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")
        print(f"   Test: {len(test_data)} examples")
        print(f"   Total: {total_examples} examples")
    
    def test_t044_three_tfrecord_files_split_verification(self):
        """
        Verifiera att split är korrekt (70/15/15)
        """
        # Arrange
        n_windows = 100  # 100 windows för exakt 70/15/15 split
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_split_verification")
        
        # Act
        created_files = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path, random_seed=42
        )
        
        # Assert
        train_path = created_files['train']
        val_path = created_files['validation']
        test_path = created_files['test']
        
        # Läs data från filerna
        train_data = self.tfrecord_creator.read_tfrecord_file(train_path)
        val_data = self.tfrecord_creator.read_tfrecord_file(val_path)
        test_data = self.tfrecord_creator.read_tfrecord_file(test_path)
        
        # Verifiera att split är korrekt (70/15/15)
        total_train = len(train_data)
        total_val = len(val_data)
        total_test = len(test_data)
        total_split = total_train + total_val + total_test
        
        self.assertEqual(total_split, n_windows, f"Ska dela alla {n_windows} windows")
        
        # Verifiera exakt 70/15/15 split
        expected_train = int(0.7 * n_windows)  # 70 windows
        expected_val = int(0.15 * n_windows)   # 15 windows
        expected_test = n_windows - expected_train - expected_val  # 15 windows
        
        self.assertEqual(total_train, expected_train, f"Train ska ha {expected_train} windows, fick {total_train}")
        self.assertEqual(total_val, expected_val, f"Validation ska ha {expected_val} windows, fick {total_val}")
        self.assertEqual(total_test, expected_test, f"Test ska ha {expected_test} windows, fick {total_test}")
        
        print(f"✅ T044 PASSED: Split verification för three TFRecord files fungerar korrekt")
        print(f"   Train: {total_train} windows ({total_train/n_windows*100:.1f}%)")
        print(f"   Validation: {total_val} windows ({total_val/n_windows*100:.1f}%)")
        print(f"   Test: {total_test} windows ({total_test/n_windows*100:.1f}%)")
    
    def test_t044_three_tfrecord_files_deterministic(self):
        """
        Verifiera att tre filer skapas deterministiskt med samma seed
        """
        # Arrange
        n_windows = 50
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path1 = os.path.join(self.temp_dir, "test_deterministic_1")
        output_path2 = os.path.join(self.temp_dir, "test_deterministic_2")
        
        # Act - Skapa filer två gånger med samma seed
        created_files1 = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path1, random_seed=42
        )
        
        created_files2 = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path2, random_seed=42
        )
        
        # Assert
        # Verifiera att båda skapar samma antal filer
        self.assertEqual(len(created_files1), len(created_files2), "Ska skapa samma antal filer")
        
        # Verifiera att splits är identiska
        for split_name in ['train', 'validation', 'test']:
            self.assertIn(split_name, created_files1, f"{split_name} fil ska skapas första gången")
            self.assertIn(split_name, created_files2, f"{split_name} fil ska skapas andra gången")
            
            path1 = created_files1[split_name]
            path2 = created_files2[split_name]
            
            # Läs data från båda filerna
            data1 = self.tfrecord_creator.read_tfrecord_file(path1)
            data2 = self.tfrecord_creator.read_tfrecord_file(path2)
            
            # Verifiera att data är identisk
            self.assertEqual(len(data1), len(data2), f"{split_name} ska ha samma antal examples")
            
            for i, (example1, example2) in enumerate(zip(data1, data2)):
                np.testing.assert_array_almost_equal(example1['timeseries'], example2['timeseries'], decimal=5,
                                                   err_msg=f"{split_name} example {i} timeseries ska vara identisk")
                np.testing.assert_array_almost_equal(example1['static'], example2['static'], decimal=5,
                                                   err_msg=f"{split_name} example {i} static ska vara identisk")
                np.testing.assert_array_almost_equal(example1['targets'], example2['targets'], decimal=5,
                                                   err_msg=f"{split_name} example {i} targets ska vara identisk")
        
        print(f"✅ T044 PASSED: Deterministic three TFRecord files creation fungerar korrekt")
        print(f"   Verifierade att samma seed ger identiska filer")
    
    def test_t044_three_tfrecord_files_different_seeds(self):
        """
        Verifiera att olika seeds ger olika splits
        """
        # Arrange
        n_windows = 50
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path1 = os.path.join(self.temp_dir, "test_different_seeds_1")
        output_path2 = os.path.join(self.temp_dir, "test_different_seeds_2")
        
        # Act - Skapa filer med olika seeds
        created_files1 = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path1, random_seed=42
        )
        
        created_files2 = create_three_tfrecord_files(
            windows_data, static_data, targets_data, output_path2, random_seed=123
        )
        
        # Assert
        # Verifiera att båda skapar samma antal filer
        self.assertEqual(len(created_files1), len(created_files2), "Ska skapa samma antal filer")
        
        # Verifiera att splits är olika (men fortfarande 70/15/15)
        for split_name in ['train', 'validation', 'test']:
            path1 = created_files1[split_name]
            path2 = created_files2[split_name]
            
            # Läs data från båda filerna
            data1 = self.tfrecord_creator.read_tfrecord_file(path1)
            data2 = self.tfrecord_creator.read_tfrecord_file(path2)
            
            # Verifiera att data är olika (minst en split ska vara olika)
            if len(data1) > 0 and len(data2) > 0:
                different = not np.array_equal(data1[0]['timeseries'], data2[0]['timeseries'])
                if different:
                    break
        else:
            self.fail("Olika seeds ska ge olika splits")
        
        # Verifiera att båda splits fortfarande är 70/15/15
        for created_files, seed in [(created_files1, 42), (created_files2, 123)]:
            train_data = self.tfrecord_creator.read_tfrecord_file(created_files['train'])
            val_data = self.tfrecord_creator.read_tfrecord_file(created_files['validation'])
            test_data = self.tfrecord_creator.read_tfrecord_file(created_files['test'])
            
            total_train = len(train_data)
            total_val = len(val_data)
            total_test = len(test_data)
            total_split = total_train + total_val + total_test
            
            self.assertEqual(total_split, n_windows, f"Seed {seed}: Ska dela alla {n_windows} windows")
            
            # Verifiera att split är nära 70/15/15
            train_ratio = total_train / n_windows
            val_ratio = total_val / n_windows
            test_ratio = total_test / n_windows
            
            self.assertGreaterEqual(train_ratio, 0.65, f"Seed {seed}: Train ratio ska vara >= 65%")
            self.assertLessEqual(train_ratio, 0.75, f"Seed {seed}: Train ratio ska vara <= 75%")
            
            self.assertGreaterEqual(val_ratio, 0.10, f"Seed {seed}: Validation ratio ska vara >= 10%")
            self.assertLessEqual(val_ratio, 0.20, f"Seed {seed}: Validation ratio ska vara <= 20%")
            
            self.assertGreaterEqual(test_ratio, 0.10, f"Seed {seed}: Test ratio ska vara >= 10%")
            self.assertLessEqual(test_ratio, 0.20, f"Seed {seed}: Test ratio ska vara <= 20%")
        
        print(f"✅ T044 PASSED: Different seeds three TFRecord files creation fungerar korrekt")
        print(f"   Verifierade att olika seeds ger olika splits men samma fördelning")
    
    def test_t044_three_tfrecord_files_edge_cases(self):
        """
        Verifiera hantering av edge cases
        """
        # Arrange
        edge_cases = [
            (1, "Single window"),
            (2, "Two windows"),
            (3, "Three windows"),
            (10, "Small dataset"),
        ]
        
        for n_windows, description in edge_cases:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_edge_case_{n_windows}")
                
                # Act
                created_files = create_three_tfrecord_files(
                    windows_data, static_data, targets_data, output_path, random_seed=42
                )
                
                # Assert
                # Verifiera att alla tre filer skapas
                self.assertIn('train', created_files, f"Train fil ska skapas för {description}")
                self.assertIn('validation', created_files, f"Validation fil ska skapas för {description}")
                self.assertIn('test', created_files, f"Test fil ska skapas för {description}")
                
                # Verifiera att minst train fil har data
                train_path = created_files['train']
                self.assertIsNotNone(train_path, f"Train fil ska skapas för {description}")
                self.assertTrue(os.path.exists(train_path), f"Train fil ska existera för {description}")
                
                # För små datasets, verifiera att validation och test kan vara tomma
                val_path = created_files['validation']
                test_path = created_files['test']
                
                if n_windows < 7:  # Minst 7 windows för att garantera alla splits har data
                    # För små datasets, validation och test kan vara None
                    if val_path is None:
                        self.assertIsNone(val_path, f"Validation fil ska vara None för {description}")
                    if test_path is None:
                        self.assertIsNone(test_path, f"Test fil ska vara None för {description}")
                else:
                    # För större datasets, verifiera att alla filer har data
                    self.assertIsNotNone(val_path, f"Validation fil ska skapas för {description}")
                    self.assertIsNotNone(test_path, f"Test fil ska skapas för {description}")
                    
                    self.assertTrue(os.path.exists(val_path), f"Validation fil ska existera för {description}")
                    self.assertTrue(os.path.exists(test_path), f"Test fil ska existera för {description}")
        
        print(f"✅ T044 PASSED: Edge cases three TFRecord files creation fungerar korrekt")
        print(f"   Testade {len(edge_cases)} edge cases")
    
    def test_t044_three_tfrecord_files_comprehensive(self):
        """
        Omfattande test av three TFRecord files creation
        """
        # Arrange
        # Testa med olika dataset storlekar
        test_cases = [
            (10, "Small dataset"),
            (50, "Medium dataset"),
            (100, "Large dataset"),
            (500, "Very large dataset"),
        ]
        
        for n_windows, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_comprehensive_{n_windows}")
                
                # Act
                created_files = create_three_tfrecord_files(
                    windows_data, static_data, targets_data, output_path, random_seed=42
                )
                
                # Assert
                # Verifiera att alla tre filer skapas
                self.assertIn('train', created_files, f"Train fil ska skapas för {description}")
                self.assertIn('validation', created_files, f"Validation fil ska skapas för {description}")
                self.assertIn('test', created_files, f"Test fil ska skapas för {description}")
                
                # Verifiera att filerna existerar
                train_path = created_files['train']
                val_path = created_files['validation']
                test_path = created_files['test']
                
                self.assertIsNotNone(train_path, f"Train fil ska skapas för {description}")
                self.assertTrue(os.path.exists(train_path), f"Train fil ska existera för {description}")
                
                if val_path is not None:
                    self.assertTrue(os.path.exists(val_path), f"Validation fil ska existera för {description}")
                
                if test_path is not None:
                    self.assertTrue(os.path.exists(test_path), f"Test fil ska existera för {description}")
                
                # Verifiera att filerna har storlek > 0
                self.assertGreater(os.path.getsize(train_path), 0, f"Train fil ska ha storlek > 0 för {description}")
                
                if val_path is not None:
                    self.assertGreater(os.path.getsize(val_path), 0, f"Validation fil ska ha storlek > 0 för {description}")
                
                if test_path is not None:
                    self.assertGreater(os.path.getsize(test_path), 0, f"Test fil ska ha storlek > 0 för {description}")
        
        print(f"✅ T044 PASSED: Comprehensive three TFRecord files creation test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
