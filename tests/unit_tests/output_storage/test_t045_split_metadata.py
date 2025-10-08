#!/usr/bin/env python3
"""
T045: Test Split Metadata
Verifiera att split-statistik sparas i metadata
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf
import shutil
import json

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
save_split_metadata = master_poc_tfrecord_creator.save_split_metadata

class TestT045SplitMetadata(unittest.TestCase):
    """T045: Test Split Metadata"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t045_split_metadata_basic(self):
        """
        T045: Test Split Metadata
        Verifiera att split-statistik sparas i metadata
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
        
        metadata = create_split_metadata(
            train_data, validation_data, test_data, n_windows, random_seed=42
        )
        
        # Assert
        # Verifiera att metadata innehåller alla nödvändiga keys
        self.assertIn('split_info', metadata, "Metadata ska innehålla 'split_info'")
        self.assertIn('split_config', metadata, "Metadata ska innehålla 'split_config'")
        self.assertIn('data_shapes', metadata, "Metadata ska innehålla 'data_shapes'")
        self.assertIn('created_at', metadata, "Metadata ska innehålla 'created_at'")
        self.assertIn('version', metadata, "Metadata ska innehålla 'version'")
        
        # Verifiera split_info
        split_info = metadata['split_info']
        self.assertIn('total_windows', split_info, "split_info ska innehålla 'total_windows'")
        self.assertIn('train_windows', split_info, "split_info ska innehålla 'train_windows'")
        self.assertIn('validation_windows', split_info, "split_info ska innehålla 'validation_windows'")
        self.assertIn('test_windows', split_info, "split_info ska innehålla 'test_windows'")
        self.assertIn('train_percentage', split_info, "split_info ska innehålla 'train_percentage'")
        self.assertIn('validation_percentage', split_info, "split_info ska innehålla 'validation_percentage'")
        self.assertIn('test_percentage', split_info, "split_info ska innehålla 'test_percentage'")
        
        # Verifiera att värden är korrekta
        self.assertEqual(split_info['total_windows'], n_windows, f"Total windows ska vara {n_windows}")
        self.assertEqual(split_info['train_windows'], len(train_data['windows']), "Train windows ska matcha")
        self.assertEqual(split_info['validation_windows'], len(validation_data['windows']), "Validation windows ska matcha")
        self.assertEqual(split_info['test_windows'], len(test_data['windows']), "Test windows ska matcha")
        
        # Verifiera att procenten är korrekt
        expected_train_pct = len(train_data['windows']) / n_windows * 100
        expected_val_pct = len(validation_data['windows']) / n_windows * 100
        expected_test_pct = len(test_data['windows']) / n_windows * 100
        
        self.assertAlmostEqual(split_info['train_percentage'], expected_train_pct, places=1, 
                             msg="Train percentage ska vara korrekt")
        self.assertAlmostEqual(split_info['validation_percentage'], expected_val_pct, places=1, 
                             msg="Validation percentage ska vara korrekt")
        self.assertAlmostEqual(split_info['test_percentage'], expected_test_pct, places=1, 
                             msg="Test percentage ska vara korrekt")
        
        print(f"✅ T045 PASSED: Basic split metadata creation fungerar korrekt")
        print(f"   Total windows: {split_info['total_windows']}")
        print(f"   Train: {split_info['train_windows']} ({split_info['train_percentage']:.1f}%)")
        print(f"   Validation: {split_info['validation_windows']} ({split_info['validation_percentage']:.1f}%)")
        print(f"   Test: {split_info['test_windows']} ({split_info['test_percentage']:.1f}%)")
    
    def test_t045_split_metadata_config(self):
        """
        Verifiera att split config information är korrekt
        """
        # Arrange
        n_windows = 50
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        random_seed = 123
        
        # Act
        train_data, validation_data, test_data = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=random_seed
        )
        
        metadata = create_split_metadata(
            train_data, validation_data, test_data, n_windows, random_seed=random_seed
        )
        
        # Assert
        # Verifiera split_config
        split_config = metadata['split_config']
        self.assertIn('random_seed', split_config, "split_config ska innehålla 'random_seed'")
        self.assertIn('split_ratio', split_config, "split_config ska innehålla 'split_ratio'")
        self.assertIn('split_method', split_config, "split_config ska innehålla 'split_method'")
        
        # Verifiera att värden är korrekta
        self.assertEqual(split_config['random_seed'], random_seed, f"Random seed ska vara {random_seed}")
        self.assertEqual(split_config['split_ratio'], '70/15/15', "Split ratio ska vara '70/15/15'")
        self.assertEqual(split_config['split_method'], 'window_based', "Split method ska vara 'window_based'")
        
        print(f"✅ T045 PASSED: Split config metadata fungerar korrekt")
        print(f"   Random seed: {split_config['random_seed']}")
        print(f"   Split ratio: {split_config['split_ratio']}")
        print(f"   Split method: {split_config['split_method']}")
    
    def test_t045_split_metadata_data_shapes(self):
        """
        Verifiera att data shapes information är korrekt
        """
        # Arrange
        n_windows = 30
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
        # Verifiera data_shapes
        data_shapes = metadata['data_shapes']
        self.assertIn('timeseries_shape', data_shapes, "data_shapes ska innehålla 'timeseries_shape'")
        self.assertIn('static_shape', data_shapes, "data_shapes ska innehålla 'static_shape'")
        self.assertIn('targets_shape', data_shapes, "data_shapes ska innehålla 'targets_shape'")
        
        # Verifiera att shapes är korrekta
        self.assertEqual(data_shapes['timeseries_shape'], [300, 16], "Timeseries shape ska vara [300, 16]")
        self.assertEqual(data_shapes['static_shape'], [6], "Static shape ska vara [6]")
        self.assertEqual(data_shapes['targets_shape'], [8], "Targets shape ska vara [8]")
        
        print(f"✅ T045 PASSED: Data shapes metadata fungerar korrekt")
        print(f"   Timeseries shape: {data_shapes['timeseries_shape']}")
        print(f"   Static shape: {data_shapes['static_shape']}")
        print(f"   Targets shape: {data_shapes['targets_shape']}")
    
    def test_t045_split_metadata_timestamp_and_version(self):
        """
        Verifiera att timestamp och version information är korrekt
        """
        # Arrange
        n_windows = 20
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
        # Verifiera created_at
        self.assertIn('created_at', metadata, "Metadata ska innehålla 'created_at'")
        created_at = metadata['created_at']
        self.assertIsInstance(created_at, str, "created_at ska vara en sträng")
        
        # Verifiera att timestamp är i ISO format
        try:
            pd.Timestamp(created_at)
        except ValueError:
            self.fail("created_at ska vara i ISO format")
        
        # Verifiera version
        self.assertIn('version', metadata, "Metadata ska innehålla 'version'")
        self.assertEqual(metadata['version'], '1.0', "Version ska vara '1.0'")
        
        print(f"✅ T045 PASSED: Timestamp och version metadata fungerar korrekt")
        print(f"   Created at: {created_at}")
        print(f"   Version: {metadata['version']}")
    
    def test_t045_split_metadata_save_to_file(self):
        """
        Verifiera att split metadata kan sparas till fil
        """
        # Arrange
        n_windows = 40
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_metadata")
        
        # Act
        train_data, validation_data, test_data = split_data_70_15_15(
            windows_data, static_data, targets_data, random_seed=42
        )
        
        metadata = create_split_metadata(
            train_data, validation_data, test_data, n_windows, random_seed=42
        )
        
        metadata_path = save_split_metadata(metadata, output_path)
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(metadata_path, "Metadata fil path ska vara definierad")
        self.assertTrue(os.path.exists(metadata_path), "Metadata fil ska existera på disk")
        
        # Verifiera att fil har korrekt namn
        expected_filename = "test_metadata_split_metadata.json"
        self.assertEqual(os.path.basename(metadata_path), expected_filename, 
                       f"Metadata fil ska ha namn {expected_filename}")
        
        # Verifiera att fil har storlek > 0
        self.assertGreater(os.path.getsize(metadata_path), 0, "Metadata fil ska ha storlek > 0")
        
        # Verifiera att fil kan läsas som JSON
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att innehållet matchar
        self.assertEqual(loaded_metadata['split_info']['total_windows'], n_windows, 
                       "Laddad metadata ska matcha original")
        self.assertEqual(loaded_metadata['split_config']['random_seed'], 42, 
                       "Laddad metadata ska matcha original")
        self.assertEqual(loaded_metadata['version'], '1.0', 
                       "Laddad metadata ska matcha original")
        
        print(f"✅ T045 PASSED: Split metadata save to file fungerar korrekt")
        print(f"   Metadata fil: {os.path.basename(metadata_path)}")
        print(f"   Fil storlek: {os.path.getsize(metadata_path)} bytes")
    
    def test_t045_split_metadata_edge_cases(self):
        """
        Verifiera split metadata med edge cases
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
                
                # Verifiera att total windows matchar
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
        
        print(f"✅ T045 PASSED: Edge cases split metadata fungerar korrekt")
        print(f"   Testade {len(edge_cases)} edge cases")
    
    def test_t045_split_metadata_comprehensive(self):
        """
        Omfattande test av split metadata
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
                
                random_seed = 42 + n_windows  # Olika seed för varje test
                
                # Act
                train_data, validation_data, test_data = split_data_70_15_15(
                    windows_data, static_data, targets_data, random_seed=random_seed
                )
                
                metadata = create_split_metadata(
                    train_data, validation_data, test_data, n_windows, random_seed=random_seed
                )
                
                # Assert
                # Verifiera att metadata innehåller alla nödvändiga keys
                self.assertIn('split_info', metadata, f"Metadata ska innehålla 'split_info' för {description}")
                self.assertIn('split_config', metadata, f"Metadata ska innehålla 'split_config' för {description}")
                self.assertIn('data_shapes', metadata, f"Metadata ska innehålla 'data_shapes' för {description}")
                self.assertIn('created_at', metadata, f"Metadata ska innehålla 'created_at' för {description}")
                self.assertIn('version', metadata, f"Metadata ska innehålla 'version' för {description}")
                
                # Verifiera att split config är korrekt
                split_config = metadata['split_config']
                self.assertEqual(split_config['random_seed'], random_seed, 
                               f"Random seed ska vara {random_seed} för {description}")
                self.assertEqual(split_config['split_ratio'], '70/15/15', 
                               f"Split ratio ska vara '70/15/15' för {description}")
                self.assertEqual(split_config['split_method'], 'window_based', 
                               f"Split method ska vara 'window_based' för {description}")
                
                # Verifiera att data shapes är korrekta
                data_shapes = metadata['data_shapes']
                self.assertEqual(data_shapes['timeseries_shape'], [300, 16], 
                               f"Timeseries shape ska vara [300, 16] för {description}")
                self.assertEqual(data_shapes['static_shape'], [6], 
                               f"Static shape ska vara [6] för {description}")
                self.assertEqual(data_shapes['targets_shape'], [8], 
                               f"Targets shape ska vara [8] för {description}")
                
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
        
        print(f"✅ T045 PASSED: Comprehensive split metadata test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
