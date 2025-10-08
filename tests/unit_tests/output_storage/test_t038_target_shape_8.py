#!/usr/bin/env python3
"""
T038: Test Target Shape [8]
Verifiera att targets har shape [8] med 3 drugs + 5 ventilator
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf

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

class TestT038TargetShape8(unittest.TestCase):
    """T038: Test Target Shape [8]"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t038_target_shape_8_basic(self):
        """
        T038: Test Target Shape [8]
        Verifiera att targets har shape [8] med 3 drugs + 5 ventilator
        """
        # Arrange
        # Skapa test data med korrekt target shape [8]
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)  # Shape [8]
        
        output_path = os.path.join(self.temp_dir, "test_target_shape_8")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att targets har korrekt shape [8]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera target shape för varje example
        for i, example in enumerate(parsed_data):
            target_shape = example['targets'].shape
            self.assertEqual(target_shape, (8,), 
                           f"Example {i} target shape ska vara (8,), fick {target_shape}")
        
        print(f"✅ T038 PASSED: Basic target shape [8] fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med target shape [8]")
    
    def test_t038_target_shape_8_drugs_ventilator_structure(self):
        """
        Verifiera att targets [8] innehåller 3 drugs + 5 ventilator enligt Master POC
        """
        # Arrange
        # Skapa test data med specifika target värden för att testa struktur
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        
        # Skapa targets med specifik struktur: [drug1, drug2, drug3, vent1, vent2, vent3, vent4, vent5]
        targets_data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Example 1
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],  # Example 2
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Example 3
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],  # Example 4
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Example 5
        ]).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_target_structure")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att targets har korrekt shape [8]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera target struktur för varje example
        for i, example in enumerate(parsed_data):
            targets = example['targets']
            
            # Verifiera shape
            self.assertEqual(targets.shape, (8,), 
                           f"Example {i} target shape ska vara (8,), fick {targets.shape}")
            
            # Verifiera att värden matchar input
            expected_targets = targets_data[i]
            np.testing.assert_array_almost_equal(targets, expected_targets, decimal=5,
                                               err_msg=f"Example {i} targets ska matcha input")
            
            # Verifiera att targets innehåller 8 värden
            self.assertEqual(len(targets), 8, f"Example {i} ska ha 8 target värden")
        
        print(f"✅ T038 PASSED: Target structure [8] med 3 drugs + 5 ventilator fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med korrekt target struktur")
    
    def test_t038_target_shape_8_edge_cases(self):
        """
        Verifiera target shape [8] med edge cases
        """
        # Arrange
        # Testa med olika edge cases
        test_cases = [
            ("Zero targets", np.zeros((3, 8), dtype=np.float32)),
            ("Negative targets", np.full((3, 8), -1.0, dtype=np.float32)),
            ("Large targets", np.full((3, 8), 1000.0, dtype=np.float32)),
            ("Mixed targets", np.array([
                [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0],
                [10.0, -10.0, 5.0, -5.0, 1.0, -1.0, 0.0, 0.0],
                [100.0, -100.0, 50.0, -50.0, 25.0, -25.0, 12.5, -12.5]
            ], dtype=np.float32))
        ]
        
        for case_name, targets_data in test_cases:
            with self.subTest(case=case_name):
                # Arrange
                n_windows = targets_data.shape[0]
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_edge_case_{case_name.replace(' ', '_')}")
                
                # Act
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert
                # Verifiera att fil skapas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {case_name}")
                
                # Verifiera att targets har korrekt shape [8]
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {case_name}")
                
                # Kontrollera target shape för varje example
                for i, example in enumerate(parsed_data):
                    target_shape = example['targets'].shape
                    self.assertEqual(target_shape, (8,), 
                                   f"Example {i} target shape ska vara (8,) för {case_name}, fick {target_shape}")
                    
                    # Verifiera att värden matchar input
                    expected_targets = targets_data[i]
                    np.testing.assert_array_almost_equal(example['targets'], expected_targets, decimal=5,
                                                       err_msg=f"Example {i} targets ska matcha input för {case_name}")
        
        print(f"✅ T038 PASSED: Edge cases target shape [8] fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} edge cases")
    
    def test_t038_target_shape_8_validation(self):
        """
        Verifiera att target shape [8] valideras korrekt
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        
        # Testa med olika target shapes som ska faila
        invalid_target_shapes = [
            (n_windows, 7),   # För få targets
            (n_windows, 9),   # För många targets
            (n_windows, 4),   # För få targets
            (n_windows, 12),  # För många targets
        ]
        
        for invalid_shape in invalid_target_shapes:
            with self.subTest(shape=invalid_shape):
                # Arrange
                invalid_targets_data = np.random.randn(*invalid_shape).astype(np.float32)
                output_path = os.path.join(self.temp_dir, f"test_invalid_shape_{invalid_shape[1]}")
                
                # Act & Assert
                # Detta ska raise ValueError
                with self.assertRaises(ValueError, msg=f"Target shape {invalid_shape} ska raise ValueError"):
                    self.tfrecord_creator.create_memory_efficient_tfrecord(
                        windows_data, static_data, invalid_targets_data, output_path, "test"
                    )
        
        print(f"✅ T038 PASSED: Target shape [8] validation fungerar korrekt")
        print(f"   Verifierade {len(invalid_target_shapes)} invalid shapes")
    
    def test_t038_target_shape_8_large_dataset(self):
        """
        Verifiera target shape [8] med stora datasets
        """
        # Arrange
        n_windows = 1000  # Stort dataset
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)  # Shape [8]
        
        output_path = os.path.join(self.temp_dir, "test_large_dataset")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas för stort dataset")
        
        # Verifiera att targets har korrekt shape [8]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera target shape för varje example
        for i, example in enumerate(parsed_data):
            target_shape = example['targets'].shape
            self.assertEqual(target_shape, (8,), 
                           f"Example {i} target shape ska vara (8,), fick {target_shape}")
        
        print(f"✅ T038 PASSED: Large dataset target shape [8] fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med target shape [8]")
    
    def test_t038_target_shape_8_tfrecord_schema(self):
        """
        Verifiera att TFRecord schema stöder target shape [8]
        """
        # Arrange
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)  # Shape [8]
        
        output_path = os.path.join(self.temp_dir, "test_schema")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att schema valideras korrekt
        schema_valid = self.tfrecord_creator.validate_tfrecord_schema(tfrecord_path)
        self.assertTrue(schema_valid, "TFRecord schema ska vara giltigt")
        
        # Verifiera att targets har korrekt shape [8] i schema
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        for example in parsed_data:
            target_shape = example['targets'].shape
            self.assertEqual(target_shape, (8,), 
                           f"Target shape i schema ska vara (8,), fick {target_shape}")
        
        print(f"✅ T038 PASSED: TFRecord schema target shape [8] fungerar korrekt")
        print(f"   Schema validering: {schema_valid}")
    
    def test_t038_target_shape_8_comprehensive(self):
        """
        Omfattande test av target shape [8]
        """
        # Arrange
        # Testa med olika dataset storlekar
        test_cases = [
            (1, "Single example"),
            (10, "Small dataset"),
            (100, "Medium dataset"),
            (500, "Large dataset"),
        ]
        
        for n_windows, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)  # Shape [8]
                
                output_path = os.path.join(self.temp_dir, f"test_comprehensive_{n_windows}")
                
                # Act
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert
                # Verifiera att fil skapas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {description}")
                
                # Verifiera att targets har korrekt shape [8]
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {description}")
                
                # Kontrollera target shape för varje example
                for i, example in enumerate(parsed_data):
                    target_shape = example['targets'].shape
                    self.assertEqual(target_shape, (8,), 
                                   f"Example {i} target shape ska vara (8,) för {description}, fick {target_shape}")
        
        print(f"✅ T038 PASSED: Comprehensive target shape [8] test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
