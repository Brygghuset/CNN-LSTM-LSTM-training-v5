#!/usr/bin/env python3
"""
T039: Test Static Shape [6]
Verifiera att static features har shape [6]
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

class TestT039StaticShape6(unittest.TestCase):
    """T039: Test Static Shape [6]"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t039_static_shape_6_basic(self):
        """
        T039: Test Static Shape [6]
        Verifiera att static features har shape [6]
        """
        # Arrange
        # Skapa test data med korrekt static shape [6]
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)  # Shape [6]
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_static_shape_6")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att static features har korrekt shape [6]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera static shape för varje example
        for i, example in enumerate(parsed_data):
            static_shape = example['static'].shape
            self.assertEqual(static_shape, (6,), 
                           f"Example {i} static shape ska vara (6,), fick {static_shape}")
        
        print(f"✅ T039 PASSED: Basic static shape [6] fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med static shape [6]")
    
    def test_t039_static_shape_6_patient_features_structure(self):
        """
        Verifiera att static [6] innehåller patient features enligt Master POC
        """
        # Arrange
        # Skapa test data med specifika static feature värden för att testa struktur
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        
        # Skapa static features med specifik struktur: [age, sex, height, weight, bmi, asa]
        static_data = np.array([
            [65.0, 1.0, 175.0, 80.0, 26.1, 2.0],  # Example 1: 65 år, man, 175cm, 80kg, BMI 26.1, ASA 2
            [45.0, 0.0, 160.0, 60.0, 23.4, 1.0],  # Example 2: 45 år, kvinna, 160cm, 60kg, BMI 23.4, ASA 1
            [30.0, 1.0, 180.0, 90.0, 27.8, 3.0],  # Example 3: 30 år, man, 180cm, 90kg, BMI 27.8, ASA 3
            [70.0, 0.0, 165.0, 70.0, 25.7, 2.0],  # Example 4: 70 år, kvinna, 165cm, 70kg, BMI 25.7, ASA 2
            [25.0, 1.0, 185.0, 85.0, 24.8, 1.0]   # Example 5: 25 år, man, 185cm, 85kg, BMI 24.8, ASA 1
        ]).astype(np.float32)
        
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_static_structure")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att static features har korrekt shape [6]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera static struktur för varje example
        for i, example in enumerate(parsed_data):
            static_features = example['static']
            
            # Verifiera shape
            self.assertEqual(static_features.shape, (6,), 
                           f"Example {i} static shape ska vara (6,), fick {static_features.shape}")
            
            # Verifiera att värden matchar input
            expected_static = static_data[i]
            np.testing.assert_array_almost_equal(static_features, expected_static, decimal=5,
                                               err_msg=f"Example {i} static features ska matcha input")
            
            # Verifiera att static features innehåller 6 värden
            self.assertEqual(len(static_features), 6, f"Example {i} ska ha 6 static feature värden")
            
            # Verifiera att värden är rimliga för patient features
            age, sex, height, weight, bmi, asa = static_features
            
            # Age ska vara positivt
            self.assertGreater(age, 0, f"Example {i} age ska vara positivt")
            
            # Sex ska vara 0 eller 1
            self.assertIn(sex, [0.0, 1.0], f"Example {i} sex ska vara 0 eller 1")
            
            # Height ska vara positivt
            self.assertGreater(height, 0, f"Example {i} height ska vara positivt")
            
            # Weight ska vara positivt
            self.assertGreater(weight, 0, f"Example {i} weight ska vara positivt")
            
            # BMI ska vara positivt
            self.assertGreater(bmi, 0, f"Example {i} BMI ska vara positivt")
            
            # ASA ska vara mellan 1 och 5
            self.assertGreaterEqual(asa, 1, f"Example {i} ASA ska vara >= 1")
            self.assertLessEqual(asa, 5, f"Example {i} ASA ska vara <= 5")
        
        print(f"✅ T039 PASSED: Static structure [6] med patient features fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med korrekt static struktur")
    
    def test_t039_static_shape_6_edge_cases(self):
        """
        Verifiera static shape [6] med edge cases
        """
        # Arrange
        # Testa med olika edge cases
        test_cases = [
            ("Zero static", np.zeros((3, 6), dtype=np.float32)),
            ("Negative static", np.full((3, 6), -1.0, dtype=np.float32)),
            ("Large static", np.full((3, 6), 1000.0, dtype=np.float32)),
            ("Mixed static", np.array([
                [0.0, 1.0, -1.0, 0.5, -0.5, 2.0],
                [10.0, -10.0, 5.0, -5.0, 1.0, -1.0],
                [100.0, -100.0, 50.0, -50.0, 25.0, -25.0]
            ], dtype=np.float32))
        ]
        
        for case_name, static_data in test_cases:
            with self.subTest(case=case_name):
                # Arrange
                n_windows = static_data.shape[0]
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_edge_case_{case_name.replace(' ', '_')}")
                
                # Act
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert
                # Verifiera att fil skapas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {case_name}")
                
                # Verifiera att static features har korrekt shape [6]
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {case_name}")
                
                # Kontrollera static shape för varje example
                for i, example in enumerate(parsed_data):
                    static_shape = example['static'].shape
                    self.assertEqual(static_shape, (6,), 
                                   f"Example {i} static shape ska vara (6,) för {case_name}, fick {static_shape}")
                    
                    # Verifiera att värden matchar input
                    expected_static = static_data[i]
                    np.testing.assert_array_almost_equal(example['static'], expected_static, decimal=5,
                                                       err_msg=f"Example {i} static features ska matcha input för {case_name}")
        
        print(f"✅ T039 PASSED: Edge cases static shape [6] fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} edge cases")
    
    def test_t039_static_shape_6_validation(self):
        """
        Verifiera att static shape [6] valideras korrekt
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Testa med olika static shapes som ska faila
        invalid_static_shapes = [
            (n_windows, 5),   # För få static features
            (n_windows, 7),   # För många static features
            (n_windows, 4),   # För få static features
            (n_windows, 10),  # För många static features
        ]
        
        for invalid_shape in invalid_static_shapes:
            with self.subTest(shape=invalid_shape):
                # Arrange
                invalid_static_data = np.random.randn(*invalid_shape).astype(np.float32)
                output_path = os.path.join(self.temp_dir, f"test_invalid_shape_{invalid_shape[1]}")
                
                # Act & Assert
                # Detta ska raise ValueError
                with self.assertRaises(ValueError, msg=f"Static shape {invalid_shape} ska raise ValueError"):
                    self.tfrecord_creator.create_memory_efficient_tfrecord(
                        windows_data, invalid_static_data, targets_data, output_path, "test"
                    )
        
        print(f"✅ T039 PASSED: Static shape [6] validation fungerar korrekt")
        print(f"   Verifierade {len(invalid_static_shapes)} invalid shapes")
    
    def test_t039_static_shape_6_large_dataset(self):
        """
        Verifiera static shape [6] med stora datasets
        """
        # Arrange
        n_windows = 1000  # Stort dataset
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)  # Shape [6]
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_large_dataset")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas för stort dataset")
        
        # Verifiera att static features har korrekt shape [6]
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Kontrollera static shape för varje example
        for i, example in enumerate(parsed_data):
            static_shape = example['static'].shape
            self.assertEqual(static_shape, (6,), 
                           f"Example {i} static shape ska vara (6,), fick {static_shape}")
        
        print(f"✅ T039 PASSED: Large dataset static shape [6] fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med static shape [6]")
    
    def test_t039_static_shape_6_tfrecord_schema(self):
        """
        Verifiera att TFRecord schema stöder static shape [6]
        """
        # Arrange
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)  # Shape [6]
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
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
        
        # Verifiera att static features har korrekt shape [6] i schema
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        for example in parsed_data:
            static_shape = example['static'].shape
            self.assertEqual(static_shape, (6,), 
                           f"Static shape i schema ska vara (6,), fick {static_shape}")
        
        print(f"✅ T039 PASSED: TFRecord schema static shape [6] fungerar korrekt")
        print(f"   Schema validering: {schema_valid}")
    
    def test_t039_static_shape_6_comprehensive(self):
        """
        Omfattande test av static shape [6]
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
                static_data = np.random.randn(n_windows, 6).astype(np.float32)  # Shape [6]
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_comprehensive_{n_windows}")
                
                # Act
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert
                # Verifiera att fil skapas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {description}")
                
                # Verifiera att static features har korrekt shape [6]
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {description}")
                
                # Kontrollera static shape för varje example
                for i, example in enumerate(parsed_data):
                    static_shape = example['static'].shape
                    self.assertEqual(static_shape, (6,), 
                                   f"Example {i} static shape ska vara (6,) för {description}, fick {static_shape}")
        
        print(f"✅ T039 PASSED: Comprehensive static shape [6] test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
