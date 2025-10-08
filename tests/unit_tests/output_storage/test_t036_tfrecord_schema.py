#!/usr/bin/env python3
"""
T036: Test TFRecord Schema
Verifiera korrekt TFRecord schema med timeseries, static, targets
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

class TestT036TFRecordSchema(unittest.TestCase):
    """T036: Test TFRecord Schema"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t036_tfrecord_schema_basic(self):
        """
        T036: Test TFRecord Schema
        Verifiera korrekt TFRecord schema med timeseries, static, targets
        """
        # Arrange
        # Skapa test data enligt Master POC spec
        timeseries_data = np.random.randn(300, 16).astype(np.float32)  # [300, 16]
        static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)  # [6]
        targets_data = np.random.randn(8).astype(np.float32)  # [8]
        
        # Act
        schema = self.tfrecord_creator.create_tfrecord_schema()
        
        # Assert
        # Verifiera att schema innehåller alla required features
        self.assertIn('timeseries', schema, "Schema ska innehålla 'timeseries' feature")
        self.assertIn('static', schema, "Schema ska innehålla 'static' feature")
        self.assertIn('targets', schema, "Schema ska innehålla 'targets' feature")
        
        # Verifiera att schema har korrekt antal features
        self.assertEqual(len(schema), 3, "Schema ska ha exakt 3 features")
        
        # Verifiera timeseries feature
        timeseries_feature = schema['timeseries']
        self.assertEqual(timeseries_feature.shape, [],
                        "Timeseries feature ska ha shape [] för FixedLenSequenceFeature")
        self.assertEqual(timeseries_feature.dtype, tf.float32,
                        "Timeseries feature ska ha dtype tf.float32")
        
        # Verifiera static feature
        static_feature = schema['static']
        self.assertEqual(static_feature.shape, [],
                        "Static feature ska ha shape [] för FixedLenSequenceFeature")
        self.assertEqual(static_feature.dtype, tf.float32,
                        "Static feature ska ha dtype tf.float32")
        
        # Verifiera targets feature
        targets_feature = schema['targets']
        self.assertEqual(targets_feature.shape, [],
                        "Targets feature ska ha shape [] för FixedLenSequenceFeature")
        self.assertEqual(targets_feature.dtype, tf.float32,
                        "Targets feature ska ha dtype tf.float32")
        
        print("✅ T036 PASSED: Basic TFRecord schema fungerar korrekt")
    
    def test_t036_tfrecord_schema_example_creation(self):
        """
        Verifiera att TFRecord examples kan skapas med korrekt schema
        """
        # Arrange
        # Skapa test data
        timeseries_data = np.random.randn(300, 16).astype(np.float32)
        static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
        targets_data = np.random.randn(8).astype(np.float32)
        
        # Act
        example = self.tfrecord_creator.create_tfrecord_example(
            timeseries_data, static_data, targets_data
        )
        
        # Assert
        # Verifiera att example skapas utan fel
        self.assertIsNotNone(example, "TFRecord example ska skapas utan fel")
        self.assertIsInstance(example, tf.train.Example,
                             "TFRecord example ska vara av typ tf.train.Example")
        
        # Verifiera att example kan serialiseras
        serialized = example.SerializeToString()
        self.assertIsNotNone(serialized, "TFRecord example ska kunna serialiseras")
        self.assertGreater(len(serialized), 0, "Serialiserat example ska ha längd > 0")
        
        print("✅ T036 PASSED: TFRecord example creation fungerar korrekt")
    
    def test_t036_tfrecord_schema_shape_validation(self):
        """
        Verifiera att schema validerar korrekt shapes
        """
        # Arrange
        # Testa med korrekt data
        timeseries_data = np.random.randn(300, 16).astype(np.float32)
        static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
        targets_data = np.random.randn(8).astype(np.float32)
        
        # Act & Assert
        # Verifiera att korrekt data accepteras
        example = self.tfrecord_creator.create_tfrecord_example(
            timeseries_data, static_data, targets_data
        )
        self.assertIsNotNone(example, "Korrekt data ska accepteras")
        
        # Testa med felaktig timeseries shape
        wrong_timeseries = np.random.randn(200, 16).astype(np.float32)
        with self.assertRaises(ValueError, msg="Felaktig timeseries shape ska ge ValueError"):
            self.tfrecord_creator.create_tfrecord_example(
                wrong_timeseries, static_data, targets_data
            )
        
        # Testa med felaktig static shape
        wrong_static = np.array([25, 1, 170], dtype=np.float32)
        with self.assertRaises(ValueError, msg="Felaktig static shape ska ge ValueError"):
            self.tfrecord_creator.create_tfrecord_example(
                timeseries_data, wrong_static, targets_data
            )
        
        # Testa med felaktig targets shape
        wrong_targets = np.random.randn(7).astype(np.float32)
        with self.assertRaises(ValueError, msg="Felaktig targets shape ska ge ValueError"):
            self.tfrecord_creator.create_tfrecord_example(
                timeseries_data, static_data, wrong_targets
            )
        
        print("✅ T036 PASSED: TFRecord schema shape validation fungerar korrekt")
    
    def test_t036_tfrecord_schema_data_types(self):
        """
        Verifiera att schema hanterar korrekt data types
        """
        # Arrange
        # Testa med olika data types som ska konverteras till float32
        timeseries_data_int = np.random.randint(0, 100, (300, 16)).astype(np.int32)
        static_data_int = np.array([25, 1, 170, 70, 24, 2], dtype=np.int32)
        targets_data_int = np.random.randint(0, 10, 8).astype(np.int32)
        
        # Act
        example = self.tfrecord_creator.create_tfrecord_example(
            timeseries_data_int, static_data_int, targets_data_int
        )
        
        # Assert
        # Verifiera att example skapas även med int32 data
        self.assertIsNotNone(example, "Int32 data ska konverteras till float32")
        
        # Verifiera att data konverteras korrekt
        serialized = example.SerializeToString()
        self.assertIsNotNone(serialized, "Konverterad data ska kunna serialiseras")
        
        print("✅ T036 PASSED: TFRecord schema data types fungerar korrekt")
    
    def test_t036_tfrecord_schema_file_creation(self):
        """
        Verifiera att TFRecord filer kan skapas med korrekt schema
        """
        # Arrange
        # Skapa test data
        data_list = []
        for i in range(5):  # 5 examples
            timeseries_data = np.random.randn(300, 16).astype(np.float32)
            static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
            targets_data = np.random.randn(8).astype(np.float32)
            data_list.append((timeseries_data, static_data, targets_data))
        
        output_path = os.path.join(self.temp_dir, "test_tfrecord")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_tfrecord_file(
            data_list, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att fil har korrekt storlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "TFRecord fil ska ha storlek > 0")
        
        print("✅ T036 PASSED: TFRecord file creation fungerar korrekt")
    
    def test_t036_tfrecord_schema_readability(self):
        """
        Verifiera att skapade TFRecord filer kan läsas
        """
        # Arrange
        # Skapa test data
        data_list = []
        for i in range(3):  # 3 examples
            timeseries_data = np.random.randn(300, 16).astype(np.float32)
            static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
            targets_data = np.random.randn(8).astype(np.float32)
            data_list.append((timeseries_data, static_data, targets_data))
        
        output_path = os.path.join(self.temp_dir, "test_tfrecord")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_tfrecord_file(
            data_list, output_path, "test"
        )
        
        # Läs tillbaka data
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        
        # Assert
        # Verifiera att data kan läsas
        self.assertIsNotNone(parsed_data, "TFRecord fil ska kunna läsas")
        self.assertEqual(len(parsed_data), 3, "Ska läsa 3 examples")
        
        # Verifiera att varje example har korrekt struktur
        for i, example_data in enumerate(parsed_data):
            self.assertIn('timeseries', example_data, f"Example {i} ska ha 'timeseries'")
            self.assertIn('static', example_data, f"Example {i} ska ha 'static'")
            self.assertIn('targets', example_data, f"Example {i} ska ha 'targets'")
            
            # Verifiera shapes
            self.assertEqual(example_data['timeseries'].shape, (300, 16),
                           f"Example {i} timeseries ska ha shape (300, 16)")
            self.assertEqual(example_data['static'].shape, (6,),
                           f"Example {i} static ska ha shape (6,)")
            self.assertEqual(example_data['targets'].shape, (8,),
                           f"Example {i} targets ska ha shape (8,)")
        
        print("✅ T036 PASSED: TFRecord readability fungerar korrekt")
    
    def test_t036_tfrecord_schema_validation(self):
        """
        Verifiera att schema validation fungerar
        """
        # Arrange
        # Skapa test data
        data_list = []
        for i in range(2):  # 2 examples
            timeseries_data = np.random.randn(300, 16).astype(np.float32)
            static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
            targets_data = np.random.randn(8).astype(np.float32)
            data_list.append((timeseries_data, static_data, targets_data))
        
        output_path = os.path.join(self.temp_dir, "test_tfrecord")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_tfrecord_file(
            data_list, output_path, "test"
        )
        
        # Validera schema
        is_valid = self.tfrecord_creator.validate_tfrecord_schema(tfrecord_path)
        
        # Assert
        # Verifiera att schema validation fungerar
        self.assertTrue(is_valid, "TFRecord schema validation ska passera")
        
        print("✅ T036 PASSED: TFRecord schema validation fungerar korrekt")
    
    def test_t036_tfrecord_schema_custom_config(self):
        """
        Verifiera att schema fungerar med custom configuration
        """
        # Arrange
        # Skapa custom config med samma parametrar
        config = TFRecordConfig(
            timeseries_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            compression_type="GZIP"
        )
        
        tfrecord_creator_custom = MasterPOCTFRecordCreator(config)
        
        # Skapa test data
        timeseries_data = np.random.randn(300, 16).astype(np.float32)
        static_data = np.array([25, 1, 170, 70, 24, 2], dtype=np.float32)
        targets_data = np.random.randn(8).astype(np.float32)
        
        # Act
        schema = tfrecord_creator_custom.create_tfrecord_schema()
        example = tfrecord_creator_custom.create_tfrecord_example(
            timeseries_data, static_data, targets_data
        )
        
        # Assert
        # Verifiera att custom config ger samma schema som default
        self.assertEqual(len(schema), 3, "Custom config schema ska ha 3 features")
        self.assertIn('timeseries', schema, "Custom config schema ska ha 'timeseries'")
        self.assertIn('static', schema, "Custom config schema ska ha 'static'")
        self.assertIn('targets', schema, "Custom config schema ska ha 'targets'")
        
        # Verifiera att example skapas
        self.assertIsNotNone(example, "Custom config ska kunna skapa examples")
        
        print("✅ T036 PASSED: TFRecord schema custom config fungerar korrekt")
    
    def test_t036_tfrecord_schema_comprehensive(self):
        """
        Omfattande test av TFRecord schema
        """
        # Arrange
        # Testa med olika data storlekar och typer
        test_cases = [
            # (timeseries_shape, static_shape, targets_shape, description)
            ((300, 16), (6,), (8,), "Standard Master POC shapes"),
        ]
        
        # Act & Assert
        for timeseries_shape, static_shape, targets_shape, description in test_cases:
            # Skapa test data
            timeseries_data = np.random.randn(*timeseries_shape).astype(np.float32)
            static_data = np.random.randn(*static_shape).astype(np.float32)
            targets_data = np.random.randn(*targets_shape).astype(np.float32)
            
            # Skapa schema
            schema = self.tfrecord_creator.create_tfrecord_schema()
            
            # Verifiera schema
            self.assertEqual(schema['timeseries'].shape, [],
                           f"{description}: Timeseries schema ska ha shape [] för FixedLenSequenceFeature")
            self.assertEqual(schema['static'].shape, [],
                           f"{description}: Static schema ska ha shape [] för FixedLenSequenceFeature")
            self.assertEqual(schema['targets'].shape, [],
                           f"{description}: Targets schema ska ha shape [] för FixedLenSequenceFeature")
            
            # Skapa example
            example = self.tfrecord_creator.create_tfrecord_example(
                timeseries_data, static_data, targets_data
            )
            
            # Verifiera example
            self.assertIsNotNone(example, f"{description}: Example ska skapas")
            
            # Verifiera serialisering
            serialized = example.SerializeToString()
            self.assertIsNotNone(serialized, f"{description}: Example ska kunna serialiseras")
        
        print("✅ T036 PASSED: Comprehensive TFRecord schema test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
