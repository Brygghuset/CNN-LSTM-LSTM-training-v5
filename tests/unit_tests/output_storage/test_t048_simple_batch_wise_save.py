#!/usr/bin/env python3
"""
T048: Test Batch-wise Save - Enkel test för att isolera problemet

AAA Format:
- Arrange: Skapa minimal testdata
- Act: Använd create_memory_efficient_tfrecord
- Assert: Verifiera att filen skapas och kan läsas
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import importlib.util

# Lägg till src i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Ladda modulen direkt för att undvika import-problem
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
tfrecord_creator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfrecord_creator_module)

# Importera funktioner
MasterPOCTFRecordCreator = tfrecord_creator_module.MasterPOCTFRecordCreator


class TestT048SimpleBatchWiseSave(unittest.TestCase):
    """T048: Enkel test för batch-wise save"""
    
    def setUp(self):
        """Setup testdata och temporär directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skapa TFRecordConfig för testet
        TFRecordConfig = tfrecord_creator_module.TFRecordConfig
        config = TFRecordConfig(
            timeseries_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            compression_type="GZIP"  # Prova utan compression först
        )
        self.tfrecord_creator = MasterPOCTFRecordCreator(config)
    
    def tearDown(self):
        """Cleanup temporär directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_t048_simple_batch_save(self):
        """T048: Enkel test för batch save"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_simple")
        
        # Skapa minimal testdata
        batch_size = 2
        windows_data = np.random.randn(batch_size, 300, 16).astype(np.float32)
        static_data = np.random.randn(batch_size, 6).astype(np.float32)
        targets_data = np.random.randn(batch_size, 8).astype(np.float32)
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data,
            output_path, "test"
        )
        
        # Assert
        self.assertIsNotNone(tfrecord_path, "TFRecord path ska inte vara None")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord-fil ska finnas")
        
        # Verifiera filstorlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "TFRecord-fil ska ha innehåll")
        
        print(f"TFRecord-fil skapad: {tfrecord_path}")
        print(f"Filstorlek: {file_size} bytes")
    
    def test_t048_no_compression(self):
        """T048: Test utan compression"""
        # Arrange
        TFRecordConfig = tfrecord_creator_module.TFRecordConfig
        config = TFRecordConfig(
            timeseries_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            compression_type=""  # Ingen compression
        )
        tfrecord_creator = MasterPOCTFRecordCreator(config)
        
        output_path = os.path.join(self.temp_dir, "test_no_compression")
        
        # Skapa minimal testdata
        batch_size = 2
        windows_data = np.random.randn(batch_size, 300, 16).astype(np.float32)
        static_data = np.random.randn(batch_size, 6).astype(np.float32)
        targets_data = np.random.randn(batch_size, 8).astype(np.float32)
        
        # Act
        tfrecord_path = tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data,
            output_path, "test"
        )
        
        # Assert
        self.assertIsNotNone(tfrecord_path, "TFRecord path ska inte vara None")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord-fil ska finnas")
        
        # Verifiera filstorlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "TFRecord-fil ska ha innehåll")
        
        print(f"TFRecord-fil utan compression: {tfrecord_path}")
        print(f"Filstorlek: {file_size} bytes")
    
    def test_t048_schema_validation(self):
        """T048: Test schema validation"""
        # Arrange
        schema = self.tfrecord_creator.create_tfrecord_schema()
        
        # Assert
        self.assertIn('timeseries', schema, "Schema ska innehålla timeseries")
        self.assertIn('static', schema, "Schema ska innehålla static")
        self.assertIn('targets', schema, "Schema ska innehålla targets")
        
        print(f"Schema: {schema}")
    
    def test_t048_example_creation(self):
        """T048: Test example creation"""
        # Arrange
        timeseries_data = np.random.randn(300, 16).astype(np.float32)
        static_data = np.random.randn(6).astype(np.float32)
        targets_data = np.random.randn(8).astype(np.float32)
        
        # Act
        example = self.tfrecord_creator.create_tfrecord_example(
            timeseries_data, static_data, targets_data
        )
        
        # Assert
        self.assertIsNotNone(example, "Example ska inte vara None")
        
        # Verifiera att example kan serialiseras
        serialized = example.SerializeToString()
        self.assertGreater(len(serialized), 0, "Serialized example ska ha innehåll")
        
        print(f"Example skapad, storlek: {len(serialized)} bytes")


if __name__ == '__main__':
    unittest.main()
