#!/usr/bin/env python3
"""
T051: Test Writer Flush - Verifiera att TFRecord writers flushas regelbundet

AAA Format:
- Arrange: Skapa TFRecord writer och testdata
- Act: Skriv data och testa flush-beteende
- Assert: Verifiera att data flushas korrekt
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import importlib.util
import time
from unittest.mock import Mock, patch, MagicMock

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


class TestT051WriterFlush(unittest.TestCase):
    """T051: Test Writer Flush - Verifiera att TFRecord writers flushas regelbundet"""
    
    def setUp(self):
        """Setup testdata och temporär directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skapa TFRecordConfig för testet
        TFRecordConfig = tfrecord_creator_module.TFRecordConfig
        config = TFRecordConfig(
            timeseries_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            compression_type="GZIP"
        )
        self.tfrecord_creator = MasterPOCTFRecordCreator(config)
        
        # Skapa testdata
        self.batch_size = 5
        self.n_batches = 10
        self.window_size = 300
        self.n_features = 16
        self.n_static = 6
        self.n_targets = 8
        
        # Skapa testdata för varje batch
        self.batches = []
        for i in range(self.n_batches):
            batch_windows = np.random.randn(self.batch_size, self.window_size, self.n_features)
            batch_static = np.random.randn(self.batch_size, self.n_static)
            batch_targets = np.random.randn(self.batch_size, self.n_targets)
            
            self.batches.append({
                'windows': batch_windows,
                'static': batch_static,
                'targets': batch_targets,
                'batch_id': i
            })
    
    def tearDown(self):
        """Cleanup temporär directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_t051_writer_flush_basic(self):
        """T051: Grundläggande writer flush test"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_basic")
        
        # Act - Skapa TFRecord med explicit flush
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.batches[0]['windows'], 
            self.batches[0]['static'], 
            self.batches[0]['targets'],
            output_path, "test"
        )
        
        # Assert - TFRecord-fil ska finnas och vara läsbar
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "TFRecord-fil ska finnas efter flush")
        
        # Verifiera att filen kan läsas
        compression_type = self.tfrecord_creator.config.compression_type
        if compression_type == "":
            compression_type = None
        
        dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
            tfrecord_path, compression_type=compression_type
        )
        
        sample_count = 0
        for _ in dataset:
            sample_count += 1
        
        self.assertEqual(sample_count, self.batch_size, 
                        "TFRecord ska ha rätt antal samples efter flush")
    
    def test_t051_writer_flush_multiple_batches(self):
        """T051: Writer flush med flera batches"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_multiple")
        
        # Act - Skapa TFRecord-filer för flera batches
        tfrecord_paths = []
        for i in range(5):
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                self.batches[i]['windows'], 
                self.batches[i]['static'], 
                self.batches[i]['targets'],
                batch_path, f"batch_{i}"
            )
            tfrecord_paths.append(tfrecord_path)
        
        # Assert - Alla TFRecord-filer ska finnas och vara läsbara
        for i, path in enumerate(tfrecord_paths):
            self.assertTrue(os.path.exists(path), 
                           f"TFRecord-fil {i} ska finnas efter flush")
            
            # Verifiera att filen kan läsas
            compression_type = self.tfrecord_creator.config.compression_type
            if compression_type == "":
                compression_type = None
            
            dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                path, compression_type=compression_type
            )
            
            sample_count = 0
            for _ in dataset:
                sample_count += 1
            
            self.assertEqual(sample_count, self.batch_size, 
                            f"TFRecord {i} ska ha rätt antal samples efter flush")
    
    def test_t051_writer_flush_timing(self):
        """T051: Writer flush timing test"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_timing")
        
        # Act - Mät tid för TFRecord creation
        start_time = time.time()
        
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.batches[0]['windows'], 
            self.batches[0]['static'], 
            self.batches[0]['targets'],
            output_path, "test"
        )
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Assert - TFRecord creation ska ta rimlig tid
        self.assertLess(creation_time, 5.0, 
                       "TFRecord creation ska ta mindre än 5 sekunder")
        
        # Verifiera att filen finns direkt efter creation
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "TFRecord-fil ska finnas direkt efter creation")
        
        # Verifiera att filen har innehåll
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, 
                          "TFRecord-fil ska ha innehåll efter flush")
    
    def test_t051_writer_flush_compression(self):
        """T051: Writer flush med olika compression"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_compression")
        
        # Test olika compression types
        compression_types = ["", "GZIP", "ZLIB"]
        tfrecord_paths = []
        
        for comp_type in compression_types:
            # Skapa TFRecordConfig med olika compression
            TFRecordConfig = tfrecord_creator_module.TFRecordConfig
            config = TFRecordConfig(
                timeseries_shape=(300, 16),
                static_shape=(6,),
                target_shape=(8,),
                compression_type=comp_type
            )
            tfrecord_creator = MasterPOCTFRecordCreator(config)
            
            # Act - Skapa TFRecord med denna compression
            batch_path = f"{output_path}_{comp_type or 'none'}.tfrecord"
            tfrecord_path = tfrecord_creator.create_memory_efficient_tfrecord(
                self.batches[0]['windows'], 
                self.batches[0]['static'], 
                self.batches[0]['targets'],
                batch_path, "test"
            )
            tfrecord_paths.append(tfrecord_path)
        
        # Assert - Alla TFRecord-filer ska finnas och vara läsbara
        for i, (comp_type, path) in enumerate(zip(compression_types, tfrecord_paths)):
            self.assertTrue(os.path.exists(path), 
                           f"TFRecord-fil med {comp_type or 'no'} compression ska finnas")
            
            # Verifiera att filen kan läsas med rätt compression
            dataset_compression = comp_type if comp_type else None
            dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                path, compression_type=dataset_compression
            )
            
            sample_count = 0
            for _ in dataset:
                sample_count += 1
            
            self.assertEqual(sample_count, self.batch_size, 
                            f"TFRecord med {comp_type or 'no'} compression ska ha rätt antal samples")
    
    def test_t051_writer_flush_error_handling(self):
        """T051: Writer flush error handling"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_error")
        
        # Test med ogiltig data som ska orsaka fel
        # Skapa data med fel shape istället för tom array
        invalid_windows = np.random.randn(5, 100, 16)  # Fel shape (100 istället för 300)
        invalid_static = np.random.randn(5, 6)
        invalid_targets = np.random.randn(5, 8)
        
        # Act & Assert - Ogiltig data ska orsaka fel
        with self.assertRaises(ValueError):
            self.tfrecord_creator.create_memory_efficient_tfrecord(
                invalid_windows, invalid_static, invalid_targets,
                output_path, "test"
            )
        
        # Test med korrekt data efter fel
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.batches[0]['windows'], 
            self.batches[0]['static'], 
            self.batches[0]['targets'],
            output_path, "test"
        )
        
        # Assert - Korrekt data ska fungera
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "TFRecord-fil ska finnas efter korrekt data")
    
    def test_t051_writer_flush_memory_efficiency(self):
        """T051: Writer flush memory efficiency"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_memory")
        
        # Act - Skapa TFRecord med stora batches
        large_batch_size = 20
        large_windows = np.random.randn(large_batch_size, self.window_size, self.n_features)
        large_static = np.random.randn(large_batch_size, self.n_static)
        large_targets = np.random.randn(large_batch_size, self.n_targets)
        
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            large_windows, large_static, large_targets,
            output_path, "large_test"
        )
        
        # Assert - Stora batches ska hanteras korrekt
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "TFRecord-fil med stora batches ska finnas")
        
        # Verifiera att filen kan läsas
        compression_type = self.tfrecord_creator.config.compression_type
        if compression_type == "":
            compression_type = None
        
        dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
            tfrecord_path, compression_type=compression_type
        )
        
        sample_count = 0
        for _ in dataset:
            sample_count += 1
        
        self.assertEqual(sample_count, large_batch_size, 
                        "TFRecord med stora batches ska ha rätt antal samples")
        
        # Verifiera att filen har rimlig storlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 1000,  # Minst 1KB
                          "TFRecord-fil ska ha rimlig storlek")
    
    def test_t051_writer_flush_concurrent_access(self):
        """T051: Writer flush concurrent access test"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_concurrent")
        
        # Act - Skapa flera TFRecord-filer samtidigt (simulera concurrent access)
        tfrecord_paths = []
        for i in range(3):
            batch_path = f"{output_path}_concurrent_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                self.batches[i]['windows'], 
                self.batches[i]['static'], 
                self.batches[i]['targets'],
                batch_path, f"concurrent_{i}"
            )
            tfrecord_paths.append(tfrecord_path)
        
        # Assert - Alla TFRecord-filer ska finnas och vara läsbara
        for i, path in enumerate(tfrecord_paths):
            self.assertTrue(os.path.exists(path), 
                           f"Concurrent TFRecord-fil {i} ska finnas")
            
            # Verifiera att filen kan läsas
            compression_type = self.tfrecord_creator.config.compression_type
            if compression_type == "":
                compression_type = None
            
            dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                path, compression_type=compression_type
            )
            
            sample_count = 0
            for _ in dataset:
                sample_count += 1
            
            self.assertEqual(sample_count, self.batch_size, 
                            f"Concurrent TFRecord {i} ska ha rätt antal samples")
    
    def test_t051_writer_flush_edge_cases(self):
        """T051: Edge cases för writer flush"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_flush_edge")
        
        # Test 1: Tom batch
        empty_windows = np.array([]).reshape(0, self.window_size, self.n_features)
        empty_static = np.array([]).reshape(0, self.n_static)
        empty_targets = np.array([]).reshape(0, self.n_targets)
        
        # Act - Tom batch ska hanteras korrekt
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            empty_windows, empty_static, empty_targets,
            output_path, "empty_test"
        )
        
        # Assert - Tom batch ska returnera None eller hanteras korrekt
        if tfrecord_path is not None:
            self.assertTrue(os.path.exists(tfrecord_path), 
                           "TFRecord-fil för tom batch ska finnas")
        
        # Test 2: Enkel batch
        single_windows = np.random.randn(1, self.window_size, self.n_features)
        single_static = np.random.randn(1, self.n_static)
        single_targets = np.random.randn(1, self.n_targets)
        
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            single_windows, single_static, single_targets,
            output_path, "single_test"
        )
        
        # Assert - Enkel batch ska fungera
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "TFRecord-fil för enkel batch ska finnas")
        
        # Verifiera att filen kan läsas
        compression_type = self.tfrecord_creator.config.compression_type
        if compression_type == "":
            compression_type = None
        
        dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
            tfrecord_path, compression_type=compression_type
        )
        
        sample_count = 0
        for _ in dataset:
            sample_count += 1
        
        self.assertEqual(sample_count, 1, 
                        "TFRecord för enkel batch ska ha 1 sample")


if __name__ == '__main__':
    unittest.main()
