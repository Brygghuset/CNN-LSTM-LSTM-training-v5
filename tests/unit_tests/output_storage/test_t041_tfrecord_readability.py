#!/usr/bin/env python3
"""
T041: Test TFRecord Readability
Verifiera att skapade TFRecord-filer kan läsas av TensorFlow
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

class TestT041TFRecordReadability(unittest.TestCase):
    """T041: Test TFRecord Readability"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t041_tfrecord_readability_basic(self):
        """
        T041: Test TFRecord Readability
        Verifiera att skapade TFRecord-filer kan läsas av TensorFlow
        """
        # Arrange
        # Skapa test data
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_readability")
        
        # Act - Skapa TFRecord fil
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert - Verifiera att fil kan läsas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att fil kan läsas med TensorFlow
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Verifiera att data har korrekt struktur
        for i, example in enumerate(parsed_data):
            # Verifiera att example innehåller alla nödvändiga keys
            self.assertIn('timeseries', example, f"Example {i} ska innehålla 'timeseries'")
            self.assertIn('static', example, f"Example {i} ska innehålla 'static'")
            self.assertIn('targets', example, f"Example {i} ska innehålla 'targets'")
            
            # Verifiera att data har korrekt shape
            self.assertEqual(example['timeseries'].shape, (300, 16), 
                           f"Example {i} timeseries ska ha shape (300, 16)")
            self.assertEqual(example['static'].shape, (6,), 
                           f"Example {i} static ska ha shape (6,)")
            self.assertEqual(example['targets'].shape, (8,), 
                           f"Example {i} targets ska ha shape (8,)")
            
            # Verifiera att data är numpy arrays
            self.assertIsInstance(example['timeseries'], np.ndarray, 
                                f"Example {i} timeseries ska vara numpy array")
            self.assertIsInstance(example['static'], np.ndarray, 
                                f"Example {i} static ska vara numpy array")
            self.assertIsInstance(example['targets'], np.ndarray, 
                                f"Example {i} targets ska vara numpy array")
        
        print(f"✅ T041 PASSED: Basic TFRecord readability fungerar korrekt")
        print(f"   Läs {len(parsed_data)} examples från TFRecord fil")
    
    def test_t041_tfrecord_readability_tensorflow_dataset(self):
        """
        Verifiera att TFRecord-filer kan läsas med TensorFlow Dataset API
        """
        # Arrange
        n_windows = 20
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_tensorflow_dataset")
        
        # Act - Skapa TFRecord fil
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert - Verifiera att fil kan läsas med TensorFlow Dataset API
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Skapa TensorFlow Dataset från TFRecord fil
        schema = self.tfrecord_creator.create_tfrecord_schema()
        
        def parse_example(example_proto):
            parsed = tf.io.parse_single_example(example_proto, schema)
            # Reshape the parsed features to their original shapes
            timeseries = tf.reshape(parsed['timeseries'], (300, 16))
            static = tf.reshape(parsed['static'], (6,))
            targets = tf.reshape(parsed['targets'], (8,))
            return timeseries, static, targets
        
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
        dataset = dataset.map(parse_example)
        
        # Verifiera att dataset kan itereras
        examples_read = 0
        for timeseries, static, targets in dataset:
            # Verifiera att data har korrekt shape
            self.assertEqual(timeseries.shape, (300, 16), 
                           f"TensorFlow Dataset timeseries ska ha shape (300, 16)")
            self.assertEqual(static.shape, (6,), 
                           f"TensorFlow Dataset static ska ha shape (6,)")
            self.assertEqual(targets.shape, (8,), 
                           f"TensorFlow Dataset targets ska ha shape (8,)")
            
            # Verifiera att data är TensorFlow tensors
            self.assertIsInstance(timeseries, tf.Tensor, 
                                f"TensorFlow Dataset timeseries ska vara tf.Tensor")
            self.assertIsInstance(static, tf.Tensor, 
                                f"TensorFlow Dataset static ska vara tf.Tensor")
            self.assertIsInstance(targets, tf.Tensor, 
                                f"TensorFlow Dataset targets ska vara tf.Tensor")
            
            examples_read += 1
        
        # Verifiera att alla examples läses
        self.assertEqual(examples_read, n_windows, f"Ska läsa {n_windows} examples med TensorFlow Dataset")
        
        print(f"✅ T041 PASSED: TensorFlow Dataset API readability fungerar korrekt")
        print(f"   Läs {examples_read} examples med TensorFlow Dataset")
    
    def test_t041_tfrecord_readability_data_integrity(self):
        """
        Verifiera att data integritet bevaras vid läsning
        """
        # Arrange
        n_windows = 5
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
        
        output_path = os.path.join(self.temp_dir, "test_data_integrity")
        
        # Act - Skapa TFRecord fil
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert - Verifiera att data integritet bevaras
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Läs data tillbaka
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Verifiera att data matchar input
        for i, example in enumerate(parsed_data):
            # Verifiera timeseries data
            expected_timeseries = windows_data[i]
            np.testing.assert_array_almost_equal(example['timeseries'], expected_timeseries, decimal=5,
                                               err_msg=f"Example {i} timeseries ska matcha input")
            
            # Verifiera static data
            expected_static = static_data[i]
            np.testing.assert_array_almost_equal(example['static'], expected_static, decimal=5,
                                               err_msg=f"Example {i} static ska matcha input")
            
            # Verifiera targets data
            expected_targets = targets_data[i]
            np.testing.assert_array_almost_equal(example['targets'], expected_targets, decimal=5,
                                               err_msg=f"Example {i} targets ska matcha input")
        
        print(f"✅ T041 PASSED: Data integrity vid läsning fungerar korrekt")
        print(f"   Verifierade {n_windows} examples med specifika värden")
    
    def test_t041_tfrecord_readability_large_dataset(self):
        """
        Verifiera att stora TFRecord-filer kan läsas korrekt
        """
        # Arrange
        n_windows = 1000  # Stort dataset
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_large_readability")
        
        # Act - Skapa TFRecord fil
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert - Verifiera att stor fil kan läsas
        self.assertIsNotNone(tfrecord_path, "Stor TFRecord fil ska skapas")
        
        # Läs data tillbaka
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples från stor fil")
        
        # Verifiera att alla examples har korrekt struktur
        for i, example in enumerate(parsed_data):
            # Verifiera att data har korrekt shape
            self.assertEqual(example['timeseries'].shape, (300, 16), 
                           f"Example {i} timeseries ska ha shape (300, 16)")
            self.assertEqual(example['static'].shape, (6,), 
                           f"Example {i} static ska ha shape (6,)")
            self.assertEqual(example['targets'].shape, (8,), 
                           f"Example {i} targets ska ha shape (8,)")
            
            # Verifiera att data är numpy arrays
            self.assertIsInstance(example['timeseries'], np.ndarray, 
                                f"Example {i} timeseries ska vara numpy array")
            self.assertIsInstance(example['static'], np.ndarray, 
                                f"Example {i} static ska vara numpy array")
            self.assertIsInstance(example['targets'], np.ndarray, 
                                f"Example {i} targets ska vara numpy array")
        
        print(f"✅ T041 PASSED: Large dataset readability fungerar korrekt")
        print(f"   Läs {len(parsed_data)} examples från stor fil")
    
    def test_t041_tfrecord_readability_compression_types(self):
        """
        Verifiera att TFRecord-filer med olika compression types kan läsas
        """
        # Arrange
        n_windows = 50
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Testa olika compression types
        compression_types = ["GZIP", "ZLIB", None]
        created_files = []
        
        # Act - Skapa filer med olika compression
        for compression_type in compression_types:
            config = TFRecordConfig(compression_type=compression_type)
            tfrecord_creator = MasterPOCTFRecordCreator(config)
            
            output_path = os.path.join(self.temp_dir, f"test_compression_readability_{compression_type or 'none'}")
            tfrecord_path = tfrecord_creator.create_memory_efficient_tfrecord(
                windows_data, static_data, targets_data, output_path, "test"
            )
            created_files.append((compression_type, tfrecord_path, tfrecord_creator))
        
        # Assert - Verifiera att alla filer kan läsas
        self.assertEqual(len(created_files), len(compression_types), "Ska skapa alla compression filer")
        
        for compression_type, tfrecord_path, tfrecord_creator in created_files:
            # Verifiera att fil existerar
            self.assertIsNotNone(tfrecord_path, f"TFRecord fil med {compression_type} compression ska skapas")
            self.assertTrue(os.path.exists(tfrecord_path), f"TFRecord fil med {compression_type} compression ska existera")
            
            # Verifiera att fil kan läsas
            parsed_data = tfrecord_creator.read_tfrecord_file(tfrecord_path)
            self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples från {compression_type} fil")
            
            # Verifiera att data har korrekt struktur
            for i, example in enumerate(parsed_data):
                self.assertEqual(example['timeseries'].shape, (300, 16), 
                               f"Example {i} timeseries ska ha shape (300, 16) för {compression_type}")
                self.assertEqual(example['static'].shape, (6,), 
                               f"Example {i} static ska ha shape (6,) för {compression_type}")
                self.assertEqual(example['targets'].shape, (8,), 
                               f"Example {i} targets ska ha shape (8,) för {compression_type}")
        
        print(f"✅ T041 PASSED: Compression types readability fungerar korrekt")
        print(f"   Testade {len(compression_types)} compression types")
    
    def test_t041_tfrecord_readability_error_handling(self):
        """
        Verifiera att error handling fungerar för ogiltiga TFRecord-filer
        """
        # Arrange
        invalid_tfrecord_path = os.path.join(self.temp_dir, "invalid_file.tfrecord")
        
        # Skapa en ogiltig fil
        with open(invalid_tfrecord_path, 'w') as f:
            f.write("This is not a valid TFRecord file")
        
        # Assert - Verifiera att error handling fungerar
        with self.assertRaises(Exception, msg="Ogiltig TFRecord fil ska raise Exception"):
            self.tfrecord_creator.read_tfrecord_file(invalid_tfrecord_path)
        
        # Testa med fil som inte finns
        non_existent_path = os.path.join(self.temp_dir, "non_existent.tfrecord")
        with self.assertRaises(FileNotFoundError, msg="Icke-existerande fil ska raise FileNotFoundError"):
            self.tfrecord_creator.read_tfrecord_file(non_existent_path)
        
        print(f"✅ T041 PASSED: Error handling för ogiltiga TFRecord-filer fungerar korrekt")
    
    def test_t041_tfrecord_readability_performance(self):
        """
        Verifiera att TFRecord-läsning har bra prestanda
        """
        # Arrange
        import time
        
        n_windows = 500
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_performance")
        
        # Act - Skapa TFRecord fil
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert - Verifiera att fil kan läsas snabbt
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Mät läsningstid
        start_time = time.time()
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        end_time = time.time()
        
        reading_time = end_time - start_time
        
        # Verifiera att data läses korrekt
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples")
        
        # Verifiera att läsningstid är rimlig (< 5 sekunder för 500 examples)
        self.assertLess(reading_time, 5, f"Läsningstid ska vara < 5s, fick {reading_time:.2f}s")
        
        # Beräkna throughput
        throughput = n_windows / reading_time
        self.assertGreater(throughput, 50, f"Throughput ska vara > 50 examples/s, fick {throughput:.2f} examples/s")
        
        print(f"✅ T041 PASSED: Performance för TFRecord-läsning fungerar korrekt")
        print(f"   Läsningstid: {reading_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} examples/s")
    
    def test_t041_tfrecord_readability_comprehensive(self):
        """
        Omfattande test av TFRecord readability
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
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_comprehensive_{n_windows}")
                
                # Act - Skapa TFRecord fil
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert - Verifiera att fil kan läsas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {description}")
                
                # Läs data tillbaka
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {description}")
                
                # Verifiera att alla examples har korrekt struktur
                for i, example in enumerate(parsed_data):
                    self.assertEqual(example['timeseries'].shape, (300, 16), 
                                   f"Example {i} timeseries ska ha shape (300, 16) för {description}")
                    self.assertEqual(example['static'].shape, (6,), 
                                   f"Example {i} static ska ha shape (6,) för {description}")
                    self.assertEqual(example['targets'].shape, (8,), 
                                   f"Example {i} targets ska ha shape (8,) för {description}")
        
        print(f"✅ T041 PASSED: Comprehensive TFRecord readability test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
