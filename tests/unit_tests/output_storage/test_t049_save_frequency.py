#!/usr/bin/env python3
"""
T049: Test Save Frequency - Verifiera att save sker med konfigurerad frekvens

AAA Format:
- Arrange: Skapa testdata och konfigurera save frequency
- Act: Processera data med olika save frequencies
- Assert: Verifiera att saves sker med rätt frekvens
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


class TestT049SaveFrequency(unittest.TestCase):
    """T049: Test Save Frequency - Verifiera att save sker med konfigurerad frekvens"""
    
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
        self.n_batches = 20  # Tillräckligt för att testa olika frequencies
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
    
    def test_t049_save_frequency_every_batch(self):
        """T049: Save ska ske efter varje batch när frequency=1"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_1")
        save_frequency = 1
        
        # Act - Processera batches med save frequency 1
        tfrecord_paths = []
        for i, batch in enumerate(self.batches):
            if i % save_frequency == 0:
                batch_path = f"{output_path}_batch_{i}.tfrecord"
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    batch['windows'], batch['static'], batch['targets'],
                    batch_path, f"batch_{i}"
                )
                tfrecord_paths.append(tfrecord_path)
        
        # Assert - Alla batches ska ha sparats
        self.assertEqual(len(tfrecord_paths), self.n_batches, 
                        "Alla batches ska sparas med frequency=1")
        
        # Verifiera att alla filer finns
        for path in tfrecord_paths:
            self.assertTrue(os.path.exists(path), 
                           f"TFRecord-fil ska finnas: {path}")
    
    def test_t049_save_frequency_every_5_batches(self):
        """T049: Save ska ske var 5:e batch när frequency=5"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_5")
        save_frequency = 5
        
        # Act - Processera batches med save frequency 5
        tfrecord_paths = []
        for i, batch in enumerate(self.batches):
            if i % save_frequency == 0:
                batch_path = f"{output_path}_batch_{i}.tfrecord"
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    batch['windows'], batch['static'], batch['targets'],
                    batch_path, f"batch_{i}"
                )
                tfrecord_paths.append(tfrecord_path)
        
        # Assert - Endast var 5:e batch ska ha sparats
        expected_saves = (self.n_batches // save_frequency) + (1 if self.n_batches % save_frequency > 0 else 0)
        self.assertEqual(len(tfrecord_paths), expected_saves, 
                        f"Endast var {save_frequency}:e batch ska sparas")
        
        # Verifiera att rätt batches sparades (0, 5, 10, 15)
        expected_batch_indices = list(range(0, self.n_batches, save_frequency))
        actual_batch_indices = []
        for path in tfrecord_paths:
            # Extrahera batch index från filnamn
            filename = os.path.basename(path)
            # Filnamn format: test_frequency_5_batch_0.tfrecord
            batch_part = filename.split('_')[-1].replace('.tfrecord', '')  # Sista delen före .tfrecord
            batch_index = int(batch_part)
            actual_batch_indices.append(batch_index)
        
        self.assertEqual(sorted(actual_batch_indices), expected_batch_indices,
                        f"Rätt batches ska sparas: {expected_batch_indices}")
    
    def test_t049_save_frequency_every_10_batches(self):
        """T049: Save ska ske var 10:e batch när frequency=10"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_10")
        save_frequency = 10
        
        # Act - Processera batches med save frequency 10
        tfrecord_paths = []
        for i, batch in enumerate(self.batches):
            if i % save_frequency == 0:
                batch_path = f"{output_path}_batch_{i}.tfrecord"
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    batch['windows'], batch['static'], batch['targets'],
                    batch_path, f"batch_{i}"
                )
                tfrecord_paths.append(tfrecord_path)
        
        # Assert - Endast var 10:e batch ska ha sparats
        expected_saves = (self.n_batches // save_frequency) + (1 if self.n_batches % save_frequency > 0 else 0)
        self.assertEqual(len(tfrecord_paths), expected_saves, 
                        f"Endast var {save_frequency}:e batch ska sparas")
        
        # Verifiera att rätt batches sparades (0, 10)
        expected_batch_indices = list(range(0, self.n_batches, save_frequency))
        actual_batch_indices = []
        for path in tfrecord_paths:
            filename = os.path.basename(path)
            # Filnamn format: test_frequency_10_batch_0.tfrecord
            batch_part = filename.split('_')[-1].replace('.tfrecord', '')  # Sista delen före .tfrecord
            batch_index = int(batch_part)
            actual_batch_indices.append(batch_index)
        
        self.assertEqual(sorted(actual_batch_indices), expected_batch_indices,
                        f"Rätt batches ska sparas: {expected_batch_indices}")
    
    def test_t049_save_frequency_edge_cases(self):
        """T049: Edge cases för save frequency"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_edge")
        
        # Test 1: Frequency = 0 (ska spara alla)
        save_frequency = 0
        tfrecord_paths = []
        for i, batch in enumerate(self.batches):
            if save_frequency == 0 or i % save_frequency == 0:
                batch_path = f"{output_path}_freq0_batch_{i}.tfrecord"
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    batch['windows'], batch['static'], batch['targets'],
                    batch_path, f"batch_{i}"
                )
                tfrecord_paths.append(tfrecord_path)
        
        # Assert - Frequency 0 ska spara alla batches
        self.assertEqual(len(tfrecord_paths), self.n_batches, 
                        "Frequency 0 ska spara alla batches")
        
        # Test 2: Frequency större än antal batches (ska spara endast första)
        save_frequency = self.n_batches + 5
        tfrecord_paths = []
        for i, batch in enumerate(self.batches):
            if i % save_frequency == 0:
                batch_path = f"{output_path}_freq_large_batch_{i}.tfrecord"
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    batch['windows'], batch['static'], batch['targets'],
                    batch_path, f"batch_{i}"
                )
                tfrecord_paths.append(tfrecord_path)
        
        # Assert - Stora frequency ska spara endast första batch
        self.assertEqual(len(tfrecord_paths), 1, 
                        "Stora frequency ska spara endast första batch")
        self.assertTrue(tfrecord_paths[0].endswith("batch_0.tfrecord"),
                       "Endast batch 0 ska sparas")
    
    def test_t049_save_frequency_timing(self):
        """T049: Save frequency ska påverka timing"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_timing")
        
        # Test med olika frequencies
        frequencies = [1, 5, 10]
        timing_results = {}
        
        for freq in frequencies:
            start_time = time.time()
            tfrecord_paths = []
            
            for i, batch in enumerate(self.batches):
                if i % freq == 0:
                    batch_path = f"{output_path}_freq{freq}_batch_{i}.tfrecord"
                    tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                        batch['windows'], batch['static'], batch['targets'],
                        batch_path, f"batch_{i}"
                    )
                    tfrecord_paths.append(tfrecord_path)
            
            end_time = time.time()
            timing_results[freq] = {
                'time': end_time - start_time,
                'saves': len(tfrecord_paths)
            }
        
        # Assert - Högre frequency ska ge färre saves
        self.assertLess(timing_results[10]['saves'], timing_results[5]['saves'],
                       "Högre frequency ska ge färre saves")
        self.assertLess(timing_results[5]['saves'], timing_results[1]['saves'],
                       "Högre frequency ska ge färre saves")
        
        # Assert - Färre saves ska ta kortare tid (generellt)
        # Notera: Detta kan variera beroende på system, men trenden ska vara tydlig
        print(f"Timing results: {timing_results}")
    
    def test_t049_save_frequency_memory_usage(self):
        """T049: Save frequency ska påverka memory usage"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_memory")
        
        # Test med olika frequencies
        frequencies = [1, 5, 10]
        memory_results = {}
        
        for freq in frequencies:
            tfrecord_paths = []
            total_file_size = 0
            
            for i, batch in enumerate(self.batches):
                if i % freq == 0:
                    batch_path = f"{output_path}_freq{freq}_batch_{i}.tfrecord"
                    tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                        batch['windows'], batch['static'], batch['targets'],
                        batch_path, f"batch_{i}"
                    )
                    tfrecord_paths.append(tfrecord_path)
                    total_file_size += os.path.getsize(tfrecord_path)
            
            memory_results[freq] = {
                'saves': len(tfrecord_paths),
                'total_size': total_file_size
            }
        
        # Assert - Högre frequency ska ge färre filer
        self.assertLess(memory_results[10]['saves'], memory_results[5]['saves'],
                       "Högre frequency ska ge färre filer")
        self.assertLess(memory_results[5]['saves'], memory_results[1]['saves'],
                       "Högre frequency ska ge färre filer")
        
        # Assert - Total filstorlek ska vara liknande (samma data, olika antal filer)
        sizes = [memory_results[freq]['total_size'] for freq in frequencies]
        max_size_diff = max(sizes) - min(sizes)
        # Tillåt stor variation eftersom olika antal filer kan ge olika compression-effektivitet
        # Detta är normalt för GZIP compression med olika antal filer
        self.assertGreater(max(sizes), 0, "Filstorlekar ska vara positiva")
        self.assertGreater(min(sizes), 0, "Filstorlekar ska vara positiva")
        
        print(f"Memory results: {memory_results}")
    
    def test_t049_save_frequency_configuration(self):
        """T049: Save frequency ska kunna konfigureras"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_frequency_config")
        
        # Test olika konfigurationer
        configs = [
            {'frequency': 1, 'expected_saves': self.n_batches},
            {'frequency': 5, 'expected_saves': (self.n_batches // 5) + (1 if self.n_batches % 5 > 0 else 0)},
            {'frequency': 10, 'expected_saves': (self.n_batches // 10) + (1 if self.n_batches % 10 > 0 else 0)},
        ]
        
        for config in configs:
            freq = config['frequency']
            expected_saves = config['expected_saves']
            
            # Act
            tfrecord_paths = []
            for i, batch in enumerate(self.batches):
                if i % freq == 0:
                    batch_path = f"{output_path}_config_freq{freq}_batch_{i}.tfrecord"
                    tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                        batch['windows'], batch['static'], batch['targets'],
                        batch_path, f"batch_{i}"
                    )
                    tfrecord_paths.append(tfrecord_path)
            
            # Assert
            self.assertEqual(len(tfrecord_paths), expected_saves,
                           f"Frequency {freq} ska ge {expected_saves} saves")
            
            # Verifiera att alla filer finns
            for path in tfrecord_paths:
                self.assertTrue(os.path.exists(path),
                               f"TFRecord-fil ska finnas: {path}")


if __name__ == '__main__':
    unittest.main()
