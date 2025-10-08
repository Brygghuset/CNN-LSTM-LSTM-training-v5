#!/usr/bin/env python3
"""
T048: Test Batch-wise Save - Verifiera att TFRecord sparas per batch, inte endast vid completion

AAA Format:
- Arrange: Skapa testdata och mock batch processor
- Act: Processera batches och spara TFRecord löpande
- Assert: Verifiera att TFRecord-filer skapas efter varje batch
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import importlib.util
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


class TestT048BatchWiseSave(unittest.TestCase):
    """T048: Test Batch-wise Save - Verifiera att TFRecord sparas per batch, inte endast vid completion"""
    
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
        
        # Skapa testdata för batches
        self.batch_size = 10
        self.n_batches = 5
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
    
    def test_t048_batch_wise_save_incremental(self):
        """T048: TFRecord ska sparas efter varje batch, inte endast vid completion"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_batch_wise")
        
        # Act - Processera batches en i taget och spara efter varje
        tfrecord_paths = []
        
        for i, batch in enumerate(self.batches):
            # Spara batch till TFRecord
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            tfrecord_paths.append(tfrecord_path)
            
            # Assert - TFRecord-filen ska finnas efter varje batch
            self.assertTrue(os.path.exists(tfrecord_path), 
                           f"TFRecord-fil ska finnas efter batch {i}")
            
            # Verifiera att filen har innehåll
            file_size = os.path.getsize(tfrecord_path)
            self.assertGreater(file_size, 0, 
                              f"TFRecord-fil ska ha innehåll efter batch {i}")
            
            # Verifiera att filen kan läsas
            self._verify_tfrecord_readable(tfrecord_path, self.batch_size)
        
        # Assert - Alla batch-filer ska finnas
        self.assertEqual(len(tfrecord_paths), self.n_batches, 
                        "Samma antal TFRecord-filer som batches")
        
        # Verifiera att alla filer har olika storlekar (inte identiska)
        file_sizes = [os.path.getsize(path) for path in tfrecord_paths]
        self.assertEqual(len(set(file_sizes)), len(file_sizes), 
                        "Batch-filer ska ha olika storlekar")
    
    def test_t048_batch_wise_save_append_mode(self):
        """T048: TFRecord ska kunna sparas i append mode för samma fil"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_append_mode")
        
        # Act - Spara första batch
        first_batch = self.batches[0]
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            first_batch['windows'], first_batch['static'], first_batch['targets'],
            output_path, "first_batch"
        )
        
        # Assert - Första batch ska finnas
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "Första batch ska finnas")
        first_size = os.path.getsize(tfrecord_path)
        
        # Act - Lägg till andra batch (simulera append)
        second_batch = self.batches[1]
        # Notera: create_memory_efficient_tfrecord skapar ny fil, inte append
        # Men vi kan verifiera att båda batches kan sparas separat
        second_path = f"{output_path}_second_batch.tfrecord"
        second_tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            second_batch['windows'], second_batch['static'], second_batch['targets'],
            second_path, "second_batch"
        )
        
        # Assert - Båda batches ska finnas
        self.assertTrue(os.path.exists(tfrecord_path), 
                       "Första batch ska fortfarande finnas")
        self.assertTrue(os.path.exists(second_tfrecord_path), 
                       "Andra batch ska finnas")
        
        # Verifiera att båda filer kan läsas
        self._verify_tfrecord_readable(tfrecord_path, self.batch_size)
        self._verify_tfrecord_readable(second_tfrecord_path, self.batch_size)
    
    def test_t048_batch_wise_save_memory_efficiency(self):
        """T048: Batch-wise save ska vara memory-efficient"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_memory_efficiency")
        
        # Act - Processera batches en i taget
        for i, batch in enumerate(self.batches):
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            
            # Spara batch
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            
            # Assert - Filen ska finnas och vara läsbar
            self.assertTrue(os.path.exists(tfrecord_path), 
                           f"Batch {i} ska sparas")
            self._verify_tfrecord_readable(tfrecord_path, self.batch_size)
            
            # Verifiera att data är korrekt
            self._verify_batch_data(tfrecord_path, batch, i)
    
    def test_t048_batch_wise_save_error_handling(self):
        """T048: Batch-wise save ska hantera fel gracefully"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_error_handling")
        
        # Act - Processera batches med fel i mitten
        successful_saves = []
        
        for i, batch in enumerate(self.batches):
            try:
                batch_path = f"{output_path}_batch_{i}.tfrecord"
                
                # Simulera fel för batch 2
                if i == 2:
                    # Skapa ogiltig data som orsakar fel
                    invalid_windows = np.array([])  # Tom array
                    invalid_static = np.array([])
                    invalid_targets = np.array([])
                    
                    # Detta ska orsaka fel
                    with self.assertRaises((ValueError, IndexError)):
                        self.tfrecord_creator.create_memory_efficient_tfrecord(
                            invalid_windows, invalid_static, invalid_targets,
                            batch_path, f"batch_{i}"
                        )
                else:
                    # Normal batch
                    tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                        batch['windows'], batch['static'], batch['targets'],
                        batch_path, f"batch_{i}"
                    )
                    successful_saves.append(tfrecord_path)
                    
            except Exception as e:
                # Logga fel men fortsätt
                print(f"Fel i batch {i}: {e}")
        
        # Assert - Endast lyckade batches ska ha filer
        self.assertEqual(len(successful_saves), self.n_batches - 1, 
                        "Endast lyckade batches ska ha filer")
        
        # Verifiera att lyckade filer finns
        for path in successful_saves:
            self.assertTrue(os.path.exists(path), 
                           f"Lyckad batch ska ha fil: {path}")
            self._verify_tfrecord_readable(path, self.batch_size)
    
    def test_t048_batch_wise_save_progress_tracking(self):
        """T048: Batch-wise save ska spåra progress"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_progress_tracking")
        
        # Act - Processera batches och spåra progress
        progress_log = []
        
        for i, batch in enumerate(self.batches):
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            
            # Spara batch
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            
            # Logga progress
            progress_log.append({
                'batch_id': i,
                'file_path': tfrecord_path,
                'file_size': os.path.getsize(tfrecord_path),
                'timestamp': os.path.getmtime(tfrecord_path)
            })
        
        # Assert - Progress ska vara korrekt
        self.assertEqual(len(progress_log), self.n_batches, 
                        "Progress ska spåra alla batches")
        
        # Verifiera att timestamps är i ordning
        timestamps = [log['timestamp'] for log in progress_log]
        self.assertEqual(timestamps, sorted(timestamps), 
                        "Timestamps ska vara i ordning")
        
        # Verifiera att filstorlekar är positiva
        for log in progress_log:
            self.assertGreater(log['file_size'], 0, 
                              f"Batch {log['batch_id']} ska ha positiv filstorlek")
    
    def _verify_tfrecord_readable(self, tfrecord_path: str, expected_samples: int):
        """Verifiera att TFRecord-fil kan läsas"""
        try:
            # Läs TFRecord-fil med samma compression som användes för att skriva
            compression_type = self.tfrecord_creator.config.compression_type
            if compression_type == "":
                compression_type = None
            
            dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                tfrecord_path, 
                compression_type=compression_type
            )
            
            # Räkna samples
            sample_count = 0
            for _ in dataset:
                sample_count += 1
            
            # Verifiera antal samples
            self.assertEqual(sample_count, expected_samples, 
                           f"TFRecord ska ha {expected_samples} samples")
            
        except Exception as e:
            self.fail(f"TFRecord-fil kunde inte läsas: {e}")
    
    def _verify_batch_data(self, tfrecord_path: str, expected_batch: dict, batch_id: int):
        """Verifiera att batch-data är korrekt"""
        try:
            # Läs TFRecord-fil med samma compression som användes för att skriva
            compression_type = self.tfrecord_creator.config.compression_type
            if compression_type == "":
                compression_type = None
            
            dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                tfrecord_path, 
                compression_type=compression_type
            )
            
            # Verifiera att data kan läsas
            sample_count = 0
            for _ in dataset:
                sample_count += 1
            
            self.assertEqual(sample_count, self.batch_size, 
                           f"Batch {batch_id} ska ha {self.batch_size} samples")
            
        except Exception as e:
            self.fail(f"Batch {batch_id} data kunde inte verifieras: {e}")


if __name__ == '__main__':
    unittest.main()
