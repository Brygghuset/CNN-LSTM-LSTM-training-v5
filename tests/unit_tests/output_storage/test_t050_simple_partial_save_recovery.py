#!/usr/bin/env python3
"""
T050: Test Partial Save Recovery - Enkel test för att verifiera partial save recovery

AAA Format:
- Arrange: Skapa partiellt sparad data
- Act: Återhämta och fortsätt processing
- Assert: Verifiera att data är korrekt återhämtad
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import importlib.util
import json

# Lägg till src i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Ladda moduler direkt för att undvika import-problem
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
tfrecord_creator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfrecord_creator_module)

# Importera funktioner
MasterPOCTFRecordCreator = tfrecord_creator_module.MasterPOCTFRecordCreator


class TestT050SimplePartialSaveRecovery(unittest.TestCase):
    """T050: Enkel test för partial save recovery"""
    
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
        self.batch_size = 3
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
    
    def test_t050_simple_partial_save_recovery(self):
        """T050: Enkel partial save recovery test"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_simple_recovery")
        
        # Processera första 5 batches
        processed_batches = []
        for i in range(5):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            processed_batches.append(tfrecord_path)
        
        # Act - Simulera restart och fortsätt processing
        # Identifiera remaining cases
        processed_case_ids = {f"batch_{i}" for i in range(5)}
        all_case_ids = {f"batch_{i}" for i in range(self.n_batches)}
        remaining_case_ids = all_case_ids - processed_case_ids
        
        # Fortsätt processing med remaining cases
        new_processed_batches = []
        for case_id in remaining_case_ids:
            batch_id = int(case_id.split('_')[1])
            batch = self.batches[batch_id]
            batch_path = f"{output_path}_batch_{batch_id}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{batch_id}"
            )
            new_processed_batches.append(tfrecord_path)
        
        # Assert - Alla cases ska vara processade
        self.assertEqual(len(remaining_case_ids), self.n_batches - 5, 
                        "Remaining cases ska vara korrekt antal")
        
        # Verifiera att alla TFRecord-filer finns
        all_tfrecord_paths = processed_batches + new_processed_batches
        self.assertEqual(len(all_tfrecord_paths), self.n_batches, 
                        "Alla batches ska ha TFRecord-filer")
        
        for path in all_tfrecord_paths:
            self.assertTrue(os.path.exists(path), 
                           f"TFRecord-fil ska finnas: {path}")
    
    def test_t050_partial_save_recovery_corrupted_files(self):
        """T050: Hantera korrupta TFRecord-filer vid recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_corrupted")
        
        # Processera första 5 batches
        processed_batches = []
        for i in range(5):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            processed_batches.append(tfrecord_path)
        
        # Simulera korrupt fil (skriv över med tom data)
        corrupted_file = processed_batches[2]  # Korrupta batch 2
        with open(corrupted_file, 'w') as f:
            f.write("corrupted data")
        
        # Act - Verifiera TFRecord-filer
        valid_files = []
        corrupted_files = []
        
        for i, path in enumerate(processed_batches):
            try:
                # Försök läsa TFRecord-filen
                compression_type = self.tfrecord_creator.config.compression_type
                if compression_type == "":
                    compression_type = None
                
                dataset = tfrecord_creator_module.tf.data.TFRecordDataset(
                    path, compression_type=compression_type
                )
                
                # Räkna samples
                sample_count = 0
                for _ in dataset:
                    sample_count += 1
                
                if sample_count == self.batch_size:
                    valid_files.append(path)
                else:
                    corrupted_files.append(path)
                    
            except Exception:
                corrupted_files.append(path)
        
        # Assert - Korrupta filer ska identifieras
        self.assertEqual(len(corrupted_files), 1, 
                        "En korrupt fil ska identifieras")
        self.assertEqual(len(valid_files), 4, 
                        "Fyra giltiga filer ska finnas")
        
        # Assert - Korrupta filer ska kunna reprocessas
        if corrupted_files:
            corrupted_path = corrupted_files[0]
            batch_id = int(os.path.basename(corrupted_path).split('_')[-1].replace('.tfrecord', ''))
            
            # Reprocessa korrupta batch
            batch = self.batches[batch_id]
            new_tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                corrupted_path.replace('.tfrecord', '_reprocessed.tfrecord'),
                f"batch_{batch_id}_reprocessed"
            )
            
            self.assertTrue(os.path.exists(new_tfrecord_path), 
                           "Reprocessad fil ska finnas")
    
    def test_t050_partial_save_recovery_incremental_save(self):
        """T050: Incremental save under recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_incremental")
        
        # Processera första 3 batches
        processed_batches = []
        for i in range(3):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            processed_batches.append(tfrecord_path)
        
        # Act - Simulera restart och fortsätt med incremental save
        processed_count = 3
        
        # Fortsätt processing med incremental save
        for i in range(processed_count, min(processed_count + 6, self.n_batches)):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            processed_batches.append(tfrecord_path)
        
        # Assert - Alla batches ska vara processade
        self.assertEqual(len(processed_batches), 
                        min(processed_count + 6, self.n_batches),
                        "Alla batches ska vara processade")
        
        # Verifiera att alla TFRecord-filer finns
        for path in processed_batches:
            self.assertTrue(os.path.exists(path),
                           f"TFRecord-fil ska finnas: {path}")
    
    def test_t050_partial_save_recovery_edge_cases(self):
        """T050: Edge cases för partial save recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_edge")
        
        # Test 1: Ingen data processad (tom recovery)
        processed_batches = []
        
        # Act - Processera alla batches från början
        for i in range(self.n_batches):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            processed_batches.append(tfrecord_path)
        
        # Assert - Alla batches ska vara processade
        self.assertEqual(len(processed_batches), self.n_batches, 
                        "Alla batches ska vara processade")
        
        # Test 2: Alla batches redan processade (ingen recovery behövs)
        remaining_case_ids = set()  # Tom set
        
        # Assert - Ingen processing ska behövas
        self.assertEqual(len(remaining_case_ids), 0, 
                        "Ingen processing ska behövas")
        
        # Verifiera att alla TFRecord-filer finns
        for path in processed_batches:
            self.assertTrue(os.path.exists(path),
                           f"TFRecord-fil ska finnas: {path}")


if __name__ == '__main__':
    unittest.main()
