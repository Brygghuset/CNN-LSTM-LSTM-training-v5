#!/usr/bin/env python3
"""
T050: Test Partial Save Recovery - Verifiera att partiellt sparad data kan återhämtas

AAA Format:
- Arrange: Skapa partiellt sparad data och simulera avbrott
- Act: Återhämta och fortsätt processing från checkpoint
- Assert: Verifiera att data är korrekt återhämtad och processing kan fortsätta
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import importlib.util
import json
from unittest.mock import Mock, patch, MagicMock

# Lägg till src i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Ladda moduler direkt för att undvika import-problem
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
tfrecord_creator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfrecord_creator_module)

spec2 = importlib.util.spec_from_file_location(
    "checkpoint_manager", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'checkpoint_manager.py')
)
checkpoint_manager_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(checkpoint_manager_module)

# Importera funktioner
MasterPOCTFRecordCreator = tfrecord_creator_module.MasterPOCTFRecordCreator
MasterPOCCheckpointManager = checkpoint_manager_module.MasterPOCCheckpointManager
create_checkpoint_manager = checkpoint_manager_module.create_checkpoint_manager


class TestT050PartialSaveRecovery(unittest.TestCase):
    """T050: Test Partial Save Recovery - Verifiera att partiellt sparad data kan återhämtas"""
    
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
        
        # Skapa checkpoint manager
        self.checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "checkpoints"),
            checkpoint_interval=5
        )
        
        # Skapa testdata
        self.batch_size = 3
        self.n_batches = 15  # Tillräckligt för att testa recovery
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
    
    def test_t050_partial_save_recovery_basic(self):
        """T050: Grundläggande partial save recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_basic")
        
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
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(f"batch_{i}")
        
        # Spara checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint("batch_4")
        
        # Act - Simulera restart och återhämta
        new_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "checkpoints"),
            checkpoint_interval=5
        )
        
        # Ladda checkpoint
        success = new_checkpoint_manager.load_checkpoint()
        
        # Assert - Checkpoint ska kunna laddas
        self.assertTrue(success, "Checkpoint ska kunna laddas")
        self.assertEqual(len(new_checkpoint_manager.processed_cases), 5, 
                        "Checkpoint ska innehålla 5 processed cases")
        
        # Verifiera att TFRecord-filer fortfarande finns
        for path in processed_batches:
            self.assertTrue(os.path.exists(path), 
                           f"TFRecord-fil ska finnas efter restart: {path}")
    
    def test_t050_partial_save_recovery_continue_processing(self):
        """T050: Fortsätt processing efter recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_continue")
        
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
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(f"batch_{i}")
        
        # Spara checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint("batch_4")
        
        # Act - Simulera restart och fortsätt processing
        new_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "checkpoints"),
            checkpoint_interval=5
        )
        
        # Ladda checkpoint
        success = new_checkpoint_manager.load_checkpoint()
        self.assertTrue(success, "Checkpoint ska kunna laddas")
        
        # Identifiera remaining cases
        processed_cases = new_checkpoint_manager.processed_cases
        all_cases = {f"batch_{i}" for i in range(self.n_batches)}
        remaining_cases = all_cases - processed_cases
        
        # Fortsätt processing med remaining cases
        new_processed_batches = []
        for case in remaining_cases:
            batch_id = int(case.split('_')[1])
            batch = self.batches[batch_id]
            batch_path = f"{output_path}_batch_{batch_id}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{batch_id}"
            )
            new_processed_batches.append(tfrecord_path)
            
            # Lägg till i checkpoint
            new_checkpoint_manager.add_processed_case(case)
        
        # Assert - Alla cases ska vara processade
        self.assertEqual(len(remaining_cases), self.n_batches - 5, 
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
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(f"batch_{i}")
        
        # Spara checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint("batch_4")
        
        # Simulera korrupt fil (skriv över med tom data)
        corrupted_file = processed_batches[2]  # Korrupta batch 2
        with open(corrupted_file, 'w') as f:
            f.write("corrupted data")
        
        # Act - Simulera restart och återhämta
        new_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "checkpoints"),
            checkpoint_interval=5
        )
        
        # Ladda checkpoint
        loaded_checkpoint = new_checkpoint_manager.load_checkpoint()
        
        # Verifiera TFRecord-filer
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
    
    def test_t050_partial_save_recovery_checkpoint_validation(self):
        """T050: Validera checkpoint data vid recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_validation")
        
        # Processera batches med olika status
        for i in range(8):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(f"batch_{i}")
            
            # Simulera fel för batch 3
            if i == 3:
                self.checkpoint_manager.add_failed_case(f"batch_{i}", "Test error")
        
        # Spara checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint("batch_4")
        
        # Act - Ladda checkpoint
        success = new_checkpoint_manager.load_checkpoint()
        
        # Assert - Checkpoint ska kunna laddas
        self.assertTrue(success, "Checkpoint ska kunna laddas")
        self.assertEqual(len(new_checkpoint_manager.processed_cases), 8, 
                        "Checkpoint ska innehålla 8 processed cases")
        self.assertEqual(len(new_checkpoint_manager.failed_cases), 1, 
                        "Checkpoint ska innehålla 1 failed case")
        
        # Verifiera att failed case är korrekt
        failed_cases = new_checkpoint_manager.failed_cases
        self.assertIn('batch_3', failed_cases, "Batch 3 ska vara i failed cases")
    
    def test_t050_partial_save_recovery_incremental_save(self):
        """T050: Incremental save under recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_incremental")
        
        # Processera första 3 batches
        for i in range(3):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(f"batch_{i}")
        
        # Spara checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint("batch_4")
        
        # Act - Simulera restart och fortsätt med incremental save
        new_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "checkpoints"),
            checkpoint_interval=3
        )
        
        # Ladda checkpoint
        success = new_checkpoint_manager.load_checkpoint()
        self.assertTrue(success, "Checkpoint ska kunna laddas")
        
        # Fortsätt processing med incremental save
        processed_count = len(new_checkpoint_manager.processed_cases)
        
        for i in range(processed_count, min(processed_count + 6, self.n_batches)):
            batch = self.batches[i]
            batch_path = f"{output_path}_batch_{i}.tfrecord"
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                batch_path, f"batch_{i}"
            )
            
            # Lägg till i checkpoint
            new_checkpoint_manager.add_processed_case(f"batch_{i}")
            
            # Spara checkpoint var 3:e batch
            if (i + 1) % 3 == 0:
                new_checkpoint_manager.save_checkpoint(f"batch_{i}")
        
        # Assert - Checkpoint ska innehålla rätt antal cases
        final_checkpoint = new_checkpoint_manager.load_checkpoint()
        self.assertEqual(len(final_checkpoint['processed_cases']), 
                        min(processed_count + 6, self.n_batches),
                        "Final checkpoint ska innehålla rätt antal cases")
    
    def test_t050_partial_save_recovery_edge_cases(self):
        """T050: Edge cases för partial save recovery"""
        # Arrange
        output_path = os.path.join(self.temp_dir, "test_recovery_edge")
        
        # Test 1: Tom checkpoint
        empty_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "empty_checkpoints"),
            checkpoint_interval=5
        )
        
        # Spara tom checkpoint
        empty_checkpoint_manager.save_checkpoint("empty")
        
        # Ladda tom checkpoint
        success = empty_checkpoint_manager.load_checkpoint()
        
        # Assert - Tom checkpoint ska hanteras korrekt
        self.assertTrue(success, "Tom checkpoint ska kunna laddas")
        self.assertEqual(len(empty_checkpoint_manager.processed_cases), 0, 
                        "Tom checkpoint ska ha 0 processed cases")
        
        # Test 2: Checkpoint med endast failed cases
        failed_checkpoint_manager = MasterPOCCheckpointManager(
            checkpoint_path=os.path.join(self.temp_dir, "failed_checkpoints"),
            checkpoint_interval=5
        )
        
        # Lägg till failed cases
        for i in range(3):
            failed_checkpoint_manager.add_failed_case(f"batch_{i}", f"Error {i}")
        
        # Spara checkpoint
        failed_checkpoint_manager.save_checkpoint("failed")
        
        # Ladda checkpoint
        success = failed_checkpoint_manager.load_checkpoint()
        
        # Assert - Checkpoint med failed cases ska hanteras korrekt
        self.assertTrue(success, "Failed checkpoint ska kunna laddas")
        self.assertEqual(len(failed_checkpoint_manager.failed_cases), 3, 
                        "Failed checkpoint ska ha 3 failed cases")
        self.assertEqual(len(failed_checkpoint_manager.processed_cases), 0, 
                        "Failed checkpoint ska ha 0 processed cases")


if __name__ == '__main__':
    unittest.main()
