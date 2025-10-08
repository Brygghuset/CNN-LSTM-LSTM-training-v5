#!/usr/bin/env python3
"""
T060: Test Checkpoint Interval
Verifiera att checkpoints sparas med konfigurerad interval
"""

import unittest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår checkpoint manager modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "checkpoint_manager", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'checkpoint_manager.py')
)
checkpoint_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_manager)

# Använd modulen
MasterPOCCheckpointManager = checkpoint_manager.MasterPOCCheckpointManager
create_checkpoint_manager = checkpoint_manager.create_checkpoint_manager

class TestT060CheckpointInterval(unittest.TestCase):
    """T060: Test Checkpoint Interval"""
    
    def setUp(self):
        """Setup för varje test."""
        self.checkpoint_path = "s3://test-bucket/checkpoints/test"
    
    def test_t060_checkpoint_interval_default(self):
        """
        T060: Test Checkpoint Interval
        Verifiera att checkpoints sparas med konfigurerad interval
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50  # Default interval
        )
        
        # Act & Assert
        # Verifiera att interval är korrekt
        self.assertEqual(checkpoint_manager.checkpoint_interval, 50)
        
        # Verifiera att checkpoint sparas var 50:e case
        for i in range(100):
            case_id = f"{i:04d}"
            
            # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
            should_save = checkpoint_manager.should_save_checkpoint(case_id)
            expected_save = (i + 1) % 50 == 0
            
            self.assertEqual(should_save, expected_save,
                           f"Case {i+1} ska {'spara' if expected_save else 'inte spara'} checkpoint")
            
            # Lägg till case efter kontrollen
            checkpoint_manager.add_processed_case(case_id)
        
        print("✅ T060 PASSED: Default checkpoint interval (50) fungerar korrekt")
    
    def test_t060_checkpoint_interval_custom(self):
        """
        Verifiera checkpoint interval med anpassad interval
        """
        # Arrange
        custom_intervals = [10, 25, 100, 200]
        
        for interval in custom_intervals:
            checkpoint_manager = create_checkpoint_manager(
                checkpoint_path=self.checkpoint_path,
                enable_checkpoints=True,
                checkpoint_interval=interval
            )
            
            # Act & Assert
            self.assertEqual(checkpoint_manager.checkpoint_interval, interval)
            
            # Verifiera att checkpoint sparas med rätt interval
            for i in range(interval * 3):  # Testa 3 intervals
                case_id = f"{i:04d}"
                
                # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
                should_save = checkpoint_manager.should_save_checkpoint(case_id)
                expected_save = (i + 1) % interval == 0
                
                self.assertEqual(should_save, expected_save,
                               f"Case {i+1} med interval {interval} ska {'spara' if expected_save else 'inte spara'} checkpoint")
                
                # Lägg till case efter kontrollen
                checkpoint_manager.add_processed_case(case_id)
        
        print("✅ T060 PASSED: Anpassade checkpoint intervals fungerar korrekt")
    
    def test_t060_checkpoint_interval_save_frequency(self):
        """
        Verifiera att checkpoint sparas med rätt frekvens
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=20
        )
        
        # Mock S3 save
        with patch.object(checkpoint_manager.s3_client, 'put_object') as mock_put:
            mock_put.return_value = {}
            
            # Act
            save_count = 0
            for i in range(100):
                case_id = f"{i:04d}"
                checkpoint_manager.add_processed_case(case_id)
                
                if checkpoint_manager.should_save_checkpoint(case_id):
                    checkpoint_manager.save_checkpoint(case_id)
                    save_count += 1
            
            # Assert
            expected_saves = 100 // 20  # 5 saves för 100 cases med interval 20
            self.assertEqual(save_count, expected_saves,
                           f"Checkpoint ska sparas {expected_saves} gånger för 100 cases med interval 20")
            
        print("✅ T060 PASSED: Checkpoint save frequency fungerar korrekt")
    
    def test_t060_checkpoint_interval_edge_cases(self):
        """
        Verifiera checkpoint interval med edge cases
        """
        # Arrange
        edge_cases = [
            (1, 100),    # Varje case
            (2, 50),     # Varannan case
            (5, 20),     # Var 5:e case
            (1000, 0),   # Var 1000:e case (ingen för 100 cases)
        ]
        
        for interval, expected_saves in edge_cases:
            checkpoint_manager = create_checkpoint_manager(
                checkpoint_path=self.checkpoint_path,
                enable_checkpoints=True,
                checkpoint_interval=interval
            )
            
            # Act
            save_count = 0
            for i in range(100):
                case_id = f"{i:04d}"
                
                # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
                if checkpoint_manager.should_save_checkpoint(case_id):
                    save_count += 1
                
                # Lägg till case efter kontrollen
                checkpoint_manager.add_processed_case(case_id)
            
            # Assert
            self.assertEqual(save_count, expected_saves,
                           f"Interval {interval} ska ge {expected_saves} saves för 100 cases")
        
        print("✅ T060 PASSED: Checkpoint interval edge cases fungerar korrekt")
    
    def test_t060_checkpoint_interval_disabled(self):
        """
        Verifiera att checkpoint interval ignoreras när disabled
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=False,
            checkpoint_interval=10
        )
        
        # Act & Assert
        # Verifiera att checkpoint aldrig sparas när disabled
        for i in range(50):
            case_id = f"{i:04d}"
            should_save = checkpoint_manager.should_save_checkpoint(case_id)
            self.assertFalse(should_save,
                           f"Case {i+1} ska inte spara checkpoint när disabled")
        
        print("✅ T060 PASSED: Checkpoint interval ignoreras när disabled")
    
    def test_t060_checkpoint_interval_configuration(self):
        """
        Verifiera att checkpoint interval konfigureras korrekt
        """
        # Arrange
        test_intervals = [1, 5, 10, 25, 50, 100, 500]
        
        for interval in test_intervals:
            # Act
            checkpoint_manager = create_checkpoint_manager(
                checkpoint_path=self.checkpoint_path,
                enable_checkpoints=True,
                checkpoint_interval=interval
            )
            
            # Assert
            self.assertEqual(checkpoint_manager.checkpoint_interval, interval)
            self.assertTrue(checkpoint_manager.enable_checkpoints)
            self.assertEqual(checkpoint_manager.checkpoint_path, self.checkpoint_path)
        
        print("✅ T060 PASSED: Checkpoint interval konfiguration fungerar korrekt")
    
    def test_t060_checkpoint_interval_performance(self):
        """
        Verifiera att checkpoint interval inte påverkar prestanda
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Act
        import time
        start_time = time.time()
        
        # Simulera många case checks
        for i in range(10000):
            case_id = f"{i:04d}"
            checkpoint_manager.should_save_checkpoint(case_id)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert
        # Verifiera att processing är snabb (mindre än 1 sekund för 10000 cases)
        self.assertLess(processing_time, 1.0,
                       f"Checkpoint interval check ska vara snabb, tog {processing_time:.3f}s")
        
        print("✅ T060 PASSED: Checkpoint interval prestanda är acceptabel")
    
    def test_t060_checkpoint_interval_consistency(self):
        """
        Verifiera att checkpoint interval är konsistent över tid
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=30
        )
        
        # Act & Assert
        # Verifiera att interval förblir konstant
        for batch in range(5):  # 5 batches
            for i in range(30):  # 30 cases per batch
                case_id = f"{batch * 30 + i:04d}"
                
                # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
                should_save = checkpoint_manager.should_save_checkpoint(case_id)
                expected_save = (i + 1) % 30 == 0
                
                self.assertEqual(should_save, expected_save,
                               f"Case {case_id} ska {'spara' if expected_save else 'inte spara'} checkpoint")
                
                # Lägg till case efter kontrollen
                checkpoint_manager.add_processed_case(case_id)
        
        print("✅ T060 PASSED: Checkpoint interval konsistens fungerar korrekt")
    
    def test_t060_checkpoint_interval_master_poc_spec(self):
        """
        Verifiera att checkpoint interval följer Master POC specifikation
        """
        # Arrange
        # Enligt AWS_CHECKLIST_V5.0_3000_CASES.md: checkpoint_interval = 50
        master_poc_interval = 50
        
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=master_poc_interval
        )
        
        # Act & Assert
        self.assertEqual(checkpoint_manager.checkpoint_interval, master_poc_interval)
        
        # Verifiera att checkpoint sparas var 50:e case enligt spec
        checkpoint_saves = []
        for i in range(300):  # Testa 300 cases (6 checkpoints)
            case_id = f"{i:04d}"
            
            # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
            if checkpoint_manager.should_save_checkpoint(case_id):
                checkpoint_saves.append(i + 1)
            
            # Lägg till case efter kontrollen
            checkpoint_manager.add_processed_case(case_id)
        
        expected_saves = [50, 100, 150, 200, 250, 300]
        self.assertEqual(checkpoint_saves, expected_saves,
                       f"Master POC spec ska ge checkpoints vid {expected_saves}")
        
        print("✅ T060 PASSED: Checkpoint interval följer Master POC specifikation")

if __name__ == '__main__':
    unittest.main()
