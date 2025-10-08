#!/usr/bin/env python3
"""
T058: Test Checkpoint Creation
Verifiera att checkpoints skapas med korrekt format
"""

import unittest
import sys
import os
import json
import tempfile
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

class TestT058CheckpointCreation(unittest.TestCase):
    """T058: Test Checkpoint Creation"""
    
    def setUp(self):
        """Setup för varje test."""
        self.checkpoint_path = "s3://test-bucket/checkpoints/test"
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
    
    def test_t058_checkpoint_data_structure(self):
        """
        T058: Test Checkpoint Creation
        Verifiera att checkpoints skapas med korrekt format
        """
        # Arrange
        self.checkpoint_manager.add_processed_case("0001")
        self.checkpoint_manager.add_processed_case("0002")
        self.checkpoint_manager.add_failed_case("0003", "Test error")
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera att alla nödvändiga fält finns
        self.assertIn('checkpoint_info', checkpoint_data)
        self.assertIn('processing_state', checkpoint_data)
        self.assertIn('processed_cases', checkpoint_data)
        self.assertIn('failed_cases', checkpoint_data)
        self.assertIn('failed_case_details', checkpoint_data)
        
        # Verifiera checkpoint_info struktur
        checkpoint_info = checkpoint_data['checkpoint_info']
        self.assertIn('timestamp', checkpoint_info)
        self.assertIn('checkpoint_path', checkpoint_info)
        self.assertIn('checkpoint_interval', checkpoint_info)
        self.assertIn('enable_checkpoints', checkpoint_info)
        self.assertIn('processing_time_seconds', checkpoint_info)
        
        # Verifiera processing_state struktur
        processing_state = checkpoint_data['processing_state']
        self.assertIn('current_case_index', processing_state)
        self.assertIn('total_processed', processing_state)
        self.assertIn('total_failed', processing_state)
        self.assertIn('last_checkpoint_time', processing_state)
        
        print("✅ T058 PASSED: Checkpoint data structure är korrekt")
    
    def test_t058_checkpoint_data_content(self):
        """
        Verifiera att checkpoint data innehåller korrekt information
        """
        # Arrange
        self.checkpoint_manager.add_processed_case("0001")
        self.checkpoint_manager.add_processed_case("0002")
        self.checkpoint_manager.add_failed_case("0003", "Test error")
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera checkpoint_info värden
        checkpoint_info = checkpoint_data['checkpoint_info']
        self.assertEqual(checkpoint_info['checkpoint_path'], self.checkpoint_path)
        self.assertEqual(checkpoint_info['checkpoint_interval'], 50)
        self.assertTrue(checkpoint_info['enable_checkpoints'])
        self.assertIsInstance(checkpoint_info['timestamp'], str)
        self.assertIsInstance(checkpoint_info['processing_time_seconds'], float)
        
        # Verifiera processing_state värden
        processing_state = checkpoint_data['processing_state']
        self.assertEqual(processing_state['current_case_index'], 3)
        self.assertEqual(processing_state['total_processed'], 2)
        self.assertEqual(processing_state['total_failed'], 1)
        
        # Verifiera case lists
        self.assertEqual(set(checkpoint_data['processed_cases']), {"0001", "0002"})
        self.assertEqual(set(checkpoint_data['failed_cases']), {"0003"})
        
        print("✅ T058 PASSED: Checkpoint data content är korrekt")
    
    def test_t058_checkpoint_json_serialization(self):
        """
        Verifiera att checkpoint data kan serialiseras till JSON
        """
        # Arrange
        self.checkpoint_manager.add_processed_case("0001")
        self.checkpoint_manager.add_processed_case("0002")
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera att data kan serialiseras till JSON
        try:
            checkpoint_json = json.dumps(checkpoint_data, indent=2)
            self.assertIsInstance(checkpoint_json, str)
            
            # Verifiera att JSON kan deserialiseras
            deserialized_data = json.loads(checkpoint_json)
            self.assertEqual(deserialized_data, checkpoint_data)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Checkpoint data kunde inte serialiseras till JSON: {e}")
        
        print("✅ T058 PASSED: Checkpoint JSON serialization fungerar korrekt")
    
    def test_t058_checkpoint_timestamp_format(self):
        """
        Verifiera att timestamp är i korrekt ISO format
        """
        # Arrange
        self.checkpoint_manager.add_processed_case("0001")
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        timestamp = checkpoint_data['checkpoint_info']['timestamp']
        
        # Assert
        # Verifiera ISO format (YYYY-MM-DDTHH:MM:SS.ffffff)
        self.assertRegex(timestamp, r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+$')
        
        print("✅ T058 PASSED: Checkpoint timestamp format är korrekt")
    
    def test_t058_checkpoint_empty_state(self):
        """
        Verifiera checkpoint creation med tom state
        """
        # Arrange - Ingen cases tillagda
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera tom state
        self.assertEqual(checkpoint_data['processed_cases'], [])
        self.assertEqual(checkpoint_data['failed_cases'], [])
        self.assertEqual(checkpoint_data['processing_state']['current_case_index'], 0)
        self.assertEqual(checkpoint_data['processing_state']['total_processed'], 0)
        self.assertEqual(checkpoint_data['processing_state']['total_failed'], 0)
        
        print("✅ T058 PASSED: Checkpoint creation med tom state fungerar korrekt")
    
    def test_t058_checkpoint_large_dataset(self):
        """
        Verifiera checkpoint creation med stort dataset
        """
        # Arrange
        # Lägg till många cases
        for i in range(100):
            case_id = f"{i:04d}"
            if i % 10 == 0:  # Var 10:e case misslyckas
                self.checkpoint_manager.add_failed_case(case_id, f"Error {i}")
            else:
                self.checkpoint_manager.add_processed_case(case_id)
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera att alla cases finns med
        self.assertEqual(len(checkpoint_data['processed_cases']), 90)
        self.assertEqual(len(checkpoint_data['failed_cases']), 10)
        self.assertEqual(checkpoint_data['processing_state']['current_case_index'], 100)
        
        # Verifiera att cases är sorterade
        self.assertEqual(checkpoint_data['processed_cases'], sorted(checkpoint_data['processed_cases']))
        self.assertEqual(checkpoint_data['failed_cases'], sorted(checkpoint_data['failed_cases']))
        
        print("✅ T058 PASSED: Checkpoint creation med stort dataset fungerar korrekt")
    
    def test_t058_checkpoint_interval_tracking(self):
        """
        Verifiera att checkpoint interval trackas korrekt
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=10  # Var 10:e case
        )
        
        # Act & Assert
        # Lägg till cases och verifiera interval tracking
        for i in range(25):
            case_id = f"{i:04d}"
            
            # Kontrollera om checkpoint ska sparas INNAN vi lägger till case
            should_save = checkpoint_manager.should_save_checkpoint(case_id)
            expected_save = (i + 1) % 10 == 0  # Var 10:e case
            
            self.assertEqual(should_save, expected_save,
                           f"Case {i+1} ska {'spara' if expected_save else 'inte spara'} checkpoint")
            
            # Lägg till case efter kontrollen
            checkpoint_manager.add_processed_case(case_id)
        
        print("✅ T058 PASSED: Checkpoint interval tracking fungerar korrekt")
    
    def test_t058_checkpoint_disabled_state(self):
        """
        Verifiera checkpoint creation när checkpoints är inaktiverade
        """
        # Arrange
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=False,
            checkpoint_interval=50
        )
        
        checkpoint_manager.add_processed_case("0001")
        
        # Act
        checkpoint_data = checkpoint_manager.create_checkpoint_data()
        
        # Assert
        # Verifiera att enable_checkpoints är False
        self.assertFalse(checkpoint_data['checkpoint_info']['enable_checkpoints'])
        
        # Verifiera att should_save_checkpoint returnerar False
        self.assertFalse(checkpoint_manager.should_save_checkpoint("0001"))
        
        print("✅ T058 PASSED: Checkpoint creation när disabled fungerar korrekt")
    
    def test_t058_checkpoint_processing_time(self):
        """
        Verifiera att processing time beräknas korrekt
        """
        # Arrange
        import time
        time.sleep(0.1)  # Vänta lite för att få en mätbar tid
        
        # Act
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        processing_time = checkpoint_data['checkpoint_info']['processing_time_seconds']
        
        # Assert
        # Verifiera att processing time är positiv och rimlig
        self.assertGreater(processing_time, 0)
        self.assertLess(processing_time, 1.0)  # Borde vara mindre än 1 sekund för detta test
        
        print("✅ T058 PASSED: Checkpoint processing time beräknas korrekt")

if __name__ == '__main__':
    unittest.main()
