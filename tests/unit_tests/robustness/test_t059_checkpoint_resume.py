#!/usr/bin/env python3
"""
T059: Test Checkpoint Resume
Verifiera att processing kan återupptas från checkpoint
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

class TestT059CheckpointResume(unittest.TestCase):
    """T059: Test Checkpoint Resume"""
    
    def setUp(self):
        """Setup för varje test."""
        self.checkpoint_path = "s3://test-bucket/checkpoints/test"
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
    
    def test_t059_checkpoint_resume_basic(self):
        """
        T059: Test Checkpoint Resume
        Verifiera att processing kan återupptas från checkpoint
        """
        # Arrange
        # Simulera att vi har processat några cases
        self.checkpoint_manager.add_processed_case("0001")
        self.checkpoint_manager.add_processed_case("0002")
        self.checkpoint_manager.add_failed_case("0003", "Test error")
        
        # Skapa checkpoint data
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data()
        
        # Skapa ny checkpoint manager för resume
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response för checkpoint load
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list, \
             patch.object(new_checkpoint_manager.s3_client, 'get_object') as mock_get:
            
            # Mock list_objects_v2 response
            mock_list.return_value = {
                'Contents': [
                    {
                        'Key': 'checkpoints/test/checkpoint_1234567890.json',
                        'LastModified': '2024-01-01T12:00:00Z'
                    }
                ]
            }
            
            # Mock get_object response
            mock_get.return_value = {
                'Body': Mock(read=lambda: json.dumps(checkpoint_data).encode('utf-8'))
            }
            
            # Act
            resume_success = new_checkpoint_manager.load_checkpoint()
            
            # Assert
            self.assertTrue(resume_success, "Checkpoint resume ska lyckas")
            self.assertEqual(len(new_checkpoint_manager.processed_cases), 2)
            self.assertEqual(len(new_checkpoint_manager.failed_cases), 1)
            self.assertEqual(new_checkpoint_manager.current_case_index, 3)
            self.assertIn("0001", new_checkpoint_manager.processed_cases)
            self.assertIn("0002", new_checkpoint_manager.processed_cases)
            self.assertIn("0003", new_checkpoint_manager.failed_cases)
            
        print("✅ T059 PASSED: Basic checkpoint resume fungerar korrekt")
    
    def test_t059_checkpoint_resume_no_existing_checkpoint(self):
        """
        Verifiera checkpoint resume när ingen checkpoint finns
        """
        # Arrange
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response för tom bucket
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list:
            mock_list.return_value = {}  # Ingen Contents key
            
            # Act
            resume_success = new_checkpoint_manager.load_checkpoint()
            
            # Assert
            self.assertTrue(resume_success, "Resume ska lyckas även utan checkpoint")
            self.assertEqual(len(new_checkpoint_manager.processed_cases), 0)
            self.assertEqual(len(new_checkpoint_manager.failed_cases), 0)
            self.assertEqual(new_checkpoint_manager.current_case_index, 0)
            
        print("✅ T059 PASSED: Checkpoint resume utan befintlig checkpoint fungerar korrekt")
    
    def test_t059_checkpoint_resume_get_remaining_cases(self):
        """
        Verifiera att get_remaining_cases fungerar efter resume
        """
        # Arrange
        all_cases = [f"{i:04d}" for i in range(1, 11)]  # 0001-0010
        
        # Simulera checkpoint med vissa cases processade
        checkpoint_data = {
            'processed_cases': ['0001', '0002', '0003'],
            'failed_cases': ['0004'],
            'processing_state': {'current_case_index': 4}
        }
        
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list, \
             patch.object(new_checkpoint_manager.s3_client, 'get_object') as mock_get:
            
            mock_list.return_value = {
                'Contents': [{'Key': 'checkpoints/test/checkpoint_1234567890.json', 'LastModified': '2024-01-01T12:00:00Z'}]
            }
            
            mock_get.return_value = {
                'Body': Mock(read=lambda: json.dumps(checkpoint_data).encode('utf-8'))
            }
            
            # Act
            new_checkpoint_manager.load_checkpoint()
            remaining_cases = new_checkpoint_manager.get_remaining_cases(all_cases)
            
            # Assert
            expected_remaining = ['0005', '0006', '0007', '0008', '0009', '0010']
            self.assertEqual(remaining_cases, expected_remaining,
                           f"Remaining cases ska vara {expected_remaining}")
            
        print("✅ T059 PASSED: Get remaining cases efter resume fungerar korrekt")
    
    def test_t059_checkpoint_resume_latest_checkpoint(self):
        """
        Verifiera att senaste checkpoint laddas när flera finns
        """
        # Arrange
        checkpoint_data_old = {
            'processed_cases': ['0001'],
            'failed_cases': [],
            'processing_state': {'current_case_index': 1}
        }
        
        checkpoint_data_new = {
            'processed_cases': ['0001', '0002', '0003'],
            'failed_cases': ['0004'],
            'processing_state': {'current_case_index': 4}
        }
        
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response med flera checkpoints
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list, \
             patch.object(new_checkpoint_manager.s3_client, 'get_object') as mock_get:
            
            mock_list.return_value = {
                'Contents': [
                    {
                        'Key': 'checkpoints/test/checkpoint_1234567890.json',
                        'LastModified': '2024-01-01T10:00:00Z'  # Äldre
                    },
                    {
                        'Key': 'checkpoints/test/checkpoint_1234567891.json',
                        'LastModified': '2024-01-01T12:00:00Z'  # Nyare
                    }
                ]
            }
            
            # Mock get_object för att returnera rätt data baserat på key
            def mock_get_object(Bucket, Key):
                if 'checkpoint_1234567891' in Key:  # Nyare checkpoint
                    return {
                        'Body': Mock(read=lambda: json.dumps(checkpoint_data_new).encode('utf-8'))
                    }
                else:  # Äldre checkpoint
                    return {
                        'Body': Mock(read=lambda: json.dumps(checkpoint_data_old).encode('utf-8'))
                    }
            
            mock_get.side_effect = mock_get_object
            
            # Act
            resume_success = new_checkpoint_manager.load_checkpoint()
            
            # Assert
            self.assertTrue(resume_success, "Resume ska lyckas")
            self.assertEqual(len(new_checkpoint_manager.processed_cases), 3)  # Från nyare checkpoint
            self.assertEqual(len(new_checkpoint_manager.failed_cases), 1)
            self.assertEqual(new_checkpoint_manager.current_case_index, 4)
            
        print("✅ T059 PASSED: Senaste checkpoint laddas korrekt")
    
    def test_t059_checkpoint_resume_error_handling(self):
        """
        Verifiera felhantering vid checkpoint resume
        """
        # Arrange
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 error
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list:
            mock_list.side_effect = Exception("S3 connection error")
            
            # Act
            resume_success = new_checkpoint_manager.load_checkpoint()
            
            # Assert
            self.assertFalse(resume_success, "Resume ska misslyckas vid S3 fel")
            
        print("✅ T059 PASSED: Checkpoint resume felhantering fungerar korrekt")
    
    def test_t059_checkpoint_resume_disabled(self):
        """
        Verifiera checkpoint resume när checkpoints är inaktiverade
        """
        # Arrange
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=False,
            checkpoint_interval=50
        )
        
        # Act
        resume_success = new_checkpoint_manager.load_checkpoint()
        
        # Assert
        self.assertTrue(resume_success, "Resume ska lyckas även när disabled")
        self.assertEqual(len(new_checkpoint_manager.processed_cases), 0)
        self.assertEqual(len(new_checkpoint_manager.failed_cases), 0)
        
        print("✅ T059 PASSED: Checkpoint resume när disabled fungerar korrekt")
    
    def test_t059_checkpoint_resume_processing_stats(self):
        """
        Verifiera att processing stats fungerar efter resume
        """
        # Arrange
        checkpoint_data = {
            'processed_cases': ['0001', '0002'],
            'failed_cases': ['0003'],
            'processing_state': {'current_case_index': 3}
        }
        
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list, \
             patch.object(new_checkpoint_manager.s3_client, 'get_object') as mock_get:
            
            mock_list.return_value = {
                'Contents': [{'Key': 'checkpoints/test/checkpoint_1234567890.json', 'LastModified': '2024-01-01T12:00:00Z'}]
            }
            
            mock_get.return_value = {
                'Body': Mock(read=lambda: json.dumps(checkpoint_data).encode('utf-8'))
            }
            
            # Act
            new_checkpoint_manager.load_checkpoint()
            stats = new_checkpoint_manager.get_processing_stats()
            
            # Assert
            self.assertEqual(stats['total_cases'], 3)
            self.assertEqual(stats['processed_cases'], 2)
            self.assertEqual(stats['failed_cases'], 1)
            self.assertEqual(stats['checkpoint_interval'], 50)
            self.assertTrue(stats['enable_checkpoints'])
            
        print("✅ T059 PASSED: Processing stats efter resume fungerar korrekt")
    
    def test_t059_checkpoint_resume_continue_processing(self):
        """
        Verifiera att processing kan fortsätta efter resume
        """
        # Arrange
        checkpoint_data = {
            'processed_cases': ['0001', '0002'],
            'failed_cases': ['0003'],
            'processing_state': {'current_case_index': 3}
        }
        
        new_checkpoint_manager = create_checkpoint_manager(
            checkpoint_path=self.checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=50
        )
        
        # Mock S3 response
        with patch.object(new_checkpoint_manager.s3_client, 'list_objects_v2') as mock_list, \
             patch.object(new_checkpoint_manager.s3_client, 'get_object') as mock_get:
            
            mock_list.return_value = {
                'Contents': [{'Key': 'checkpoints/test/checkpoint_1234567890.json', 'LastModified': '2024-01-01T12:00:00Z'}]
            }
            
            mock_get.return_value = {
                'Body': Mock(read=lambda: json.dumps(checkpoint_data).encode('utf-8'))
            }
            
            # Act
            new_checkpoint_manager.load_checkpoint()
            
            # Fortsätt processing
            new_checkpoint_manager.add_processed_case("0004")
            new_checkpoint_manager.add_processed_case("0005")
            
            # Assert
            self.assertEqual(len(new_checkpoint_manager.processed_cases), 4)  # 2 från checkpoint + 2 nya
            self.assertEqual(len(new_checkpoint_manager.failed_cases), 1)
            self.assertEqual(new_checkpoint_manager.current_case_index, 5)
            
        print("✅ T059 PASSED: Processing kan fortsätta efter resume")

if __name__ == '__main__':
    unittest.main()
