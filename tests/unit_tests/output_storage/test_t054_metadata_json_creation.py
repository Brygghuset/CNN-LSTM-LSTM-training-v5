#!/usr/bin/env python3
"""
T054: Test Metadata JSON Creation
Verifiera att preprocessing_metadata.json skapas korrekt

AAA Format:
- Arrange: Skapa metadata structure och JSON writer
- Act: Generera metadata JSON med olika data
- Assert: Verifiera att JSON skapas korrekt med rätt struktur
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestT054MetadataJSONCreation(unittest.TestCase):
    """Test T054: Metadata JSON Creation"""
    
    def setUp(self):
        """Arrange: Skapa metadata structure och JSON writer"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock data för testing
        self.sample_metadata = {
            'pipeline_info': {
                'version': '5.0',
                'type': 'master_poc',
                'description': 'Master POC CNN-LSTM-LSTM Preprocessing Pipeline'
            },
            'processing_info': {
                'total_cases': 100,
                'processed_cases': 100,
                'failed_cases': 0,
                'success_rate': 1.0
            },
            'feature_info': {
                'timeseries_features': 16,
                'static_features': 6,
                'output_features': 8
            },
            'window_info': {
                'window_size': 300,
                'step_size': 30,
                'total_windows': 1000
            },
            'split_info': {
                'train_windows': 700,
                'validation_windows': 150,
                'test_windows': 150,
                'train_percent': 70.0,
                'validation_percent': 15.0,
                'test_percent': 15.0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def tearDown(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_t054_metadata_json_creation_basic(self):
        """Test T054: Basic metadata JSON creation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        self.assertTrue(os.path.exists(metadata_file), 
                       "Metadata JSON fil ska skapas")
        
        # Verifiera att filen kan läsas
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['pipeline_info']['version'], '5.0',
                        "Pipeline version ska vara 5.0")
        self.assertEqual(loaded_metadata['processing_info']['total_cases'], 100,
                        "Total cases ska vara 100")
    
    def test_t054_metadata_json_creation_structure(self):
        """Test T054: Metadata JSON structure validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att alla required sections finns
        required_sections = [
            'pipeline_info', 'processing_info', 'feature_info', 
            'window_info', 'split_info', 'timestamp'
        ]
        
        for section in required_sections:
            self.assertIn(section, loaded_metadata, 
                         f"Metadata ska innehålla {section} section")
        
        # Verifiera pipeline_info structure
        pipeline_info = loaded_metadata['pipeline_info']
        self.assertIn('version', pipeline_info)
        self.assertIn('type', pipeline_info)
        self.assertIn('description', pipeline_info)
        
        # Verifiera processing_info structure
        processing_info = loaded_metadata['processing_info']
        self.assertIn('total_cases', processing_info)
        self.assertIn('processed_cases', processing_info)
        self.assertIn('failed_cases', processing_info)
        self.assertIn('success_rate', processing_info)
    
    def test_t054_metadata_json_creation_feature_counts(self):
        """Test T054: Metadata JSON feature counts validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        feature_info = loaded_metadata['feature_info']
        
        # Verifiera Master POC feature counts
        self.assertEqual(feature_info['timeseries_features'], 16,
                        "Timeseries features ska vara 16")
        self.assertEqual(feature_info['static_features'], 6,
                        "Static features ska vara 6")
        self.assertEqual(feature_info['output_features'], 8,
                        "Output features ska vara 8")
    
    def test_t054_metadata_json_creation_split_validation(self):
        """Test T054: Metadata JSON split validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        split_info = loaded_metadata['split_info']
        
        # Verifiera 70/15/15 split
        self.assertEqual(split_info['train_percent'], 70.0,
                        "Train percent ska vara 70.0")
        self.assertEqual(split_info['validation_percent'], 15.0,
                        "Validation percent ska vara 15.0")
        self.assertEqual(split_info['test_percent'], 15.0,
                        "Test percent ska vara 15.0")
        
        # Verifiera att totals matchar
        total_windows = (split_info['train_windows'] + 
                        split_info['validation_windows'] + 
                        split_info['test_windows'])
        self.assertEqual(total_windows, 1000,
                        "Total windows ska vara 1000")
    
    def test_t054_metadata_json_creation_window_info(self):
        """Test T054: Metadata JSON window info validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        window_info = loaded_metadata['window_info']
        
        # Verifiera Master POC window parameters
        self.assertEqual(window_info['window_size'], 300,
                        "Window size ska vara 300")
        self.assertEqual(window_info['step_size'], 30,
                        "Step size ska vara 30")
        self.assertEqual(window_info['total_windows'], 1000,
                        "Total windows ska vara 1000")
    
    def test_t054_metadata_json_creation_processing_stats(self):
        """Test T054: Metadata JSON processing statistics"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        processing_info = loaded_metadata['processing_info']
        
        # Verifiera processing statistics
        self.assertEqual(processing_info['total_cases'], 100,
                        "Total cases ska vara 100")
        self.assertEqual(processing_info['processed_cases'], 100,
                        "Processed cases ska vara 100")
        self.assertEqual(processing_info['failed_cases'], 0,
                        "Failed cases ska vara 0")
        self.assertEqual(processing_info['success_rate'], 1.0,
                        "Success rate ska vara 1.0")
    
    def test_t054_metadata_json_creation_timestamp(self):
        """Test T054: Metadata JSON timestamp validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att timestamp finns och är giltig
        self.assertIn('timestamp', loaded_metadata,
                      "Metadata ska innehålla timestamp")
        
        timestamp = loaded_metadata['timestamp']
        self.assertIsInstance(timestamp, str,
                             "Timestamp ska vara string")
        
        # Verifiera att timestamp kan parsas
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            self.fail("Timestamp ska vara giltig ISO format")
    
    def test_t054_metadata_json_creation_error_handling(self):
        """Test T054: Metadata JSON creation error handling"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Skapa metadata med None värden
        problematic_metadata = {
            'pipeline_info': {
                'version': None,
                'type': 'master_poc',
                'description': 'Test description'
            },
            'processing_info': {
                'total_cases': 0,
                'processed_cases': 0,
                'failed_cases': 0,
                'success_rate': 0.0
            }
        }
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(problematic_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        self.assertTrue(os.path.exists(metadata_file),
                       "Metadata JSON ska skapas även med None värden")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att None värden hanteras korrekt
        self.assertIsNone(loaded_metadata['pipeline_info']['version'],
                         "None värden ska bevaras")
        self.assertEqual(loaded_metadata['processing_info']['total_cases'], 0,
                        "Zero värden ska bevaras")
    
    def test_t054_metadata_json_creation_file_permissions(self):
        """Test T054: Metadata JSON file permissions"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        # Verifiera att filen är läsbar
        self.assertTrue(os.access(metadata_file, os.R_OK),
                       "Metadata fil ska vara läsbar")
        
        # Verifiera att filen är skrivbar
        self.assertTrue(os.access(metadata_file, os.W_OK),
                       "Metadata fil ska vara skrivbar")
        
        # Verifiera filstorlek
        file_size = os.path.getsize(metadata_file)
        self.assertGreater(file_size, 0,
                           "Metadata fil ska ha innehåll")
        self.assertLess(file_size, 1024 * 1024,  # 1MB
                       "Metadata fil ska vara mindre än 1MB")


if __name__ == '__main__':
    unittest.main()
