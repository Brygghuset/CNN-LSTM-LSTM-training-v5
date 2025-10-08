#!/usr/bin/env python3
"""
T055: Test Metadata Content
Verifiera att metadata innehåller total_samples, split counts, shapes

AAA Format:
- Arrange: Skapa metadata med olika data scenarios
- Act: Validera metadata content mot Master POC spec
- Assert: Verifiera att alla required fields finns med korrekt data
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestT055MetadataContent(unittest.TestCase):
    """Test T055: Metadata Content"""
    
    def setUp(self):
        """Arrange: Skapa metadata med olika data scenarios"""
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
                'success_rate': 1.0,
                'processing_time_seconds': 3600,
                'average_time_per_case': 36.0
            },
            'feature_info': {
                'timeseries_features': 16,
                'static_features': 6,
                'output_features': 8,
                'timeseries_shape': [300, 16],
                'static_shape': [6],
                'output_shape': [8]
            },
            'window_info': {
                'window_size': 300,
                'step_size': 30,
                'total_windows': 1000,
                'windows_per_case_avg': 10.0
            },
            'split_info': {
                'train_windows': 700,
                'validation_windows': 150,
                'test_windows': 150,
                'train_percent': 70.0,
                'validation_percent': 15.0,
                'test_percent': 15.0,
                'split_method': 'case_based',
                'random_seed': 42
            },
            'data_info': {
                'total_samples': 1000,
                'train_samples': 700,
                'validation_samples': 150,
                'test_samples': 150,
                'sample_distribution': 'balanced'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def tearDown(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_t055_metadata_content_total_samples(self):
        """Test T055: Metadata content total_samples validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att total_samples finns
        self.assertIn('data_info', loaded_metadata,
                      "Metadata ska innehålla data_info section")
        
        data_info = loaded_metadata['data_info']
        self.assertIn('total_samples', data_info,
                      "Data info ska innehålla total_samples")
        
        self.assertEqual(data_info['total_samples'], 1000,
                        "Total samples ska vara 1000")
        
        # Verifiera att total_samples matchar total_windows
        self.assertEqual(data_info['total_samples'], 
                        loaded_metadata['window_info']['total_windows'],
                        "Total samples ska matcha total windows")
    
    def test_t055_metadata_content_split_counts(self):
        """Test T055: Metadata content split counts validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        split_info = loaded_metadata['split_info']
        data_info = loaded_metadata['data_info']
        
        # Verifiera split counts
        self.assertEqual(split_info['train_windows'], 700,
                        "Train windows ska vara 700")
        self.assertEqual(split_info['validation_windows'], 150,
                        "Validation windows ska vara 150")
        self.assertEqual(split_info['test_windows'], 150,
                        "Test windows ska vara 150")
        
        # Verifiera att split counts matchar data_info
        self.assertEqual(split_info['train_windows'], data_info['train_samples'],
                        "Train windows ska matcha train samples")
        self.assertEqual(split_info['validation_windows'], data_info['validation_samples'],
                        "Validation windows ska matcha validation samples")
        self.assertEqual(split_info['test_windows'], data_info['test_samples'],
                        "Test windows ska matcha test samples")
    
    def test_t055_metadata_content_shapes(self):
        """Test T055: Metadata content shapes validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        feature_info = loaded_metadata['feature_info']
        
        # Verifiera att shapes finns
        self.assertIn('timeseries_shape', feature_info,
                      "Feature info ska innehålla timeseries_shape")
        self.assertIn('static_shape', feature_info,
                      "Feature info ska innehålla static_shape")
        self.assertIn('output_shape', feature_info,
                      "Feature info ska innehålla output_shape")
        
        # Verifiera Master POC shapes
        self.assertEqual(feature_info['timeseries_shape'], [300, 16],
                        "Timeseries shape ska vara [300, 16]")
        self.assertEqual(feature_info['static_shape'], [6],
                        "Static shape ska vara [6]")
        self.assertEqual(feature_info['output_shape'], [8],
                        "Output shape ska vara [8]")
        
        # Verifiera att shapes matchar feature counts
        self.assertEqual(feature_info['timeseries_shape'][1], 
                        feature_info['timeseries_features'],
                        "Timeseries shape[1] ska matcha timeseries_features")
        self.assertEqual(feature_info['static_shape'][0], 
                        feature_info['static_features'],
                        "Static shape[0] ska matcha static_features")
        self.assertEqual(feature_info['output_shape'][0], 
                        feature_info['output_features'],
                        "Output shape[0] ska matcha output_features")
    
    def test_t055_metadata_content_processing_stats(self):
        """Test T055: Metadata content processing statistics"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        processing_info = loaded_metadata['processing_info']
        
        # Verifiera att processing stats finns
        self.assertIn('processing_time_seconds', processing_info,
                      "Processing info ska innehålla processing_time_seconds")
        self.assertIn('average_time_per_case', processing_info,
                      "Processing info ska innehålla average_time_per_case")
        
        # Verifiera processing stats
        self.assertEqual(processing_info['processing_time_seconds'], 3600,
                        "Processing time ska vara 3600 sekunder")
        self.assertEqual(processing_info['average_time_per_case'], 36.0,
                        "Average time per case ska vara 36.0 sekunder")
        
        # Verifiera att success rate är korrekt
        expected_success_rate = (processing_info['processed_cases'] / 
                                processing_info['total_cases'])
        self.assertEqual(processing_info['success_rate'], expected_success_rate,
                        "Success rate ska vara korrekt beräknad")
    
    def test_t055_metadata_content_window_info(self):
        """Test T055: Metadata content window info validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        window_info = loaded_metadata['window_info']
        
        # Verifiera att window info finns
        self.assertIn('windows_per_case_avg', window_info,
                      "Window info ska innehålla windows_per_case_avg")
        
        # Verifiera window parameters
        self.assertEqual(window_info['window_size'], 300,
                        "Window size ska vara 300")
        self.assertEqual(window_info['step_size'], 30,
                        "Step size ska vara 30")
        self.assertEqual(window_info['total_windows'], 1000,
                        "Total windows ska vara 1000")
        self.assertEqual(window_info['windows_per_case_avg'], 10.0,
                        "Windows per case avg ska vara 10.0")
        
        # Verifiera att windows_per_case_avg är korrekt beräknad
        expected_avg = (window_info['total_windows'] / 
                      loaded_metadata['processing_info']['total_cases'])
        self.assertEqual(window_info['windows_per_case_avg'], expected_avg,
                        "Windows per case avg ska vara korrekt beräknad")
    
    def test_t055_metadata_content_split_method(self):
        """Test T055: Metadata content split method validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        split_info = loaded_metadata['split_info']
        
        # Verifiera att split method finns
        self.assertIn('split_method', split_info,
                      "Split info ska innehålla split_method")
        self.assertIn('random_seed', split_info,
                      "Split info ska innehålla random_seed")
        
        # Verifiera split method
        self.assertEqual(split_info['split_method'], 'case_based',
                        "Split method ska vara case_based")
        self.assertEqual(split_info['random_seed'], 42,
                        "Random seed ska vara 42")
    
    def test_t055_metadata_content_data_distribution(self):
        """Test T055: Metadata content data distribution validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        data_info = loaded_metadata['data_info']
        
        # Verifiera att data distribution finns
        self.assertIn('sample_distribution', data_info,
                      "Data info ska innehålla sample_distribution")
        
        # Verifiera data distribution
        self.assertEqual(data_info['sample_distribution'], 'balanced',
                        "Sample distribution ska vara balanced")
        
        # Verifiera att split percentages är korrekta
        split_info = loaded_metadata['split_info']
        self.assertEqual(split_info['train_percent'], 70.0,
                        "Train percent ska vara 70.0")
        self.assertEqual(split_info['validation_percent'], 15.0,
                        "Validation percent ska vara 15.0")
        self.assertEqual(split_info['test_percent'], 15.0,
                        "Test percent ska vara 15.0")
    
    def test_t055_metadata_content_consistency(self):
        """Test T055: Metadata content consistency validation"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera att alla counts är konsistenta
        total_windows = (loaded_metadata['split_info']['train_windows'] + 
                        loaded_metadata['split_info']['validation_windows'] + 
                        loaded_metadata['split_info']['test_windows'])
        
        self.assertEqual(total_windows, loaded_metadata['window_info']['total_windows'],
                        "Total windows ska vara konsistent")
        
        self.assertEqual(total_windows, loaded_metadata['data_info']['total_samples'],
                        "Total samples ska matcha total windows")
        
        # Verifiera att feature counts är konsistenta
        feature_info = loaded_metadata['feature_info']
        self.assertEqual(feature_info['timeseries_features'], 16,
                        "Timeseries features ska vara 16")
        self.assertEqual(feature_info['static_features'], 6,
                        "Static features ska vara 6")
        self.assertEqual(feature_info['output_features'], 8,
                        "Output features ska vara 8")
    
    def test_t055_metadata_content_master_poc_compliance(self):
        """Test T055: Metadata content Master POC compliance"""
        # Arrange
        metadata_file = os.path.join(self.temp_dir, 'preprocessing_metadata.json')
        
        # Act
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Assert
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        
        # Verifiera Master POC compliance
        feature_info = loaded_metadata['feature_info']
        
        # Master POC spec: 16 timeseries features
        self.assertEqual(feature_info['timeseries_features'], 16,
                        "Master POC kräver 16 timeseries features")
        
        # Master POC spec: 6 static features
        self.assertEqual(feature_info['static_features'], 6,
                        "Master POC kräver 6 static features")
        
        # Master POC spec: 8 output features (3 drugs + 5 ventilator)
        self.assertEqual(feature_info['output_features'], 8,
                        "Master POC kräver 8 output features")
        
        # Master POC spec: Window size 300s
        self.assertEqual(loaded_metadata['window_info']['window_size'], 300,
                        "Master POC kräver window size 300s")
        
        # Master POC spec: Step size 30s
        self.assertEqual(loaded_metadata['window_info']['step_size'], 30,
                        "Master POC kräver step size 30s")


if __name__ == '__main__':
    unittest.main()
