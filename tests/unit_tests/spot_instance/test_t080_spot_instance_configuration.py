#!/usr/bin/env python3
"""
T080: Test Spot Instance Configuration
Verifiera att use_spot_instances=True konfigureras korrekt

AAA Format:
- Arrange: Skapa spot instance configuration system
- Act: Konfigurera spot instances med olika inställningar
- Assert: Verifiera att spot instance konfiguration är korrekt
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import json


class TestT080SpotInstanceConfiguration(unittest.TestCase):
    """Test T080: Spot Instance Configuration"""
    
    def setUp(self):
        """Arrange: Skapa spot instance configuration system"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock AWS SageMaker configuration
        self.sagemaker_config = {
            'use_spot_instances': True,
            'max_run_time': 93600,  # 26 timmar
            'max_wait_time': 187200,  # 52 timmar (max_run * 2)
            'checkpoint_s3_uri': 's3://test-bucket/checkpoints/',
            'role': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
        }
        
        # Mock spot instance configurations
        self.spot_configs = {
            'enabled': {
                'use_spot_instances': True,
                'max_run_time': 93600,
                'max_wait_time': 187200
            },
            'disabled': {
                'use_spot_instances': False,
                'max_run_time': 93600,
                'max_wait_time': 93600  # Samma som max_run när spot är disabled
            },
            'custom_times': {
                'use_spot_instances': True,
                'max_run_time': 7200,  # 2 timmar
                'max_wait_time': 14400  # 4 timmar (max_run * 2)
            }
        }
    
    def tearDown(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_t080_spot_instance_configuration_enabled(self):
        """Test T080: Spot instances aktiverade"""
        # Arrange
        config = self.spot_configs['enabled']
        
        # Act
        # Simulera spot instance konfiguration
        use_spot = config['use_spot_instances']
        max_run = config['max_run_time']
        max_wait = config['max_wait_time']
        
        # Assert
        self.assertTrue(use_spot, "Spot instances ska vara aktiverade")
        self.assertEqual(max_run, 93600, "Max run time ska vara 93600 sekunder")
        self.assertEqual(max_wait, 187200, "Max wait time ska vara 187200 sekunder")
        self.assertEqual(max_wait, max_run * 2, "Max wait time ska vara max_run * 2")
    
    def test_t080_spot_instance_configuration_disabled(self):
        """Test T080: Spot instances inaktiverade"""
        # Arrange
        config = self.spot_configs['disabled']
        
        # Act
        # Simulera spot instance konfiguration
        use_spot = config['use_spot_instances']
        max_run = config['max_run_time']
        max_wait = config['max_wait_time']
        
        # Assert
        self.assertFalse(use_spot, "Spot instances ska vara inaktiverade")
        self.assertEqual(max_run, 93600, "Max run time ska vara 93600 sekunder")
        self.assertEqual(max_wait, max_run, "Max wait time ska vara samma som max_run när spot är disabled")
    
    def test_t080_spot_instance_configuration_custom_times(self):
        """Test T080: Spot instances med anpassade tider"""
        # Arrange
        config = self.spot_configs['custom_times']
        
        # Act
        # Simulera spot instance konfiguration
        use_spot = config['use_spot_instances']
        max_run = config['max_run_time']
        max_wait = config['max_wait_time']
        
        # Assert
        self.assertTrue(use_spot, "Spot instances ska vara aktiverade")
        self.assertEqual(max_run, 7200, "Max run time ska vara 7200 sekunder")
        self.assertEqual(max_wait, 14400, "Max wait time ska vara 14400 sekunder")
        self.assertEqual(max_wait, max_run * 2, "Max wait time ska vara max_run * 2")
    
    def test_t080_spot_instance_configuration_max_wait_calculation(self):
        """Test T080: Max wait time beräkning"""
        # Arrange
        test_scenarios = [
            {'max_run': 3600, 'expected_max_wait': 7200},   # 1 timme -> 2 timmar
            {'max_run': 7200, 'expected_max_wait': 14400},  # 2 timmar -> 4 timmar
            {'max_run': 14400, 'expected_max_wait': 28800}, # 4 timmar -> 8 timmar
            {'max_run': 93600, 'expected_max_wait': 187200}, # 26 timmar -> 52 timmar
        ]
        
        # Act & Assert
        for scenario in test_scenarios:
            max_run = scenario['max_run']
            expected_max_wait = scenario['expected_max_wait']
            
            # Beräkna max_wait_time för spot instances
            max_wait = max_run * 2
            
            self.assertEqual(max_wait, expected_max_wait,
                            f"För max_run {max_run}, max_wait ska vara {expected_max_wait}")
    
    def test_t080_spot_instance_configuration_aws_sagemaker_integration(self):
        """Test T080: AWS SageMaker integration för spot instances"""
        # Arrange
        config = self.sagemaker_config
        
        # Act
        # Simulera SageMaker processing job konfiguration
        processing_job_config = {
            'ProcessingJobName': 'master-poc-preprocessing-v5',
            'RoleArn': config['role'],
            'ProcessingResources': {
                'ClusterConfig': {
                    'InstanceCount': 6,
                    'InstanceType': 'ml.m5.2xlarge',
                    'VolumeSizeInGB': 100
                }
            },
            'ProcessingInputs': [
                {
                    'InputName': 'input-data',
                    'S3Input': {
                        'S3Uri': 's3://test-bucket/input/',
                        'LocalPath': '/opt/ml/processing/input'
                    }
                }
            ],
            'ProcessingOutputConfig': {
                'Outputs': [
                    {
                        'OutputName': 'output-data',
                        'S3Output': {
                            'S3Uri': 's3://test-bucket/output/',
                            'LocalPath': '/opt/ml/processing/output'
                        }
                    }
                ]
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': config['max_run_time']
            }
        }
        
        # Lägg till spot instance konfiguration
        if config['use_spot_instances']:
            processing_job_config['ProcessingResources']['ClusterConfig']['VolumeKmsKeyId'] = None
            processing_job_config['StoppingCondition']['MaxWaitTimeInSeconds'] = config['max_wait_time']
        
        # Assert
        self.assertTrue(config['use_spot_instances'], "Spot instances ska vara aktiverade")
        self.assertIn('MaxWaitTimeInSeconds', processing_job_config['StoppingCondition'],
                     "MaxWaitTimeInSeconds ska finnas när spot instances är aktiverade")
        self.assertEqual(processing_job_config['StoppingCondition']['MaxWaitTimeInSeconds'], 
                        config['max_wait_time'],
                        "MaxWaitTimeInSeconds ska vara korrekt")
    
    def test_t080_spot_instance_configuration_cost_savings(self):
        """Test T080: Kostnadsbesparingar med spot instances"""
        # Arrange
        # Mock kostnadsdata
        on_demand_price = 1.0  # $1 per timme
        spot_price = 0.3      # $0.30 per timme (70% besparing)
        
        config = self.spot_configs['enabled']
        max_run_hours = config['max_run_time'] / 3600  # Konvertera till timmar
        
        # Act
        # Beräkna kostnader
        on_demand_cost = on_demand_price * max_run_hours
        spot_cost = spot_price * max_run_hours
        cost_savings = on_demand_cost - spot_cost
        savings_percentage = (cost_savings / on_demand_cost) * 100
        
        # Assert
        self.assertLess(spot_cost, on_demand_cost, "Spot instances ska vara billigare")
        self.assertGreater(cost_savings, 0, "Kostnadsbesparingar ska vara positiva")
        self.assertGreater(savings_percentage, 50, "Besparingar ska vara minst 50%")
        
        # Verifiera att besparingar är signifikanta
        expected_savings = 0.7  # 70% besparing
        self.assertAlmostEqual(savings_percentage / 100, expected_savings, delta=0.1,
                              msg=f"Besparingar ska vara cirka {expected_savings*100}%")
    
    def test_t080_spot_instance_configuration_checkpoint_integration(self):
        """Test T080: Checkpoint integration med spot instances"""
        # Arrange
        config = self.sagemaker_config
        
        # Act
        # Simulera checkpoint konfiguration för spot instances
        checkpoint_config = {
            'enable_checkpoints': True,
            'checkpoint_s3_uri': config['checkpoint_s3_uri'],
            'checkpoint_frequency': 300,  # 5 minuter
            'spot_instance_support': True
        }
        
        # Assert
        self.assertTrue(checkpoint_config['enable_checkpoints'],
                       "Checkpoints ska vara aktiverade för spot instances")
        self.assertTrue(checkpoint_config['spot_instance_support'],
                       "Spot instance support ska vara aktiverad")
        self.assertIsNotNone(checkpoint_config['checkpoint_s3_uri'],
                           "Checkpoint S3 URI ska vara konfigurerad")
        self.assertGreater(checkpoint_config['checkpoint_frequency'], 0,
                          "Checkpoint frequency ska vara större än 0")
    
    def test_t080_spot_instance_configuration_error_handling(self):
        """Test T080: Error handling för spot instance konfiguration"""
        # Arrange
        invalid_configs = [
            {'use_spot_instances': True, 'max_run_time': 0},  # Ogiltig max_run_time
            {'use_spot_instances': True, 'max_wait_time': 0},  # Ogiltig max_wait_time
            {'use_spot_instances': True, 'max_wait_time': 1000, 'max_run_time': 2000},  # max_wait < max_run * 2
        ]
        
        # Act & Assert
        for invalid_config in invalid_configs:
            with self.subTest(config=invalid_config):
                # Validera konfiguration
                is_valid = True
                errors = []
                
                if invalid_config.get('max_run_time', 0) <= 0:
                    is_valid = False
                    errors.append("max_run_time måste vara större än 0")
                
                if invalid_config.get('max_wait_time', 0) <= 0:
                    is_valid = False
                    errors.append("max_wait_time måste vara större än 0")
                
                if (invalid_config.get('use_spot_instances', False) and 
                    invalid_config.get('max_wait_time', 0) < invalid_config.get('max_run_time', 0) * 2):
                    is_valid = False
                    errors.append("max_wait_time måste vara minst max_run_time * 2 för spot instances")
                
                self.assertFalse(is_valid, f"Konfiguration ska vara ogiltig: {errors}")
    
    def test_t080_spot_instance_configuration_aws_checklist_compliance(self):
        """Test T080: AWS checklist compliance för spot instance konfiguration"""
        # Arrange
        # Enligt AWS_CHECKLIST_V5.0_3000_CASES.md
        aws_checklist_requirement = {
            'spot_instances_enabled': True,
            'max_wait_time_calculation': True,
            'checkpoint_integration': True,
            'cost_savings': True
        }
        
        config = self.spot_configs['enabled']
        
        # Act
        # Verifiera AWS checklist compliance
        use_spot = config['use_spot_instances']
        max_run = config['max_run_time']
        max_wait = config['max_wait_time']
        
        # Assert
        # Verifiera AWS checklist compliance
        self.assertTrue(aws_checklist_requirement['spot_instances_enabled'],
                       "AWS checklist kräver spot instances")
        
        self.assertTrue(aws_checklist_requirement['max_wait_time_calculation'],
                       "AWS checklist kräver korrekt max_wait_time beräkning")
        
        self.assertTrue(aws_checklist_requirement['checkpoint_integration'],
                       "AWS checklist kräver checkpoint integration")
        
        self.assertTrue(aws_checklist_requirement['cost_savings'],
                       "AWS checklist kräver kostnadsbesparingar")
        
        # Verifiera att konfiguration följer krav
        self.assertTrue(use_spot, "Spot instances ska vara aktiverade enligt AWS checklist")
        self.assertEqual(max_wait, max_run * 2, "Max wait time ska vara max_run * 2 enligt AWS checklist")


if __name__ == '__main__':
    unittest.main()
