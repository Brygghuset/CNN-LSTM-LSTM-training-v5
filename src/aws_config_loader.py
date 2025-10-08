#!/usr/bin/env python3
"""
AWS Configuration Loader v5.0
=============================

Laddar AWS-konfiguration från .env fil för Master POC preprocessing pipeline.
Baserat på AWS_develop_instruction_V5.0.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class AWSConfig:
    """AWS Configuration Manager för Master POC v5.0"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initiera AWS configuration.
        
        Args:
            env_file: Sökväg till .env fil (default: aws_config.env)
        """
        self.env_file = env_file or "aws_config.env"
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Ladda konfiguration från .env fil."""
        try:
            # Kontrollera om .env fil finns
            if not os.path.exists(self.env_file):
                logger.warning(f"⚠️ {self.env_file} inte hittad, använder system environment variables")
                self._load_from_environment()
                return
            
            # Ladda från .env fil
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Hoppa över kommentarer och tomma rader
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parsa key=value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Ta bort quotes om de finns
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        self.config[key] = value
            
            logger.info(f"✅ AWS konfiguration laddad från {self.env_file}")
            
        except Exception as e:
            logger.error(f"❌ Fel vid laddning av {self.env_file}: {e}")
            self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Ladda konfiguration från system environment variables."""
        # Lista över alla konfigurationsnycklar vi behöver
        required_keys = [
            'AWS_ACCOUNT_ID', 'AWS_REGION', 'AWS_SAGEMAKER_ROLE_ARN',
            'S3_PRIMARY_BUCKET', 'S3_INPUT_PATH', 'S3_OUTPUT_PATH',
            'S3_CHECKPOINT_PATH', 'SAGEMAKER_INSTANCE_TYPE',
            'SAGEMAKER_INSTANCE_COUNT', 'PROCESSING_CASE_RANGE'
        ]
        
        for key in required_keys:
            value = os.getenv(key)
            if value:
                self.config[key] = value
        
        logger.info("✅ AWS konfiguration laddad från environment variables")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Hämta konfigurationsvärde.
        
        Args:
            key: Konfigurationsnyckel
            default: Defaultvärde om nyckeln inte finns
            
        Returns:
            Konfigurationsvärde eller default
        """
        return self.config.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Hämta konfigurationsvärde som integer."""
        try:
            return int(self.get(key, str(default)))
        except ValueError:
            logger.warning(f"⚠️ Kunde inte konvertera {key} till integer, använder default {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Hämta konfigurationsvärde som boolean."""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_list(self, key: str, separator: str = ',', default: Optional[list] = None) -> list:
        """Hämta konfigurationsvärde som lista."""
        value = self.get(key)
        if not value:
            return default or []
        
        return [item.strip() for item in value.split(separator)]
    
    def get_s3_path(self, key: str) -> str:
        """Hämta S3 path och verifiera format."""
        path = self.get(key, '')
        if not path.startswith('s3://'):
            logger.warning(f"⚠️ S3 path {key} saknar s3:// prefix: {path}")
        return path
    
    def get_bucket_name(self, s3_path: str) -> str:
        """Extrahera bucket namn från S3 path."""
        if s3_path.startswith('s3://'):
            return s3_path[5:].split('/')[0]
        return s3_path.split('/')[0]
    
    def get_s3_key(self, s3_path: str) -> str:
        """Extrahera S3 key från S3 path."""
        if s3_path.startswith('s3://'):
            parts = s3_path[5:].split('/', 1)
            return parts[1] if len(parts) > 1 else ''
        return s3_path.split('/', 1)[1] if '/' in s3_path else ''
    
    def validate_config(self) -> bool:
        """
        Validera att alla kritiska konfigurationer finns.
        
        Returns:
            True om konfigurationen är giltig
        """
        critical_keys = [
            'AWS_ACCOUNT_ID',
            'AWS_REGION', 
            'S3_PRIMARY_BUCKET',
            'S3_INPUT_PATH',
            'S3_OUTPUT_PATH',
            'S3_CHECKPOINT_PATH',
            'SAGEMAKER_ROLE_ARN'
        ]
        
        missing_keys = []
        for key in critical_keys:
            if not self.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"❌ Saknade kritiska konfigurationer: {missing_keys}")
            return False
        
        logger.info("✅ AWS konfiguration validerad")
        return True
    
    def get_aws_credentials(self) -> Dict[str, Optional[str]]:
        """Hämta AWS credentials."""
        return {
            'aws_access_key_id': self.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': self.get('AWS_SECRET_ACCESS_KEY'),
            'aws_session_token': self.get('AWS_SESSION_TOKEN'),
            'region_name': self.get('AWS_REGION', 'eu-north-1')
        }
    
    def get_sagemaker_config(self) -> Dict[str, Any]:
        """Hämta SageMaker konfiguration."""
        return {
            'framework': self.get('SAGEMAKER_FRAMEWORK', 'pytorch'),
            'framework_version': self.get('SAGEMAKER_FRAMEWORK_VERSION', '1.12.1'),
            'py_version': self.get('SAGEMAKER_PYTHON_VERSION', 'py38'),
            'instance_type': self.get('SAGEMAKER_INSTANCE_TYPE', 'ml.m5.2xlarge'),
            'instance_count': self.get_int('SAGEMAKER_INSTANCE_COUNT', 6),
            'use_spot_instances': self.get_bool('SAGEMAKER_USE_SPOT_INSTANCES', True),
            'max_run': self.get_int('SAGEMAKER_MAX_RUN_TIME', 93600),
            'max_wait': self.get_int('SAGEMAKER_MAX_WAIT_TIME', 187200),
            'base_job_name': self.get('SAGEMAKER_BASE_JOB_NAME', 'master-poc-preprocessing-v5'),
            'role': self.get('AWS_SAGEMAKER_ROLE_ARN')
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Hämta processing konfiguration."""
        return {
            'case_range': self.get('PROCESSING_CASE_RANGE', '1-3000'),
            'batch_size': self.get_int('PROCESSING_BATCH_SIZE', 50),
            'checkpoint_interval': self.get_int('PROCESSING_CHECKPOINT_INTERVAL', 50),
            'enable_checkpoints': self.get_bool('PROCESSING_ENABLE_CHECKPOINTS', True),
            'timeseries_features': self.get_int('MASTER_POC_TIMESERIES_FEATURES', 16),
            'static_features': self.get_int('MASTER_POC_STATIC_FEATURES', 6),
            'output_features': self.get_int('MASTER_POC_OUTPUT_FEATURES', 8),
            'window_size': self.get_int('MASTER_POC_WINDOW_SIZE', 300),
            'step_size': self.get_int('MASTER_POC_STEP_SIZE', 30),
            'normalization_range': self.get_list('MASTER_POC_NORMALIZATION_RANGE', default=['-1.0', '1.0'])
        }
    
    def get_s3_config(self) -> Dict[str, str]:
        """Hämta S3 konfiguration."""
        return {
            'primary_bucket': self.get('S3_PRIMARY_BUCKET'),
            'secondary_bucket': self.get('S3_SECONDARY_BUCKET'),
            'input_path': self.get_s3_path('S3_INPUT_PATH'),
            'output_path': self.get_s3_path('S3_OUTPUT_PATH'),
            'checkpoint_path': self.get_s3_path('S3_CHECKPOINT_PATH'),
            'code_path': self.get_s3_path('S3_CODE_PATH')
        }
    
    def print_config_summary(self) -> None:
        """Skriv ut konfigurationssammanfattning."""
        print("\n🔧 AWS Configuration Summary")
        print("=" * 50)
        
        # AWS Account
        print(f"AWS Account ID: {self.get('AWS_ACCOUNT_ID')}")
        print(f"AWS Region: {self.get('AWS_REGION')}")
        
        # S3 Configuration
        s3_config = self.get_s3_config()
        print(f"\nS3 Primary Bucket: {s3_config['primary_bucket']}")
        print(f"S3 Input Path: {s3_config['input_path']}")
        print(f"S3 Output Path: {s3_config['output_path']}")
        print(f"S3 Checkpoint Path: {s3_config['checkpoint_path']}")
        
        # SageMaker Configuration
        sm_config = self.get_sagemaker_config()
        print(f"\nSageMaker Instance: {sm_config['instance_type']} x{sm_config['instance_count']}")
        print(f"SageMaker Framework: {sm_config['framework']} {sm_config['framework_version']}")
        print(f"Spot Instances: {sm_config['use_spot_instances']}")
        
        # Processing Configuration
        proc_config = self.get_processing_config()
        print(f"\nProcessing Case Range: {proc_config['case_range']}")
        print(f"Batch Size: {proc_config['batch_size']}")
        print(f"Checkpoint Interval: {proc_config['checkpoint_interval']}")
        print(f"Features: {proc_config['timeseries_features']} timeseries + {proc_config['static_features']} static → {proc_config['output_features']} output")
        
        print("=" * 50)


# Global configuration instance
aws_config = AWSConfig()

# Convenience functions
def get_aws_config() -> AWSConfig:
    """Hämta global AWS configuration instance."""
    return aws_config

def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Hämta konfigurationsvärde från global instance."""
    return aws_config.get(key, default)

def validate_aws_config() -> bool:
    """Validera global AWS configuration."""
    return aws_config.validate_config()

def print_aws_config() -> None:
    """Skriv ut global AWS configuration summary."""
    aws_config.print_config_summary()


if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing AWS Configuration Loader...")
    
    config = AWSConfig()
    
    # Validera konfiguration
    if config.validate_config():
        print("✅ Configuration validation passed")
        config.print_config_summary()
    else:
        print("❌ Configuration validation failed")
    
    # Testa olika konfigurationstyper
    print(f"\n📊 Configuration Tests:")
    print(f"Instance Count (int): {config.get_int('SAGEMAKER_INSTANCE_COUNT', 6)}")
    print(f"Spot Instances (bool): {config.get_bool('SAGEMAKER_USE_SPOT_INSTANCES', True)}")
    print(f"Normalization Range (list): {config.get_list('MASTER_POC_NORMALIZATION_RANGE', default=['-1.0', '1.0'])}")
    
    # Testa S3 path parsing
    s3_path = config.get('S3_INPUT_PATH', 's3://test-bucket/path/')
    print(f"S3 Bucket: {config.get_bucket_name(s3_path)}")
    print(f"S3 Key: {config.get_s3_key(s3_path)}")
