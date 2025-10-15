#!/usr/bin/env python3
"""
Minimal config module för Master POC v5.0
Löser import-fel för alla moduler som importerar 'from config import get_config'
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

def get_config():
    """
    Returnera konfiguration med fallback-hierarki:
    1. YAML-fil (om tillgänglig)
    2. Environment variables
    3. Hardkodade defaults
    """
    try:
        # Försök ladda från YAML först
        yaml_path = "configs/master_poc_v5_config.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Config loaded from YAML")
            return config
    except Exception as e:
        logger.warning(f"Could not load YAML config: {e}")
    
    # Fallback till minimal config
    return {
        'master_poc': {
            'timeseries_features': 16,
            'static_features': 6,
            'output_features': 8,
            'window_size': 300,
            'step_size': 30,
            'normalization_range': [-1, 1]
        },
        'processing': {
            'batch_size': 50,
            'checkpoint_interval': 50,
            'enable_checkpoints': True
        },
        's3': {
            'bucket': 'master-poc-v1.0',
            'input_path': 's3://master-poc-v1.0/raw-data/',
            'output_path': 's3://master-poc-v1.0/processed-data/master-poc-pipeline/'
        }
    }

# Bakåtkompatibilitet för olika import-patterns
def get_config_manager():
    """Alias för get_config() för bakåtkompatibilitet."""
    return get_config()

# Ytterligare bakåtkompatibilitetsfunktioner
def get_feature_mapping():
    """Returnera feature mapping konfiguration."""
    return {
        'timeseries_features': 16,
        'static_features': 6,
        'output_features': 8
    }

def get_safety_limits_dict(*args, **kwargs):
    """Returnera säkerhetsgränser för validering."""
    return {
        'HR': {'min': 20, 'max': 200},
        'BP_SYS': {'min': 60, 'max': 250},
        'BP_DIA': {'min': 30, 'max': 150}
    }

def get_drug_concentrations():
    """Returnera läkemedelskoncentrationer."""
    return {
        'propofol_concentration': 20.0,  # mg/ml
        'remifentanil_concentration': 20.0,  # μg/ml
        'noradrenalin_concentration': 0.1   # mg/ml
    }
