#!/usr/bin/env python3
"""
Master POC Feature Handler v5.0
===============================

Hanterar missing features och feature validation enligt Master POC spec.
Baserat på Master_POC_CNN-LSTM-LSTM_v5.0.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

class MasterPOCFeatureHandler:
    """Hanterar missing features och feature validation för Master POC"""
    
    def __init__(self):
        self.required_timeseries_features = [
            'HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS',
            'Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF',
            'TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev'
        ]
        
        self.required_static_features = ['age', 'sex', 'height', 'weight', 'bmi', 'asa']
        
        # Default values enligt Master POC spec
        self.default_values = {
            'HR': 70,
            'BP_SYS': 140,
            'BP_DIA': 80,
            'BP_MAP': None,  # Beräknas från BP_SYS och BP_DIA
            'SPO2': 96,
            'ETCO2': 0.0,
            'BIS': 0.0,
            'Propofol_INF': 0.0,
            'Remifentanil_INF': 0.0,
            'Noradrenalin_INF': 0.0,
            'TV': 0.0,
            'PEEP': 0.0,
            'FIO2': 0.0,
            'RR': 0.0,
            'etSEV': 0.0,
            'inSev': 0.0,
            'age': 50,
            'sex': -1,  # Female
            'height': 170,
            'weight': 70,
            'bmi': 24.2,
            'asa': 2
        }
    
    def validate_required_features(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validera att alla required features finns i data.
        
        Args:
            data: Dictionary med feature data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_features)
        """
        missing_features = []
        
        # Kontrollera timeseries features
        for feature in self.required_timeseries_features:
            if feature not in data:
                missing_features.append(feature)
        
        # Kontrollera static features
        for feature in self.required_static_features:
            if feature not in data:
                missing_features.append(feature)
        
        is_valid = len(missing_features) == 0
        
        if not is_valid:
            logger.warning(f"⚠️ Missing required features: {missing_features}")
        else:
            logger.info("✅ All required features present")
        
        return is_valid, missing_features
    
    def handle_missing_features(self, data: Dict[str, Any], missing_features: List[str]) -> Dict[str, Any]:
        """
        Hantera missing features genom att lägga till default values.
        
        Args:
            data: Dictionary med feature data
            missing_features: Lista med missing features
            
        Returns:
            Dict[str, Any]: Data med missing features fyllda med default values
        """
        handled_data = data.copy()
        
        for feature in missing_features:
            if feature in self.default_values:
                default_value = self.default_values[feature]
                
                # Special hantering för BP_MAP (beräknas från BP_SYS och BP_DIA)
                if feature == 'BP_MAP':
                    if 'BP_SYS' in handled_data and 'BP_DIA' in handled_data:
                        # Beräkna MAP från SYS och DIA
                        sys_val = handled_data['BP_SYS']
                        dia_val = handled_data['BP_DIA']
                        if pd.notna(sys_val) and pd.notna(dia_val):
                            default_value = (sys_val + 2 * dia_val) / 3
                        else:
                            default_value = (140 + 2 * 80) / 3  # Default MAP från default SYS/DIA
                    else:
                        default_value = (140 + 2 * 80) / 3  # Default MAP
                
                handled_data[feature] = default_value
                logger.info(f"🔧 Added missing feature {feature} with default value {default_value}")
            else:
                logger.error(f"❌ No default value defined for missing feature {feature}")
        
        return handled_data
    
    def validate_feature_ranges(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validera att alla features ligger inom kliniska ranges.
        
        Args:
            data: Dictionary med feature data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, out_of_range_features)
        """
        out_of_range_features = []
        
        # Kliniska ranges enligt Master POC spec
        clinical_ranges = {
            'HR': (20, 200),
            'BP_SYS': (60, 250),
            'BP_DIA': (30, 150),
            'BP_MAP': (40, 180),
            'SPO2': (70, 100),
            'ETCO2': (2.0, 8.0),
            'BIS': (0, 100),
            'Propofol_INF': (0, 12),
            'Remifentanil_INF': (0, 0.8),
            'Noradrenalin_INF': (0, 0.5),
            'TV': (0, 12),
            'PEEP': (0, 30),
            'FIO2': (21, 100),
            'RR': (6, 30),
            'etSEV': (0, 6),
            'inSev': (0, 8),
            'age': (0, 120),
            'sex': (-1, 1),
            'height': (100, 230),
            'weight': (20, 200),
            'bmi': (10, 50),
            'asa': (1, 6)
        }
        
        for feature, value in data.items():
            if feature in clinical_ranges and pd.notna(value):
                min_val, max_val = clinical_ranges[feature]
                if value < min_val or value > max_val:
                    out_of_range_features.append(f"{feature}={value} (range: {min_val}-{max_val})")
        
        is_valid = len(out_of_range_features) == 0
        
        if not is_valid:
            logger.warning(f"⚠️ Features out of clinical range: {out_of_range_features}")
        else:
            logger.info("✅ All features within clinical ranges")
        
        return is_valid, out_of_range_features
    
    def process_missing_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Komplett process för att hantera missing features.
        
        Args:
            data: Dictionary med feature data
            
        Returns:
            Dict[str, Any]: Data med missing features hanterade
        """
        # Validera required features
        is_valid, missing_features = self.validate_required_features(data)
        
        # Hantera missing features
        if missing_features:
            data = self.handle_missing_features(data, missing_features)
        
        # Validera ranges
        is_range_valid, out_of_range = self.validate_feature_ranges(data)
        
        return data

def create_feature_handler() -> MasterPOCFeatureHandler:
    """Skapa en ny MasterPOCFeatureHandler instans."""
    return MasterPOCFeatureHandler()
