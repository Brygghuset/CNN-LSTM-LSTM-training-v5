#!/usr/bin/env python3
"""
Master POC Unified Normalization v5.0
=====================================

Implementerar unified normalization enligt Master POC CNN-LSTM-LSTM v5.0 specifikation.
Baserat på Master_POC_CNN-LSTM-LSTM_v5.0.md och AWS_CHECKLIST_V5.0_3000_CASES.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class MasterPOCUnifiedNormalizer:
    """Unified Normalization för Master POC preprocessing"""
    
    def __init__(self, target_range: Tuple[float, float] = (-1, 1)):
        """
        Initialisera Master POC Unified Normalizer.
        
        Args:
            target_range: Mål-intervall för normalisering (default: [-1, 1])
        """
        self.target_min, self.target_max = target_range
        self.target_range = target_range
        
        # Master POC specificerade kliniska ranges enligt specifikationen
        self.clinical_ranges = {
            # Vital Signs (7 features)
            'HR': (20, 200),
            'BP_SYS': (60, 250),
            'BP_DIA': (30, 150),
            'BP_MAP': (40, 180),
            'SPO2': (70, 100),
            'ETCO2': (2.0, 8.0),
            'BIS': (0, 100),
            
            # Drug Infusions (3 features)
            'Propofol_INF': (0, 12),
            'Remifentanil_INF': (0, 0.8),
            'Noradrenalin_INF': (0, 0.5),
            
            # Ventilator Settings (6 features)
            'TV': (0, 12),
            'PEEP': (0, 30),
            'FIO2': (21, 100),
            'RR': (6, 30),
            'etSEV': (0, 6),
            'inSev': (0, 8),
            
            # Static Patient Features (6 features)
            'age': (0, 120),
            'sex': (-1, 1),
            'height': (100, 230),
            'weight': (20, 200),
            'bmi': (10, 50),
            'asa': (1, 6)
        }
        
        logger.info(f"Master POC Unified Normalizer initialiserad med target range: {target_range}")
    
    def normalize_feature(self, values: Union[np.ndarray, pd.Series, float], feature_name: str) -> np.ndarray:
        """
        Normalisera en enskild feature enligt Master POC specifikation.
        
        Formula: normalized_value = (value - min_clinical) / (max_clinical - min_clinical) × (target_max - target_min) + target_min
        
        Args:
            values: Värden att normalisera
            feature_name: Namn på feature
            
        Returns:
            Normaliserade värden i target range
        """
        if feature_name not in self.clinical_ranges:
            raise ValueError(f"Okänd feature: {feature_name}. Tillgängliga: {list(self.clinical_ranges.keys())}")
        
        # Konvertera till numpy array
        if isinstance(values, (float, int)):
            values = np.array([values])
        elif isinstance(values, pd.Series):
            values = values.values
        elif not isinstance(values, np.ndarray):
            values = np.array(values)
        
        min_clinical, max_clinical = self.clinical_ranges[feature_name]
        
        # Master POC formula: normalized_value = (value - min_clinical) / (max_clinical - min_clinical) × (target_max - target_min) + target_min
        normalized = (values - min_clinical) / (max_clinical - min_clinical) * (self.target_max - self.target_min) + self.target_min
        
        # Hantera NaN värden (behåll som NaN)
        normalized = np.where(np.isnan(values), np.nan, normalized)
        
        # Clamp till target range för att hantera extremvärden
        normalized = np.clip(normalized, self.target_min, self.target_max)
        
        return normalized
    
    def denormalize_feature(self, normalized_values: Union[np.ndarray, pd.Series, float], feature_name: str) -> np.ndarray:
        """
        Denormalisera en feature tillbaka till ursprungliga enheter.
        
        Args:
            normalized_values: Normaliserade värden i target range
            feature_name: Namn på feature
            
        Returns:
            Denormaliserade värden i ursprungliga enheter
        """
        if feature_name not in self.clinical_ranges:
            raise ValueError(f"Okänd feature: {feature_name}. Tillgängliga: {list(self.clinical_ranges.keys())}")
        
        # Konvertera till numpy array
        if isinstance(normalized_values, (float, int)):
            normalized_values = np.array([normalized_values])
        elif isinstance(normalized_values, pd.Series):
            normalized_values = normalized_values.values
        elif not isinstance(normalized_values, np.ndarray):
            normalized_values = np.array(normalized_values)
        
        min_clinical, max_clinical = self.clinical_ranges[feature_name]
        
        # Reverse formula: value = (normalized - target_min) / (target_max - target_min) * (max_clinical - min_clinical) + min_clinical
        denormalized = (normalized_values - self.target_min) / (self.target_max - self.target_min) * (max_clinical - min_clinical) + min_clinical
        
        # Hantera NaN värden (behåll som NaN)
        denormalized = np.where(np.isnan(normalized_values), np.nan, denormalized)
        
        return denormalized
    
    def normalize_dataframe(self, df: pd.DataFrame, feature_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Normalisera en DataFrame med features.
        
        Args:
            df: DataFrame att normalisera
            feature_columns: Lista med kolumner att normalisera (None = alla kända features)
            
        Returns:
            Normaliserad DataFrame
        """
        result = df.copy()
        
        if feature_columns is None:
            # Använd alla kolumner som finns i clinical_ranges
            feature_columns = [col for col in df.columns if col in self.clinical_ranges]
        
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Kolumn {col} finns inte i DataFrame")
                continue
                
            if col not in self.clinical_ranges:
                logger.warning(f"Okänd feature {col}, hoppar över normalisering")
                continue
            
            original_values = df[col].values
            normalized_values = self.normalize_feature(original_values, col)
            result[col] = normalized_values
        
        logger.info(f"Normaliserade {len(feature_columns)} features till {self.target_range}")
        
        return result
    
    def denormalize_dataframe(self, df: pd.DataFrame, feature_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Denormalisera en DataFrame tillbaka till ursprungliga enheter.
        
        Args:
            df: Normaliserad DataFrame
            feature_columns: Lista med kolumner att denormalisera (None = alla kända features)
            
        Returns:
            Denormaliserad DataFrame
        """
        result = df.copy()
        
        if feature_columns is None:
            # Använd alla kolumner som finns i clinical_ranges
            feature_columns = [col for col in df.columns if col in self.clinical_ranges]
        
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Kolumn {col} finns inte i DataFrame")
                continue
                
            if col not in self.clinical_ranges:
                logger.warning(f"Okänd feature {col}, hoppar över denormalisering")
                continue
            
            normalized_values = df[col].values
            denormalized_values = self.denormalize_feature(normalized_values, col)
            result[col] = denormalized_values
        
        logger.info(f"Denormaliserade {len(feature_columns)} features från {self.target_range}")
        
        return result
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Hämta information om en feature.
        
        Args:
            feature_name: Namn på feature
            
        Returns:
            Dict med feature-information
        """
        if feature_name not in self.clinical_ranges:
            raise ValueError(f"Okänd feature: {feature_name}")
        
        min_clinical, max_clinical = self.clinical_ranges[feature_name]
        return {
            'name': feature_name,
            'min_clinical': min_clinical,
            'max_clinical': max_clinical,
            'target_range': self.target_range
        }
    
    def get_all_features(self) -> list:
        """Hämta lista med alla tillgängliga features."""
        return list(self.clinical_ranges.keys())
    
    def validate_ranges(self, df: pd.DataFrame, feature_columns: Optional[list] = None) -> Dict[str, Dict]:
        """
        Validera att värden ligger inom rimliga gränser.
        
        Args:
            df: DataFrame att validera
            feature_columns: Lista med kolumner att validera
            
        Returns:
            Dict med valideringsresultat per feature
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col in self.clinical_ranges]
        
        validation_results = {}
        
        for col in feature_columns:
            if col not in df.columns or col not in self.clinical_ranges:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            min_clinical, max_clinical = self.clinical_ranges[col]
            min_val, max_val = values.min(), values.max()
            
            # Kontrollera om värden ligger utanför kliniska ranges
            below_min = (values < min_clinical).sum()
            above_max = (values > max_clinical).sum()
            
            validation_results[col] = {
                'data_range': (min_val, max_val),
                'clinical_range': (min_clinical, max_clinical),
                'values_below_min': below_min,
                'values_above_max': above_max,
                'total_values': len(values),
                'within_range': below_min == 0 and above_max == 0
            }
        
        return validation_results

def create_master_poc_unified_normalizer() -> MasterPOCUnifiedNormalizer:
    """Skapa en ny MasterPOCUnifiedNormalizer instans."""
    return MasterPOCUnifiedNormalizer(target_range=(-1, 1))
