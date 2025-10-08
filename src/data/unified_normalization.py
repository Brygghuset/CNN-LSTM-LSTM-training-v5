"""
Unified Normalization enligt Master POC CNN-LSTM-LSTM specifikation.

Implementerar standardiserad normalisering för alla features till [-1, 1] intervall
med specificerade kliniska ranges för varje parameter.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from data.preprocessing.master_poc_smart_forward_fill import get_master_poc_defaults

logger = logging.getLogger(__name__)


@dataclass
class FeatureRange:
    """Klinisk range för en feature."""
    min_clinical: float
    max_clinical: float
    unit: str
    description: str


class UnifiedNormalizer:
    """
    Unified Normalization enligt Master POC specifikation.
    
    Normaliserar alla features till [-1, 1] intervall med specificerade kliniska ranges.
    Formula: normalized_value = (value - min_clinical) / (max_clinical - min_clinical) × (target_max - target_min) + target_min
    """
    
    def __init__(self, target_range: Tuple[float, float] = (-1, 1)):
        """
        Initialisera Unified Normalizer.
        
        Args:
            target_range: Mål-intervall för normalisering (default: [-1, 1])
        """
        self.target_min, self.target_max = target_range
        self.target_range = target_range
        
        # Master POC specificerade kliniska ranges
        self.feature_ranges = self._get_master_poc_ranges()
        
        # Master POC default-värden för imputation
        self.master_poc_defaults = get_master_poc_defaults()
        
        logger.info(f"Unified Normalizer initialiserad med target range: {target_range}")
        logger.info(f"Master POC default-värden laddade: {len(self.master_poc_defaults)} parametrar")
    
    def _get_master_poc_ranges(self) -> Dict[str, FeatureRange]:
        """Hämta Master POC specificerade kliniska ranges."""
        return {
            # Vital Signs (7 features)
            'HR': FeatureRange(20, 200, 'BPM', 'Heart Rate'),
            'BP_SYS': FeatureRange(60, 250, 'mmHg', 'Systolic Blood Pressure'),
            'BP_DIA': FeatureRange(30, 150, 'mmHg', 'Diastolic Blood Pressure'),
            'BP_MAP': FeatureRange(40, 180, 'mmHg', 'Mean Arterial Pressure'),
            'SPO2': FeatureRange(70, 100, '%', 'Oxygen Saturation'),
            'ETCO2': FeatureRange(2.0, 8.0, 'kPa', 'End-Tidal CO2'),
            'BIS': FeatureRange(0, 100, '', 'Bispectral Index'),
            
            # Drug Infusions (3 features)
            'Propofol_INF': FeatureRange(0, 12, 'mg/kg/h', 'Propofol Infusion'),
            'Remifentanil_INF': FeatureRange(0, 0.8, 'mcg/kg/min', 'Remifentanil Infusion'),
            'Noradrenalin_INF': FeatureRange(0, 0.5, 'mcg/kg/min', 'Noradrenalin Infusion'),
            
            # Ventilator Settings (6 features)
            'TV': FeatureRange(0, 12, 'ml/kg IBW', 'Tidal Volume'),
            'PEEP': FeatureRange(0, 30, 'cmH2O', 'PEEP'),
            'FIO2': FeatureRange(21, 100, '%', 'FiO2'),
            'RR': FeatureRange(6, 30, 'breaths/min', 'Respiratory Rate'),
            'etSEV': FeatureRange(0, 6, 'kPa', 'Expiratory Sevoflurane'),
            'inSev': FeatureRange(0, 8, 'kPa', 'Inspiratory Sevoflurane'),
            
            # Static Patient Features (6 features)
            'age': FeatureRange(0, 120, 'years', 'Age'),
            'sex': FeatureRange(-1, 1, '', 'Sex (-1=Female, 1=Male)'),
            'height': FeatureRange(100, 230, 'cm', 'Height'),
            'weight': FeatureRange(20, 200, 'kg', 'Weight'),
            'bmi': FeatureRange(10, 50, 'kg/m²', 'BMI'),
            'asa': FeatureRange(1, 6, '', 'ASA Score')
        }
    
    def normalize_feature(self, values: Union[np.ndarray, pd.Series, float], feature_name: str, 
                          use_master_poc_defaults: bool = True) -> np.ndarray:
        """
        Normalisera en enskild feature enligt Master POC specifikation.
        
        Args:
            values: Värden att normalisera
            feature_name: Namn på feature
            use_master_poc_defaults: Om True används Master POC default-värden för NaN
            
        Returns:
            Normaliserade värden i [-1, 1] intervall
        """
        if feature_name not in self.feature_ranges:
            raise ValueError(f"Okänd feature: {feature_name}. Tillgängliga: {list(self.feature_ranges.keys())}")
        
        # Konvertera till numpy array
        if isinstance(values, (float, int)):
            values = np.array([values])
        elif isinstance(values, pd.Series):
            values = values.values
        elif not isinstance(values, np.ndarray):
            values = np.array(values)
        
        range_info = self.feature_ranges[feature_name]
        min_clinical = range_info.min_clinical
        max_clinical = range_info.max_clinical
        
        # Master POC formula: normalized_value = (value - min_clinical) / (max_clinical - min_clinical) × (target_max - target_min) + target_min
        normalized = (values - min_clinical) / (max_clinical - min_clinical) * (self.target_max - self.target_min) + self.target_min
        
        # Hantera NaN värden
        if use_master_poc_defaults and feature_name in self.master_poc_defaults:
            # Använd Master POC default-värden för NaN
            master_poc_default = self.master_poc_defaults[feature_name]
            normalized_default = (master_poc_default - min_clinical) / (max_clinical - min_clinical) * (self.target_max - self.target_min) + self.target_min
            normalized = np.where(np.isnan(values), normalized_default, normalized)
            logger.debug(f"Använde Master POC default ({master_poc_default}) för NaN i {feature_name}")
        else:
            # Behåll NaN som NaN
            normalized = np.where(np.isnan(values), np.nan, normalized)
        
        # Clamp till target range för att hantera extremvärden
        normalized = np.clip(normalized, self.target_min, self.target_max)
        
        return normalized
    
    def denormalize_feature(self, normalized_values: Union[np.ndarray, pd.Series, float], feature_name: str) -> np.ndarray:
        """
        Denormalisera en feature tillbaka till ursprungliga enheter.
        
        Args:
            normalized_values: Normaliserade värden i [-1, 1]
            feature_name: Namn på feature
            
        Returns:
            Denormaliserade värden i ursprungliga enheter
        """
        if feature_name not in self.feature_ranges:
            raise ValueError(f"Okänd feature: {feature_name}. Tillgängliga: {list(self.feature_ranges.keys())}")
        
        # Konvertera till numpy array
        if isinstance(normalized_values, (float, int)):
            normalized_values = np.array([normalized_values])
        elif isinstance(normalized_values, pd.Series):
            normalized_values = normalized_values.values
        elif not isinstance(normalized_values, np.ndarray):
            normalized_values = np.array(normalized_values)
        
        range_info = self.feature_ranges[feature_name]
        min_clinical = range_info.min_clinical
        max_clinical = range_info.max_clinical
        
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
            # Använd alla kolumner som finns i feature_ranges
            feature_columns = [col for col in df.columns if col in self.feature_ranges]
        
        normalization_stats = {}
        
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Kolumn {col} finns inte i DataFrame")
                continue
                
            if col not in self.feature_ranges:
                logger.warning(f"Okänd feature {col}, hoppar över normalisering")
                continue
            
            original_values = df[col].values
            normalized_values = self.normalize_feature(original_values, col)
            result[col] = normalized_values
            
            # Statistik för loggning
            if not np.all(np.isnan(original_values)):
                normalization_stats[col] = {
                    'original_range': (np.nanmin(original_values), np.nanmax(original_values)),
                    'normalized_range': (np.nanmin(normalized_values), np.nanmax(normalized_values)),
                    'clinical_range': (self.feature_ranges[col].min_clinical, self.feature_ranges[col].max_clinical),
                    'unit': self.feature_ranges[col].unit
                }
        
        logger.info(f"Normaliserade {len(feature_columns)} features till {self.target_range}")
        for col, stats in normalization_stats.items():
            logger.debug(f"{col}: {stats['original_range']} {stats['unit']} → {stats['normalized_range']}")
        
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
            # Använd alla kolumner som finns i feature_ranges
            feature_columns = [col for col in df.columns if col in self.feature_ranges]
        
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Kolumn {col} finns inte i DataFrame")
                continue
                
            if col not in self.feature_ranges:
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
        if feature_name not in self.feature_ranges:
            raise ValueError(f"Okänd feature: {feature_name}")
        
        range_info = self.feature_ranges[feature_name]
        return {
            'name': feature_name,
            'min_clinical': range_info.min_clinical,
            'max_clinical': range_info.max_clinical,
            'unit': range_info.unit,
            'description': range_info.description,
            'target_range': self.target_range
        }
    
    def get_all_features(self) -> list:
        """Hämta lista med alla tillgängliga features."""
        return list(self.feature_ranges.keys())
    
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
            feature_columns = [col for col in df.columns if col in self.feature_ranges]
        
        validation_results = {}
        
        for col in feature_columns:
            if col not in df.columns or col not in self.feature_ranges:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            range_info = self.feature_ranges[col]
            min_val, max_val = values.min(), values.max()
            
            # Kontrollera om värden ligger utanför kliniska ranges
            below_min = (values < range_info.min_clinical).sum()
            above_max = (values > range_info.max_clinical).sum()
            
            validation_results[col] = {
                'data_range': (min_val, max_val),
                'clinical_range': (range_info.min_clinical, range_info.max_clinical),
                'values_below_min': below_min,
                'values_above_max': above_max,
                'total_values': len(values),
                'unit': range_info.unit,
                'within_range': below_min == 0 and above_max == 0
            }
        
        return validation_results


def create_unified_normalizer() -> UnifiedNormalizer:
    """Factory function för att skapa Unified Normalizer."""
    return UnifiedNormalizer(target_range=(-1, 1))


# Convenience functions för bakåtkompatibilitet
def normalize_features_unified(df: pd.DataFrame, feature_columns: Optional[list] = None) -> Tuple[pd.DataFrame, UnifiedNormalizer]:
    """
    Normalisera features med Unified Normalization.
    
    Returns:
        Tuple av (normaliserad DataFrame, normalizer instance)
    """
    normalizer = create_unified_normalizer()
    normalized_df = normalizer.normalize_dataframe(df, feature_columns)
    return normalized_df, normalizer


def denormalize_features_unified(df: pd.DataFrame, normalizer: UnifiedNormalizer, feature_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Denormalisera features med Unified Normalization.
    """
    return normalizer.denormalize_dataframe(df, feature_columns)
