#!/usr/bin/env python3
"""
Master POC Smart Forward Fill v5.0
==================================

Implementerar smart forward fill enligt Master POC CNN-LSTM-LSTM v5.0 specifikation.
Baserat på Master_POC_CNN-LSTM-LSTM_v5.0.md och AWS_CHECKLIST_V5.0_3000_CASES.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class MasterPOCSmartForwardFill:
    """Smart forward fill för Master POC preprocessing"""
    
    def __init__(self):
        # Clinical zeros enligt Master POC spec
        self.clinical_zeros = {
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
        }
        
        # Default values för vital signs enligt Master POC spec
        self.default_values = {
            'HR': 70.0,
            'BP_SYS': 140.0,
            'BP_DIA': 80.0,
            'BP_MAP': 93.3,  # Beräknat från default SYS/DBP
            'SPO2': 96.0,
        }
        
        # Clinical ranges för normalization enligt Master POC spec
        self.clinical_ranges = {
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
        }
    
    def apply_clinical_zeros(self, series: pd.Series, feature_name: str) -> pd.Series:
        """
        Applicera kliniska nollor för parametrar som ska ha noll som default.
        
        Args:
            series: Pandas Series med data
            feature_name: Namn på feature
            
        Returns:
            pd.Series: Series med kliniska nollor applicerade
        """
        if feature_name in self.clinical_zeros:
            clinical_zero = self.clinical_zeros[feature_name]
            
            # Ersätt NaN med klinisk nolla
            series_filled = series.fillna(clinical_zero)
            
            logger.debug(f"Applied clinical zero {clinical_zero} for {feature_name}")
            return series_filled
        
        return series
    
    def apply_default_values(self, series: pd.Series, feature_name: str) -> pd.Series:
        """
        Applicera default-värden för vital signs.
        
        Args:
            series: Pandas Series med data
            feature_name: Namn på feature
            
        Returns:
            pd.Series: Series med default-värden applicerade
        """
        if feature_name in self.default_values:
            default_value = self.default_values[feature_name]
            
            # Ersätt NaN med default-värde
            series_filled = series.fillna(default_value)
            
            logger.debug(f"Applied default value {default_value} for {feature_name}")
            return series_filled
        
        return series
    
    def smart_forward_fill(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """
        Implementerar smart forward fill enligt Master POC spec.
        
        Steg 1: Identifiera isolerade NaN
        Steg 2: Mean-imputation för isolerade NaN
        Steg 3: Forward fill för kvarvarande NaN
        Steg 4: Backward fill som fallback för initiala NaN
        
        Args:
            series: Pandas Series med data
            max_consecutive_nans: Max antal konsekutiva NaN att forward fill
            
        Returns:
            pd.Series: Series med smart forward fill applicerat
        """
        if series.empty:
            return series
        
        # Kopiera series för att undvika mutation av original
        imputed = series.copy()
        
        # Steg 1: Identifiera isolerade NaN (NaN mellan två giltiga värden)
        isolated_nans = []
        for i in range(1, len(imputed) - 1):
            if pd.isna(imputed.iloc[i]) and not pd.isna(imputed.iloc[i-1]) and not pd.isna(imputed.iloc[i+1]):
                isolated_nans.append(i)
        
        # Steg 2: Mean-imputation för isolerade NaN
        if isolated_nans:
            # Beräkna mean från giltiga värden (från träningsdata enligt spec)
            valid_values = imputed.dropna()
            if len(valid_values) > 0:
                mean_value = valid_values.mean()
                for idx in isolated_nans:
                    imputed.iloc[idx] = mean_value
                logger.debug(f"Applied mean imputation {mean_value} to {len(isolated_nans)} isolated NaNs")
        
        # Steg 3: Forward fill för kvarvarande NaN (inklusive initiala)
        if max_consecutive_nans is not None:
            imputed = imputed.ffill(limit=max_consecutive_nans)
        else:
            imputed = imputed.ffill()
        
        # Steg 4: Backward fill som fallback för initiala NaN
        if imputed.isna().any():
            imputed = imputed.bfill()
            logger.debug("Applied backward fill as fallback for remaining NaNs")
        
        return imputed
    
    def apply_smart_forward_fill(self, series: pd.Series, feature_name: str, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """
        Applicera smart forward fill baserat på feature typ.
        
        Args:
            series: Pandas Series med data
            feature_name: Namn på feature
            max_consecutive_nans: Max antal konsekutiva NaN att forward fill
            
        Returns:
            pd.Series: Series med smart forward fill applicerat
        """
        # För vital signs: Använd default-värden först, sedan smart forward fill
        if feature_name in self.default_values:
            series = self.apply_default_values(series, feature_name)
            return self.smart_forward_fill(series, max_consecutive_nans)
        
        # För parametrar med kliniska nollor: Använd kliniska nollor först, sedan smart forward fill
        elif feature_name in self.clinical_zeros:
            series = self.apply_clinical_zeros(series, feature_name)
            return self.smart_forward_fill(series, max_consecutive_nans)
        
        # För andra parametrar: Använd endast smart forward fill
        else:
            return self.smart_forward_fill(series, max_consecutive_nans)
    
    def validate_clinical_ranges(self, series: pd.Series, feature_name: str) -> Tuple[bool, List[float]]:
        """
        Validera att värden ligger inom kliniska ranges.
        
        Args:
            series: Pandas Series med data
            feature_name: Namn på feature
            
        Returns:
            Tuple[bool, List[float]]: (is_valid, out_of_range_values)
        """
        if feature_name not in self.clinical_ranges:
            return True, []
        
        min_val, max_val = self.clinical_ranges[feature_name]
        valid_values = series.dropna()
        
        out_of_range = valid_values[(valid_values < min_val) | (valid_values > max_val)].tolist()
        is_valid = len(out_of_range) == 0
        
        if not is_valid:
            logger.warning(f"Feature {feature_name} has {len(out_of_range)} values outside clinical range [{min_val}, {max_val}]")
        
        return is_valid, out_of_range
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Hämta information om en feature.
        
        Args:
            feature_name: Namn på feature
            
        Returns:
            Dict[str, Any]: Feature information
        """
        info = {
            'name': feature_name,
            'has_clinical_zero': feature_name in self.clinical_zeros,
            'has_default_value': feature_name in self.default_values,
            'has_clinical_range': feature_name in self.clinical_ranges,
        }
        
        if info['has_clinical_zero']:
            info['clinical_zero'] = self.clinical_zeros[feature_name]
        
        if info['has_default_value']:
            info['default_value'] = self.default_values[feature_name]
        
        if info['has_clinical_range']:
            info['clinical_range'] = self.clinical_ranges[feature_name]
        
        return info
    
    def process_feature(self, series: pd.Series, feature_name: str, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """
        Processa en feature med smart forward fill.
        
        Args:
            series: Pandas Series med data
            feature_name: Namn på feature
            max_consecutive_nans: Max antal konsekutiva NaN att forward fill
            
        Returns:
            pd.Series: Processed series
        """
        logger.debug(f"Processing feature {feature_name}")
        
        # Applicera smart forward fill
        processed_series = self.apply_smart_forward_fill(series, feature_name, max_consecutive_nans)
        
        # Validera kliniska ranges
        is_valid, out_of_range = self.validate_clinical_ranges(processed_series, feature_name)
        
        if not is_valid:
            logger.warning(f"Feature {feature_name} has values outside clinical range: {out_of_range[:5]}...")
        
        return processed_series

def create_smart_forward_fill() -> MasterPOCSmartForwardFill:
    """
    Skapa en ny MasterPOCSmartForwardFill instans.
    
    Returns:
        MasterPOCSmartForwardFill: Ny instans av smart forward fill processor
    """
    return MasterPOCSmartForwardFill()

def get_master_poc_defaults() -> Dict[str, float]:
    """
    Hämta Master POC default-värden för alla features.
    
    Returns:
        Dict[str, float]: Mapping av feature namn till default-värden
    """
    imputer = create_smart_forward_fill()
    
    # Kombinera clinical zeros och default values
    defaults = {}
    defaults.update(imputer.clinical_zeros)
    defaults.update(imputer.default_values)
    
    return defaults
