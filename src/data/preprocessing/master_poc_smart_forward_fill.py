"""
Master POC Smart Forward Fill Strategy enligt Master POC CNN-LSTM-LSTM specifikation.

Implementerar exakta default-värden och tidsbaserad imputation enligt Master POC dokumentet.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from data.preprocessing.imputation_strategies import ImputationStrategy

logger = logging.getLogger(__name__)


class MasterPOCSmartForwardFillStrategy(ImputationStrategy):
    """
    Master POC Smart Forward Fill Strategy enligt specifikationen.
    
    Implementerar:
    - Exakta default-värden per parameter
    - Tidsbaserad imputation (10 min för vital signs, 5 min för SPO2)
    - Klinisk nolla för drugs och ventilator
    - Smart forward fill med mean-imputation för isolerade NaN
    """
    
    def __init__(self):
        """Initialisera Master POC Smart Forward Fill Strategy."""
        # Master POC specificerade default-värden
        self.default_values = {
            # Vital Signs (7 features) - Default-värden före första giltiga värde
            'HR': 70.0,           # Default (70)
            'BP_SYS': 140.0,      # Default (140)
            'BP_DIA': 80.0,       # Default (80)
            'BP_MAP': None,       # Beräknas från default (SBP + 2*DBP)/3
            'SPO2': 96.0,         # Default (96)
            'ETCO2': 0.0,         # Klinisk nolla (0.0)
            'BIS': 0.0,           # Klinisk nolla (0.0)
            
            # Drug Infusions (3 features) - Klinisk nolla
            'Propofol_INF': 0.0,      # Klinisk nolla (0.0)
            'Remifentanil_INF': 0.0,  # Klinisk nolla (0.0)
            'Noradrenalin_INF': 0.0,   # Klinisk nolla (0.0)
            
            # Ventilator Settings (6 features) - Klinisk nolla
            'TV': 0.0,            # Klinisk nolla (0.0)
            'PEEP': 0.0,          # Klinisk nolla (0.0)
            'FIO2': 0.0,          # Klinisk nolla (0.0) - Note: Master POC säger 0.0, inte 21
            'RR': 0.0,            # Klinisk nolla (0.0)
            'etSEV': 0.0,         # Klinisk nolla (0.0)
            'inSev': 0.0          # Klinisk nolla (0.0)
        }
        
        # Tidsbaserade default-värden (i sekunder)
        self.time_based_defaults = {
            # Vital Signs - Default värde 10 min efter sista giltiga värde
            'HR': 600,            # 10 minuter
            'BP_SYS': 600,        # 10 minuter
            'BP_DIA': 600,        # 10 minuter
            'BP_MAP': 600,        # 10 minuter
            'SPO2': 300,          # 5 minuter (specifik för SPO2)
            'ETCO2': 0,           # Ingen tidsbaserad default (klinisk nolla direkt)
            'BIS': 0,             # Ingen tidsbaserad default (klinisk nolla direkt)
            
            # Drug Infusions - Ingen tidsbaserad default (klinisk nolla direkt)
            'Propofol_INF': 0,
            'Remifentanil_INF': 0,
            'Noradrenalin_INF': 0,
            
            # Ventilator Settings - Ingen tidsbaserad default (klinisk nolla direkt)
            'TV': 0,
            'PEEP': 0,
            'FIO2': 0,
            'RR': 0,
            'etSEV': 0,
            'inSev': 0
        }
        
        logger.info("Master POC Smart Forward Fill Strategy initialiserad")
        logger.info(f"Default-värden: {len(self.default_values)} parametrar")
        logger.info(f"Tidsbaserade defaults: {len([k for k, v in self.time_based_defaults.items() if v > 0])} parametrar")
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None, 
               feature_name: Optional[str] = None, time_index: Optional[pd.Series] = None) -> pd.Series:
        """
        Imputera saknade värden enligt Master POC Smart Forward Fill.
        
        Args:
            series: Serie att imputera
            max_consecutive_nans: Maximalt antal konsekutiva NaN (ignoreras för Master POC)
            feature_name: Namn på feature för att hämta rätt default-värden
            time_index: Tidsserie för tidsbaserad imputation (optional)
            
        Returns:
            Imputerad serie enligt Master POC specifikation
        """
        if series.empty:
            return series.copy()
        
        # Skapa kopia för att undvika att modifiera original
        imputed = series.copy()
        
        # Hämta Master POC default-värden för denna feature
        default_value = self._get_default_value(feature_name)
        time_based_seconds = self._get_time_based_seconds(feature_name)
        
        logger.debug(f"Imputerar {feature_name}: default={default_value}, time_based={time_based_seconds}s")
        
        # Steg 1: Identifiera isolerade NaN och använd mean-imputation
        isolated_nans = self._identify_isolated_nans(imputed)
        if isolated_nans:
            logger.debug(f"Hittade {len(isolated_nans)} isolerade NaN-värden för {feature_name}")
            mean_value = series.mean()
            if not pd.isna(mean_value):
                for idx in isolated_nans:
                    imputed.iloc[idx] = mean_value
                logger.debug(f"Använde mean-imputation ({mean_value:.2f}) för isolerade NaN")
        
        # Steg 2: Hantera initiala NaN med Master POC default-värden
        imputed = self._handle_initial_nans(imputed, default_value)
        
        # Steg 3: Tidsbaserad imputation för NaN efter sista giltiga värde (FÖRST)
        # Men bara för NaN som INTE är isolerade (redan hanterade med mean-imputation)
        if time_based_seconds > 0 and time_index is not None:
            imputed = self._handle_time_based_nans_excluding_isolated(imputed, time_index, default_value, time_based_seconds, isolated_nans)
        
        # Steg 4: Forward fill för kvarvarande NaN (EFTER tidsbaserad imputation)
        imputed = imputed.ffill()
        
        # Steg 5: Backward fill som fallback för kvarvarande NaN
        if imputed.isna().any():
            logger.debug(f"Använder backward fill fallback för {feature_name}")
            imputed = imputed.bfill()
        
        # Steg 6: Final fallback med Master POC default-värden
        if imputed.isna().any():
            logger.debug(f"Använder Master POC default fallback ({default_value}) för {feature_name}")
            imputed = imputed.fillna(default_value)
        
        return imputed
    
    def _get_default_value(self, feature_name: Optional[str]) -> float:
        """Hämta Master POC default-värde för feature."""
        if feature_name is None:
            return 0.0  # Fallback
        
        default_value = self.default_values.get(feature_name, 0.0)
        
        # Special case för BP_MAP - beräkna från BP_SYS och BP_DIA defaults
        if feature_name == 'BP_MAP' and default_value is None:
            # MAP ≈ (SBP + 2*DBP)/3
            sbp_default = self.default_values.get('BP_SYS', 140.0)
            dbp_default = self.default_values.get('BP_DIA', 80.0)
            default_value = (sbp_default + 2 * dbp_default) / 3
            logger.debug(f"Beräknade BP_MAP default: ({sbp_default} + 2*{dbp_default})/3 = {default_value:.1f}")
        
        return default_value
    
    def _get_time_based_seconds(self, feature_name: Optional[str]) -> int:
        """Hämta tidsbaserad default i sekunder för feature."""
        if feature_name is None:
            return 0
        return self.time_based_defaults.get(feature_name, 0)
    
    def _identify_isolated_nans(self, series: pd.Series) -> list:
        """Identifiera isolerade NaN-värden (NaN mellan två giltiga värden)."""
        isolated_nans = []
        for i in range(1, len(series) - 1):
            if (pd.isna(series.iloc[i]) and 
                not pd.isna(series.iloc[i-1]) and 
                not pd.isna(series.iloc[i+1])):
                isolated_nans.append(i)
        return isolated_nans
    
    def _handle_initial_nans(self, series: pd.Series, default_value: float) -> pd.Series:
        """Hantera initiala NaN-värden med Master POC default-värden."""
        imputed = series.copy()
        
        # Hitta första giltiga värde
        first_valid_idx = None
        for i, value in enumerate(series):
            if not pd.isna(value):
                first_valid_idx = i
                break
        
        # Om alla värden är NaN, använd default för alla
        if first_valid_idx is None:
            logger.debug("Alla värden är NaN, använder Master POC default för alla")
            return pd.Series([default_value] * len(series), index=series.index)
        
        # Ersätt initiala NaN med Master POC default-värden
        if first_valid_idx > 0:
            logger.debug(f"Ersätter {first_valid_idx} initiala NaN med Master POC default ({default_value})")
            imputed.iloc[:first_valid_idx] = default_value
        
        return imputed
    
    def _handle_time_based_nans(self, series: pd.Series, time_index: pd.Series, 
                               default_value: float, time_based_seconds: int) -> pd.Series:
        """
        Hantera tidsbaserade NaN-värden efter sista giltiga värde.
        
        Args:
            series: Serie att imputera
            time_index: Tidsserie (i sekunder)
            default_value: Default-värde att använda
            time_based_seconds: Antal sekunder efter sista giltiga värde
        """
        imputed = series.copy()
        
        # Hitta sista giltiga värde
        last_valid_idx = None
        for i in range(len(series) - 1, -1, -1):
            if not pd.isna(series.iloc[i]):
                last_valid_idx = i
                break
        
        if last_valid_idx is None:
            logger.debug("Inga giltiga värden hittades för tidsbaserad imputation")
            return imputed
        
        # Hitta NaN-värden inom tidsbaserad gräns efter sista giltiga värde
        last_valid_time = time_index.iloc[last_valid_idx]
        time_threshold = last_valid_time + time_based_seconds
        
        for i in range(last_valid_idx + 1, len(series)):
            if pd.isna(series.iloc[i]) and time_index.iloc[i] <= time_threshold:
                imputed.iloc[i] = default_value
                logger.debug(f"Tidsbaserad imputation: idx {i}, tid {time_index.iloc[i]}s <= {time_threshold}s")
        
        return imputed
    
    def _handle_time_based_nans_excluding_isolated(self, series: pd.Series, time_index: pd.Series, 
                                                  default_value: float, time_based_seconds: int, 
                                                  isolated_nans: list) -> pd.Series:
        """
        Hantera tidsbaserade NaN-värden efter sista giltiga värde, men exkludera isolerade NaN.
        
        Args:
            series: Serie att imputera
            time_index: Tidsserie (i sekunder)
            default_value: Default-värde att använda
            time_based_seconds: Antal sekunder efter sista giltiga värde
            isolated_nans: Lista med index för isolerade NaN (ska inte hanteras här)
        """
        imputed = series.copy()
        
        # Hitta sista giltiga värde
        last_valid_idx = None
        for i in range(len(series) - 1, -1, -1):
            if not pd.isna(series.iloc[i]):
                last_valid_idx = i
                break
        
        if last_valid_idx is None:
            logger.debug("Inga giltiga värden hittades för tidsbaserad imputation")
            return imputed
        
        # Hitta NaN-värden inom tidsbaserad gräns efter sista giltiga värde
        # Men exkludera isolerade NaN (redan hanterade med mean-imputation)
        last_valid_time = time_index.iloc[last_valid_idx]
        time_threshold = last_valid_time + time_based_seconds
        
        for i in range(last_valid_idx + 1, len(series)):
            if (pd.isna(series.iloc[i]) and 
                time_index.iloc[i] <= time_threshold and 
                i not in isolated_nans):
                imputed.iloc[i] = default_value
                logger.debug(f"Tidsbaserad imputation: idx {i}, tid {time_index.iloc[i]}s <= {time_threshold}s")
        
        return imputed
    
    def get_feature_defaults(self) -> Dict[str, float]:
        """Hämta alla Master POC default-värden."""
        defaults = {}
        for feature, default in self.default_values.items():
            if default is None and feature == 'BP_MAP':
                # Beräkna BP_MAP default
                defaults[feature] = self._get_default_value(feature)
            else:
                defaults[feature] = default
        return defaults
    
    def get_time_based_defaults(self) -> Dict[str, int]:
        """Hämta alla tidsbaserade default-värden."""
        return self.time_based_defaults.copy()
    
    def validate_master_poc_compliance(self, feature_name: str, imputed_series: pd.Series) -> Dict[str, Any]:
        """
        Validera att imputering följer Master POC specifikationer.
        
        Args:
            feature_name: Namn på feature
            imputed_series: Imputerad serie att validera
            
        Returns:
            Dict med valideringsresultat
        """
        validation_result = {
            'feature_name': feature_name,
            'total_values': len(imputed_series),
            'nan_count': imputed_series.isna().sum(),
            'default_value_used': self._get_default_value(feature_name),
            'time_based_seconds': self._get_time_based_seconds(feature_name),
            'compliance': True,
            'issues': []
        }
        
        # Kontrollera att inga NaN-värden kvarstår
        if validation_result['nan_count'] > 0:
            validation_result['compliance'] = False
            validation_result['issues'].append(f"Kvarvarande NaN-värden: {validation_result['nan_count']}")
        
        # Kontrollera att default-värden används korrekt
        expected_default = self._get_default_value(feature_name)
        if expected_default != 0.0:  # Endast för icke-noll default-värden
            default_count = (imputed_series == expected_default).sum()
            if default_count == 0:
                validation_result['issues'].append(f"Förväntat default-värde ({expected_default}) användes inte")
        
        return validation_result


def create_master_poc_smart_forward_fill() -> MasterPOCSmartForwardFillStrategy:
    """Factory function för att skapa Master POC Smart Forward Fill Strategy."""
    return MasterPOCSmartForwardFillStrategy()


# Convenience functions för bakåtkompatibilitet
def impute_with_master_poc(series: pd.Series, feature_name: str, 
                          time_index: Optional[pd.Series] = None) -> pd.Series:
    """
    Imputera serie med Master POC Smart Forward Fill.
    
    Args:
        series: Serie att imputera
        feature_name: Namn på feature
        time_index: Tidsserie för tidsbaserad imputation
        
    Returns:
        Imputerad serie enligt Master POC specifikation
    """
    strategy = create_master_poc_smart_forward_fill()
    return strategy.impute(series, feature_name=feature_name, time_index=time_index)


def get_master_poc_defaults() -> Dict[str, float]:
    """Hämta alla Master POC default-värden."""
    strategy = create_master_poc_smart_forward_fill()
    return strategy.get_feature_defaults()
