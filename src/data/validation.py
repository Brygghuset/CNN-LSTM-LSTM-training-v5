import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

# from domain.services import MedicalConfig  # KOMMENTERAD BORT - inte kritisk för AWS körning

logger = logging.getLogger(__name__)


class DataValidator:
    """Validerar data för klinisk användning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validera data och returnera valideringsresultat."""
        log = {}
        
        # Grundläggande validering
        log['shape'] = df.shape
        log['columns'] = list(df.columns)
        
        # Kontrollera för saknade värden
        log['missing_values'] = df.isna().sum().to_dict()
        
        # Kontrollera datatyper
        log['dtypes'] = df.dtypes.to_dict()
        
        # Kontrollera för duplicerade rader
        log['duplicates'] = int(df.duplicated().sum())
        
        # Kontrollera temporal ordning om Time finns
        if 'Time' in df.columns:
            log['temporal_order'] = df['Time'].is_monotonic_increasing
        
        return log


class CleaningStep(ABC):
    """Abstract base class för data cleaning steg."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Processera data och returnera (processed_df, log_dict).
        
        Args:
            df: DataFrame att processera
            
        Returns:
            Tuple av (processed_df, log_dict)
        """
        pass


class ImputationProcessor(CleaningStep):
    """Hanterar missing value imputation med detaljerad logging."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Imputera saknade värden med multi-step process."""
        from data.preprocessing import impute_missing_values
        
        log = {}
        
        # Identifiera numeriska kolumner (exklusive Time)
        numeric_cols = self._get_numeric_columns(df)
        
        # Räkna initiala NaN-värden
        total_nans = df[numeric_cols].isna().sum().sum()
        log['total'] = int(total_nans)
        
        if total_nans == 0:
            log.update({'forward_fill': 0, 'backward_fill': 0, 'mean': 0, 'remaining': 0})
            return df.copy(), log
        
        self.logger.info(f"Börjar imputering av {total_nans} saknade värden")
        
        # Steg 1: Forward fill
        imputed, ffill_count = self._apply_forward_fill(df, numeric_cols)
        log['forward_fill'] = ffill_count
        
        # Steg 2: Backward fill fallback
        imputed, bfill_count = self._apply_backward_fill(imputed, numeric_cols)
        log['backward_fill'] = bfill_count
        
        # Steg 3: Mean fallback
        imputed, mean_count = self._apply_mean_fill(imputed, numeric_cols)
        log['mean'] = mean_count
        
        # Räkna kvarvarande NaN
        remaining_nans = imputed[numeric_cols].isna().sum().sum()
        log['remaining'] = int(remaining_nans)
        
        self.logger.info(f"Imputering slutförd: {total_nans} → {remaining_nans} saknade värden")
        
        return imputed, log
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Hämta numeriska kolumner exklusive Time."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Time' in numeric_cols:
            numeric_cols.remove('Time')
        return numeric_cols
    
    def _apply_forward_fill(self, df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, int]:
        """Applicera forward fill och räkna fyllda värden."""
        from data.preprocessing import impute_missing_values
        
        nans_before = df[numeric_cols].isna().sum().sum()
        imputed = impute_missing_values(df, method='forward_fill', columns=numeric_cols, validate_physiological=False)
        nans_after = imputed[numeric_cols].isna().sum().sum()
        
        filled_count = int(nans_before - nans_after)
        return imputed, filled_count
    
    def _apply_backward_fill(self, df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, int]:
        """Applicera backward fill och räkna fyllda värden."""
        from data.preprocessing import impute_missing_values
        
        nans_before = df[numeric_cols].isna().sum().sum()
        imputed = impute_missing_values(df, method='backward_fill', columns=numeric_cols, validate_physiological=False)
        nans_after = imputed[numeric_cols].isna().sum().sum()
        
        filled_count = int(nans_before - nans_after)
        return imputed, filled_count
    
    def _apply_mean_fill(self, df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, int]:
        """Applicera mean fill och räkna fyllda värden."""
        from data.preprocessing import impute_missing_values
        
        nans_before = df[numeric_cols].isna().sum().sum()
        imputed = impute_missing_values(df, method='mean', columns=numeric_cols, validate_physiological=False)
        nans_after = imputed[numeric_cols].isna().sum().sum()
        
        filled_count = int(nans_before - nans_after)
        return imputed, filled_count


class OutlierDetector(CleaningStep):
    """Detekterar och hanterar fysiologiska outliers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.outlier_rules = self._initialize_outlier_rules()
    
    def _initialize_outlier_rules(self) -> Dict[str, Dict]:
        """Initialisera regler för outlier-detection."""
        return {
            'Solar8000/HR': {'min': 20, 'max': 300, 'name': 'HR'},
            'Solar8000/PLETH_SPO2': {'min': 70, 'max': 100, 'name': 'SPO2'},
            'Solar8000/ETCO2': {'min': 10, 'max': 80, 'name': 'ETCO2'},
            'Solar8000/NBP_SYS': {'min': 40, 'max': 250, 'name': 'NBP'}
        }
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detektera och ta bort outliers."""
        log = {}
        outlier_counts = {}
        outlier_total = 0
        
        cleaned = df.copy()
        
        # Detektera outliers för varje parameter
        for column, rules in self.outlier_rules.items():
            if column in cleaned.columns:
                count = self._count_outliers(cleaned[column], rules['min'], rules['max'])
                outlier_counts[rules['name']] = int(count)
                outlier_total += count
        
        log = {'total': int(outlier_total), **outlier_counts}
        
        if outlier_total > 0:
            self.logger.info(f"Hittade {outlier_total} outliers, flaggar som NaN")
            
            # Ta bort outliers
            cleaned = self._remove_outliers(cleaned)
            
            # Re-imputera efter outlier removal
            cleaned = self._reimpute_after_outliers(cleaned)
        
        return cleaned, log
    
    def _count_outliers(self, series: pd.Series, min_val: float, max_val: float) -> int:
        """Räkna antal outliers i en serie."""
        return ((series < min_val) | (series > max_val)).sum()
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ta bort outliers genom att flagga som NaN."""
        cleaned = df.copy()
        
        for column, rules in self.outlier_rules.items():
            if column in cleaned.columns:
                series = cleaned[column]
                mask = (series >= rules['min']) & (series <= rules['max'])
                cleaned[column] = series.where(mask, np.nan)
        
        return cleaned
    
    def _reimpute_after_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-imputera efter outlier removal."""
        from data.preprocessing import impute_missing_values
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Time' in numeric_cols:
            numeric_cols.remove('Time')
        
        return impute_missing_values(df, method='mean', columns=numeric_cols, validate_physiological=False)


class TemporalCorrector(CleaningStep):
    """Korrigerar temporal ordning i data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Korrigera temporal ordning."""
        log = {}
        
        if 'Time' not in df.columns:
            self.logger.warning("Ingen Time-kolumn hittad för temporal korrektion")
            log['corrections'] = 0
            return df.copy(), log
        
        # Räkna antal icke-monotona hopp
        time_col = df['Time']
        non_monotonic = np.sum(np.diff(time_col) <= 0)
        log['corrections'] = int(non_monotonic)
        
        if non_monotonic > 0:
            self.logger.info(f"Korrigerar {non_monotonic} temporal disorder")
            corrected = self._ensure_temporal_order(df)
        else:
            corrected = df.reset_index(drop=True)
        
        # Validera att korrektion fungerade
        assert corrected['Time'].is_monotonic_increasing, "Temporal korrektion misslyckades"
        
        return corrected, log
    
    def _ensure_temporal_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Säkerställ att DataFrame är sorterad i strikt stigande ordning."""
        if not df['Time'].is_monotonic_increasing:
            df_sorted = df.sort_values('Time', kind='mergesort').reset_index(drop=True)
            return df_sorted
        return df.reset_index(drop=True)


class DataCleaningPipeline:
    """
    Refaktorerad data cleaning pipeline med separerade ansvarsområden.
    
    Använder Chain of Responsibility pattern för att köra cleaning-steg i ordning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialisera cleaning-steg i korrekt ordning
        self.steps = [
            ('imputation', ImputationProcessor()),
            ('outliers', OutlierDetector()),
            ('temporal', TemporalCorrector())
        ]
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Kör hela cleaning pipeline.
        
        Args:
            df: DataFrame att rensa
            
        Returns:
            Tuple av (cleaned_df, complete_log)
        """
        self.logger.info(f"Börjar data cleaning pipeline för {len(df)} rader")
        
        complete_log = {}
        current_df = df.copy()
        
        # Kör varje steg i ordning
        for step_name, processor in self.steps:
            self.logger.debug(f"Kör {step_name} processor")
            current_df, step_log = processor.process(current_df)
            complete_log[step_name] = step_log
        
        self.logger.info("Data cleaning pipeline slutförd")
        
        return current_df, complete_log


# Bakåtkompatibel funktion för befintlig kod
def data_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Bakåtkompatibel wrapper för refaktorerad pipeline.
    
    DEPRECATED: Använd DataCleaningPipeline direkt för nya implementationer.
    """
    pipeline = DataCleaningPipeline()
    cleaned_df, new_log = pipeline.clean_data(df)
    
    # Mappa om loggen till det gamla formatet för bakåtkompatibilitet
    old_format_log = {
        'missing': new_log.get('imputation', {}),
        'outliers': new_log.get('outliers', {}),
        'temporal': new_log.get('temporal', {})
    }
    
    return cleaned_df, old_format_log


# Befintliga funktioner behålls för bakåtkompatibilitet
def validate_clinical_ranges(df):
    """
    Validera att varje parameter i df ligger inom kliniska säkerhetsintervall enligt centraliserad konfiguration.
    Returnerar en dict: {param: {valid: bool, min: float, max: float, value: float}}
    """
    from config import get_safety_limits_dict
    limits = get_safety_limits_dict()
    results = {}
    for col in df.columns:
        if col in limits:
            min_val = limits[col]["min"]
            max_val = limits[col]["max"]
            value = float(df[col].iloc[0]) if len(df[col]) > 0 else np.nan
            valid = (min_val <= value <= max_val)
            results[col] = {"valid": valid, "min": min_val, "max": max_val, "value": value}
    return results


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flagga fysiologiskt omöjliga värden som NaN enligt medicinska gränser.
    
    DEPRECATED: Använd OutlierDetector för nya implementationer.
    """
    # Använd original logik för bakåtkompatibilitet utan re-imputation
    cleaned = df.copy()
    # HR: 20-300 BPM
    if 'Solar8000/HR' in cleaned.columns:
        hr = cleaned['Solar8000/HR']
        cleaned['Solar8000/HR'] = hr.where((hr >= 20) & (hr <= 300), np.nan)
    # SPO2: 70-100%
    if 'Solar8000/PLETH_SPO2' in cleaned.columns:
        spo2 = cleaned['Solar8000/PLETH_SPO2']
        cleaned['Solar8000/PLETH_SPO2'] = spo2.where((spo2 >= 70) & (spo2 <= 100), np.nan)
    # ETCO2: 10-80 mmHg (0 och >80 omöjligt)
    if 'Solar8000/ETCO2' in cleaned.columns:
        etco2 = cleaned['Solar8000/ETCO2']
        cleaned['Solar8000/ETCO2'] = etco2.where((etco2 >= 10) & (etco2 <= 80), np.nan)
    return cleaned


def ensure_temporal_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att DataFrame är sorterad i strikt stigande ordning enligt Time-kolumnen.
    
    DEPRECATED: Använd TemporalCorrector för nya implementationer.
    """
    corrector = TemporalCorrector()
    corrected_df, _ = corrector.process(df)
    return corrected_df


def ensure_consistent_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att alla parametrar har samma längd genom att trunkera till kortaste gemensamma längden.
    Trunkerar från slutet av längre parametrar för att behålla tidigare data.
    """
    # Hitta kortaste giltiga längden för varje kolumn (exklusive NaN i slutet)
    min_lengths = []
    for col in df.columns:
        if col == 'Time':
            continue
        # Hitta sista giltiga (icke-NaN) index för varje kolumn
        series = df[col].dropna()
        if len(series) > 0:
            min_lengths.append(len(series))
        else:
            min_lengths.append(0)
    
    # Hitta kortaste gemensamma längden
    if not min_lengths:
        return df
    
    target_length = min(min_lengths)
    
    if target_length == 0:
        # Alla kolumner är tomma, returnera tom DataFrame
        return df.iloc[:0]
    
    # Trunkera alla kolumner till target_length
    harmonized_df = df.iloc[:target_length].copy()
    
    return harmonized_df
