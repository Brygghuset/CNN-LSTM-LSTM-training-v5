"""
Master POC Preprocessing Pipeline för CNN-LSTM-LSTM modell.

Denna pipeline implementerar en komplett preprocessing pipeline enligt Master POC specifikationer:
- 16 timeseries features + 6 static features = 22 total input
- 8 output predictions (3 drugs + 5 ventilator)
- Master POC Smart Forward Fill imputation
- Unified Normalization [-1, 1]
- Standardiserad feature mapping
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Projekt imports
from data.preprocessing.data_preprocessor import DataPreprocessor
from data.preprocessing.imputation_methods import ImputationMethod
from data.unified_normalization import UnifiedNormalizer
from data.static_feature_normalizer import StaticFeatureNormalizer
from data.master_normalizer import MasterNormalizer
from data.master_poc_unit_conversion import MasterPOCUnitConverter, convert_vitaldb_units_master_poc
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MasterPOCPreprocessingConfig:
    """Konfiguration för Master POC preprocessing pipeline."""
    
    # Input/Output struktur enligt Master POC
    timeseries_features: int = 16  # 7 vital + 3 drugs + 6 ventilator
    static_features: int = 6        # age, sex, height, weight, bmi, asa
    total_input_features: int = 22  # 16 timeseries + 6 static
    output_features: int = 8        # 3 drugs + 5 ventilator
    
    # Window parametrar
    window_size: int = 300
    step_size: int = 30
    
    # Normalization
    target_range: Tuple[float, float] = (-1, 1)
    
    # Imputation
    imputation_method: str = "master_poc_smart_forward_fill"
    
    # Validering
    min_required_features: int = 12  # Minst 12 av 16 timeseries features
    min_samples_per_case: int = 300  # Minst 300 sampel per case
    
    # Output paths
    output_path: str = "data/processed/master_poc"
    metadata_path: str = "data/metadata/master_poc"


class MasterPOCPreprocessingPipeline:
    """
    Master POC Preprocessing Pipeline för CNN-LSTM-LSTM modell.
    
    Implementerar komplett preprocessing enligt Master POC specifikationer.
    """
    
    def __init__(self, config: Optional[MasterPOCPreprocessingConfig] = None, enable_s3: bool = True, s3_bucket: Optional[str] = None):
        """
        Initialisera Master POC preprocessing pipeline.
        
        Args:
            config: Konfiguration för pipeline (optional)
            enable_s3: Om S3-support ska aktiveras för vital data loading
            s3_bucket: S3 bucket namn för vital data (om None, använd default)
        """
        self.config = config or MasterPOCPreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        self.enable_s3 = enable_s3
        self.s3_bucket = s3_bucket
        
        # Initialisera komponenter
        self._initialize_components()
        
        # Master POC Unit Converter
        self.unit_converter = MasterPOCUnitConverter()
        
        # Master POC feature mapping
        self.master_poc_feature_mapping = self._get_master_poc_feature_mapping()
        
        # Data loader med S3-support
        self.data_loader = None
        self._initialize_data_loader()
        
        # Expected features enligt Master POC
        self.expected_timeseries_features = [
            # Vital Signs (7 features)
            'HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS',
            # Drug Infusions (3 features)
            'Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF',
            # Ventilator Settings (6 features)
            'TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev'
        ]
        
        self.expected_static_features = [
            'age', 'sex', 'height', 'weight', 'bmi', 'asa'
        ]
        
        self.expected_output_features = [
            # Drug Predictions (3)
            'Propofol_Predict', 'Remifentanil_Predict', 'Noradrenalin_Predict',
            # Ventilator Predictions (5)
            'TV_Predict', 'PEEP_Predict', 'FIO2_Predict', 'RR_Predict', 'etSEV_Predict'
        ]
        
        self.logger.info("Master POC Preprocessing Pipeline initialiserad")
        self.logger.info(f"Timeseries features: {self.config.timeseries_features}")
        self.logger.info(f"Static features: {self.config.static_features}")
        self.logger.info(f"Output features: {self.config.output_features}")
    
    def _initialize_components(self):
        """Initialisera preprocessing komponenter."""
        try:
            # DataPreprocessor med Master POC Smart Forward Fill
            self.data_preprocessor = DataPreprocessor(validate_physiological=True)
            
            # MasterNormalizer för Unified Normalization
            self.master_normalizer = MasterNormalizer()
            
            self.logger.info("Preprocessing komponenter initialiserade")
            
        except Exception as e:
            self.logger.error(f"Fel vid initialisering av komponenter: {e}")
            raise
    
    def _initialize_data_loader(self):
        """Initialisera data loader med S3-support."""
        try:
            from data.data_loader import VitalDBDataLoader
            self.data_loader = VitalDBDataLoader(enable_s3=self.enable_s3, s3_bucket=self.s3_bucket)
            self.logger.info(f"Data loader initialiserad med S3-support: {self.enable_s3}")
            if self.s3_bucket:
                self.logger.info(f"S3 bucket: {self.s3_bucket}")
        except Exception as e:
            self.logger.error(f"Fel vid initialisering av data loader: {e}")
            raise
    
    def _get_master_poc_feature_mapping(self) -> Dict[str, List[str]]:
        """Hämta Master POC feature mapping."""
        return {
            # Vital Signs (7 features)
            'HR': ['Solar8000/HR', 'HeartRate', 'HR', 'Pulse'],
            'BP_SYS': ['Solar8000/ART_SBP', 'Solar8000/FEM_SBP', 'Solar8000/PA_SBP', 'Solar8000/NIBP_SBP', 'SystolicBP', 'SBP'],
            'BP_DIA': ['Solar8000/ART_DBP', 'Solar8000/FEM_DBP', 'Solar8000/PA_DBP', 'Solar8000/NIBP_DBP', 'DiastolicBP', 'DBP'],
            'BP_MAP': ['Solar8000/ART_MBP', 'Solar8000/FEM_MAP', 'Solar8000/PA_MAP', 'Solar8000/NIBP_MBP', 'EV1000/ART_MBP', 'MeanBP', 'MBP'],
            'SPO2': ['Solar8000/PLETH_SPO2', 'SPO2', 'SPO2', 'SaO2'],
            'ETCO2': ['Solar8000/ETCO2', 'ETCO2', 'EndTidalCO2'],
            'BIS': ['BIS/BIS', 'BIS', 'BispectralIndex'],
            
            # Drug Infusions (3 features)
            'Propofol_INF': ['Orchestra/PPF20_RATE', 'Propofol_Rate', 'PPF20'],
            'Remifentanil_INF': ['Orchestra/RFTN20_RATE', 'Orchestra/RFTN50_RATE', 'Remifentanil_Rate'],
            'Noradrenalin_INF': ['Orchestra/NEPI_RATE', 'Norepinephrine_Rate', 'NEPI'],
            
            # Ventilator Settings (6 features)
            'TV': ['Solar8000/VENT_TV', 'Primus/TV', 'TidalVolume', 'VENT_TV'],
            'PEEP': ['Solar8000/VENT_MEAS_PEEP', 'Primus/PEEP_MBAR', 'PEEP', 'VENT_PEEP'],
            'FIO2': ['Solar8000/FIO2', 'Primus/FIO2', 'FiO2', 'VENT_FIO2'],
            'RR': ['Solar8000/VENT_RR', 'Primus/RR_CO2', 'Solar8000/RR', 'Solar8000/RR_CO2', 'RespiratoryRate', 'VENT_RR'],
            'etSEV': ['Primus/EXP_SEVO', 'Sevoflurane', 'SEVO', 'EXP_SEVO'],
            'inSev': ['Primus/INSP_SEVO', 'InspSevoflurane', 'INSP_SEVO']
        }
    
    def apply_master_poc_feature_mapping(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applicera Master POC feature mapping.
        
        Args:
            timeseries_df: Data från VitalDB (kan vara rådata eller efter unit conversion)
            
        Returns:
            DataFrame med Master POC standardiserade feature-namn
        """
        self.logger.info("Applicerar Master POC feature mapping")
        
        mapped_data = pd.DataFrame()
        available_columns = list(timeseries_df.columns)
        
        for target_feature, possible_tracks in self.master_poc_feature_mapping.items():
            found = False
            
            # Kontrollera först om target_feature redan finns (från unit conversion)
            if target_feature in available_columns:
                mapped_data[target_feature] = timeseries_df[target_feature]
                found = True
                self.logger.debug(f"Behåller befintlig kolumn: {target_feature}")
            else:
                # Annars leta efter möjliga källkolumner
                for track in possible_tracks:
                    if track in available_columns:
                        mapped_data[target_feature] = timeseries_df[track]
                        found = True
                        self.logger.debug(f"Mappade {track} -> {target_feature}")
                        break
            
            if not found:
                self.logger.warning(f"Ingen mappning hittades för {target_feature}")
                # Skapa kolumn med NaN-värden
                mapped_data[target_feature] = np.nan
        
        # Kontrollera vilka features som finns
        available_features = [f for f in self.expected_timeseries_features if f in mapped_data.columns]
        self.logger.info(f"Tillgängliga Master POC features: {len(available_features)}/{len(self.expected_timeseries_features)}")
        
        return mapped_data
    
    def load_case_data(self, case_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Ladda case data med automatisk fallback-hierarki (S3 -> SSD -> Local -> Mock).
        
        Args:
            case_id: Case identifier (t.ex. "0001")
            
        Returns:
            Tuple med (timeseries_df, clinical_df)
        """
        self.logger.info(f"Laddar case data för {case_id} med S3 fallback-hierarki")
        
        try:
            # Konvertera case_id till int för VitalDBDataLoader
            case_id_int = int(case_id)
            
            # Ladda data med automatisk fallback-hierarki
            timeseries_df, clinical_df = self.data_loader.load_vitaldb_case(case_id_int)
            
            if timeseries_df is not None:
                self.logger.info(f"Case {case_id} laddad: {timeseries_df.shape[0]} sampel, {timeseries_df.shape[1]} features")
            else:
                self.logger.warning(f"Ingen timeseries data för case {case_id}")
            
            if clinical_df is not None:
                self.logger.info(f"Clinical data för case {case_id}: {clinical_df.shape[0]} rader")
            else:
                self.logger.warning(f"Ingen clinical data för case {case_id}")
            
            return timeseries_df, clinical_df
            
        except Exception as e:
            self.logger.error(f"Fel vid laddning av case {case_id}: {e}")
            return None, None
    
    def load_raw_case_data(self, case_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Ladda rådata för case UTAN feature mapping - för unit conversion.
        
        Args:
            case_id: Case identifier (t.ex. "0001")
            
        Returns:
            Tuple med (raw_timeseries_df, clinical_df)
        """
        self.logger.info(f"Laddar rådata för {case_id} utan feature mapping")
        
        try:
            # Konvertera case_id till int för VitalDBDataLoader
            case_id_int = int(case_id)
            
            # Ladda rådata utan feature mapping
            raw_timeseries_df = self.data_loader._load_timeseries_data(case_id_int, apply_feature_mapping=False)
            clinical_df = self.data_loader._load_clinical_data(case_id_int)
            
            if raw_timeseries_df is not None:
                self.logger.info(f"Rådata för case {case_id} laddad: {len(raw_timeseries_df)} sampel, {len(raw_timeseries_df.columns)} features")
            else:
                self.logger.warning(f"Ingen rådata för case {case_id}")
            
            if clinical_df is not None:
                self.logger.info(f"Clinical data för case {case_id}: {len(clinical_df)} rader")
            else:
                self.logger.warning(f"Ingen clinical data för case {case_id}")
            
            return raw_timeseries_df, clinical_df
            
        except Exception as e:
            self.logger.error(f"Fel vid laddning av rådata för case {case_id}: {e}")
            return None, None
    
    def preprocess_case(self, case_id: str, timeseries_df: pd.DataFrame, 
                       clinical_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Preprocessa ett specifikt case enligt Master POC specifikation.
        
        Skapar ENDAST input features för modellen - inga targets/outputs.
        Targets skapas av modellen baserat på input features.
        
        Args:
            case_id: Case identifier
            timeseries_df: Timeseries data från VitalDB
            clinical_df: Clinical data från VitalDB
            
        Returns:
            Tuple med (timeseries_windows, static_features, metadata)
            - timeseries_windows: (n_windows, 300, 16) normaliserade timeseries features
            - static_features: (6,) normaliserade static patient features
            - metadata: Dict med preprocessing metadata
        """
        self.logger.info(f"Preprocessar case {case_id} enligt Master POC specifikation")
        
        try:
            # Steg 1: Ladda rådata för unit conversion
            self.logger.info("Laddar rådata för unit conversion")
            raw_timeseries_df, _ = self.load_raw_case_data(case_id)
            
            if raw_timeseries_df is None:
                self.logger.error(f"Kunde inte ladda rådata för case {case_id}")
                return None, None, {}
            
            # Steg 2: Master POC Unit Conversion (på rådata)
            self.logger.info("Applicerar Master POC Unit Conversion på rådata")
            # Extrahera patient weight från clinical data
            patient_weight = self._extract_patient_weight(clinical_df)
            
            # Konvertera enheter enligt Master POC specifikationer (på rådata)
            conversion_result = convert_vitaldb_units_master_poc(raw_timeseries_df, patient_weight)
            converted_data = conversion_result.converted_data
            
            # Steg 3: Master POC Feature Mapping (på konverterade data)
            mapped_data = self.apply_master_poc_feature_mapping(converted_data)
            
            # Kontrollera tillgängliga features
            available_features = [f for f in self.expected_timeseries_features if f in mapped_data.columns]
            
            if len(available_features) < self.config.min_required_features:
                self.logger.warning(f"För få features för case {case_id}: {len(available_features)}/{self.config.timeseries_features}")
                return None, None, None, {}
            
            # Logga conversion details
            if conversion_result.master_poc_compliant:
                self.logger.info(f"✅ Master POC unit conversion compliant för case {case_id}")
            else:
                self.logger.warning(f"⚠️ Master POC unit conversion issues för case {case_id}")
            
            # Steg 4: Master POC Smart Forward Fill Imputation
            self.logger.info("Applicerar Master POC Smart Forward Fill imputation")
            cleaned_data = self.data_preprocessor.impute_missing_values(
                mapped_data,
                method=ImputationMethod.MASTER_POC_SMART_FORWARD_FILL,
                columns=available_features
            )
            
            # Ta bort rader med för många NaN-värden
            cleaned_data = cleaned_data.dropna(thresh=len(cleaned_data.columns) * 0.7)
            
            if len(cleaned_data) < self.config.min_samples_per_case:
                self.logger.warning(f"För lite data för case {case_id} efter cleaning: {len(cleaned_data)}")
                return None, None, {}
            
            # Steg 4: Skapa sliding windows (bara timeseries windows, inga targets)
            timeseries_windows = self._create_master_poc_sliding_windows(cleaned_data, available_features)
            
            if len(timeseries_windows) == 0:
                self.logger.warning(f"Inga giltiga windows skapades för case {case_id}")
                return None, None, {}
            
            # Steg 5: Master POC Unified Normalization
            self.logger.info("Applicerar Master POC Unified Normalization")
            normalized_timeseries_windows = self.master_normalizer.normalize_timeseries_data(
                pd.DataFrame(timeseries_windows.reshape(-1, timeseries_windows.shape[-1]), columns=available_features)
            ).values.reshape(timeseries_windows.shape)
            
            # Steg 6: Extrahera och normalisera static features (StaticFeatureNormalizer)
            normalized_static_features = self.master_normalizer.extract_and_normalize_static_features(clinical_df)
            self.logger.info(f"Static features extraherade och normaliserade: {normalized_static_features.shape}")
            
            # Steg 7: Targets skapas INTE i preprocessing - de kommer från modellen
            # Preprocessing skapar ENDAST input features för modellen
            
            # Metadata
            metadata = {
                'case_id': case_id,
                'original_samples': len(timeseries_df),
                'cleaned_samples': len(cleaned_data),
                'timeseries_windows_created': len(timeseries_windows),
                'available_features': available_features,
                'static_features': self.expected_static_features,
                'target_features': 'N/A - targets skapas av modellen',
                'window_size': self.config.window_size,
                'step_size': self.config.step_size,
                'preprocessing_method': 'master_poc',
                'patient_weight_used': conversion_result.patient_weight_used,
                'unit_conversion_compliant': conversion_result.master_poc_compliant,
                'conversion_details': conversion_result.conversion_details,
                'failed_conversions': conversion_result.failed_conversions,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Case {case_id} preprocessat: {len(timeseries_windows)} timeseries windows, {len(normalized_static_features)} static features")
            return normalized_timeseries_windows, normalized_static_features, metadata
            
        except Exception as e:
            self.logger.error(f"Fel vid preprocessing av case {case_id}: {e}")
            return None, None, {}
    
    def _create_master_poc_sliding_windows(self, data: pd.DataFrame, 
                                          feature_columns: List[str]) -> np.ndarray:
        """
        Skapa sliding windows enligt Master POC specifikation.
        
        Args:
            data: Cleaned data
            feature_columns: Tillgängliga feature-kolumner (16 timeseries features)
            
        Returns:
            np.ndarray med shape (n_windows, window_size, n_features)
        """
        self.logger.info(f"Skapar sliding windows: window_size={self.config.window_size}, step_size={self.config.step_size}")
        
        windows = []
        
        for i in range(0, len(data) - self.config.window_size + 1, self.config.step_size):
            # Skapa window
            window_data = data.iloc[i:i + self.config.window_size]
            
            # Kontrollera att window har tillräckligt med data
            if len(window_data) < self.config.window_size:
                continue
            
            # Kontrollera att window inte har för många NaN-värden
            nan_count = window_data[feature_columns].isna().sum().sum()
            if nan_count > len(feature_columns) * self.config.window_size * 0.3:  # Max 30% NaN
                continue
            
            # Skapa window array med 16 timeseries features
            window_array = window_data[feature_columns].values
            
            windows.append(window_array)
        
        if len(windows) == 0:
            self.logger.warning("Inga giltiga windows skapades")
            return np.array([])
        
        windows_array = np.array(windows)
        
        self.logger.info(f"Skapade {len(windows)} windows med shape {windows_array.shape}")
        self.logger.info(f"Window shape: ({self.config.window_size} sekunder, {len(feature_columns)} timeseries features)")
        
        return windows_array
    
    def validate_master_poc_compliance(self, windows: np.ndarray, targets: Optional[np.ndarray], 
                                      static_features: np.ndarray) -> Dict[str, Any]:
        """
        Validera att preprocessad data följer Master POC specifikationer.
        
        Args:
            windows: Timeseries windows
            targets: Target values (kan vara None eftersom targets skapas av modellen)
            static_features: Static patient features
            
        Returns:
            Dict med valideringsresultat
        """
        validation_result = {
            'compliance': True,
            'issues': [],
            'shapes': {},
            'ranges': {}
        }
        
        # Validera shapes
        if len(windows.shape) == 3:
            validation_result['shapes']['windows'] = windows.shape
            if windows.shape[1] != self.config.window_size:
                validation_result['compliance'] = False
                validation_result['issues'].append(f"Window size mismatch: {windows.shape[1]} != {self.config.window_size}")
            
            if windows.shape[2] != self.config.timeseries_features:
                validation_result['compliance'] = False
                validation_result['issues'].append(f"Timeseries features mismatch: {windows.shape[2]} != {self.config.timeseries_features}")
        else:
            validation_result['compliance'] = False
            validation_result['issues'].append(f"Invalid windows shape: {windows.shape}")
        
        # Targets valideras inte eftersom de skapas av modellen
        if targets is not None:
            validation_result['shapes']['targets'] = targets.shape
            validation_result['issues'].append("WARNING: Targets skapas av modellen, inte i preprocessing")
        else:
            validation_result['shapes']['targets'] = 'N/A - skapas av modellen'
        
        if len(static_features.shape) == 1:
            validation_result['shapes']['static_features'] = static_features.shape
            if static_features.shape[0] != self.config.static_features:
                validation_result['compliance'] = False
                validation_result['issues'].append(f"Static features mismatch: {static_features.shape[0]} != {self.config.static_features}")
        else:
            validation_result['compliance'] = False
            validation_result['issues'].append(f"Invalid static features shape: {static_features.shape}")
        
        # Validera ranges (normaliserade värden ska vara i [-1, 1])
        if len(windows) > 0:
            window_min, window_max = windows.min(), windows.max()
            validation_result['ranges']['windows'] = {'min': window_min, 'max': window_max}
            if window_min < -1.1 or window_max > 1.1:  # Liten tolerans
                validation_result['compliance'] = False
                validation_result['issues'].append(f"Windows out of range [-1, 1]: [{window_min:.3f}, {window_max:.3f}]")
        
        # Targets valideras inte eftersom de skapas av modellen
        if targets is not None:
            validation_result['ranges']['targets'] = 'N/A - skapas av modellen'
        else:
            validation_result['ranges']['targets'] = 'N/A - skapas av modellen'
        
        if len(static_features) > 0:
            static_min, static_max = static_features.min(), static_features.max()
            validation_result['ranges']['static_features'] = {'min': static_min, 'max': static_max}
            if static_min < -1.1 or static_max > 1.1:  # Liten tolerans
                validation_result['compliance'] = False
                validation_result['issues'].append(f"Static features out of range [-1, 1]: [{static_min:.3f}, {static_max:.3f}]")
        
        return validation_result
    
    def _extract_patient_weight(self, clinical_df: pd.DataFrame) -> float:
        """
        Extrahera patient weight från clinical data med Master POC fallback.
        
        Args:
            clinical_df: Clinical data från VitalDB
            
        Returns:
            Patient weight i kg
        """
        if clinical_df is None or clinical_df.empty:
            self.logger.warning("Ingen clinical data tillgänglig, använder Master POC default weight")
            return self.unit_converter.get_master_poc_default_weight()
        
        if 'weight' in clinical_df.columns:
            weight_value = clinical_df['weight'].iloc[0] if len(clinical_df) > 0 else None
            if weight_value is not None and weight_value > 0:
                self.logger.info(f"Använder patient weight: {weight_value} kg")
                return float(weight_value)
            else:
                self.logger.warning(f"Ogiltig patient weight: {weight_value}, använder Master POC default")
        else:
            self.logger.warning("Patient weight kolumn saknas, använder Master POC default weight")
        
        # Master POC fallback weight
        default_weight = self.unit_converter.get_master_poc_default_weight()
        self.logger.info(f"Använder Master POC default weight: {default_weight} kg")
        return default_weight


def create_master_poc_preprocessing_pipeline(config: Optional[MasterPOCPreprocessingConfig] = None) -> MasterPOCPreprocessingPipeline:
    """Factory function för att skapa Master POC preprocessing pipeline."""
    return MasterPOCPreprocessingPipeline(config)


# Convenience functions för bakåtkompatibilitet
def preprocess_case_master_poc(case_id: str, timeseries_df: pd.DataFrame, 
                              clinical_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                                Optional[np.ndarray], Dict]:
    """
    Preprocessa case med Master POC pipeline.
    
    Args:
        case_id: Case identifier
        timeseries_df: Timeseries data från VitalDB
        clinical_df: Clinical data från VitalDB
        
    Returns:
        Tuple med (windows, targets, static_features, metadata)
    """
    pipeline = create_master_poc_preprocessing_pipeline()
    return pipeline.preprocess_case(case_id, timeseries_df, clinical_df)


def validate_master_poc_data(windows: np.ndarray, targets: np.ndarray, 
                           static_features: np.ndarray) -> Dict[str, Any]:
    """
    Validera Master POC compliance för preprocessad data.
    
    Args:
        windows: Timeseries windows
        targets: Target values
        static_features: Static patient features
        
    Returns:
        Dict med valideringsresultat
    """
    pipeline = create_master_poc_preprocessing_pipeline()
    return pipeline.validate_master_poc_compliance(windows, targets, static_features)
