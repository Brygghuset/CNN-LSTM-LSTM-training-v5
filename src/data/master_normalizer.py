"""
Master Normalizer enligt Master POC CNN-LSTM-LSTM specifikation.

Kombinerar alla normalizers för en komplett normaliserings-pipeline:
- Unified Normalization för timeseries features
- Static Feature Normalization för patient features  
- Output Denormalization för predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union, List
import logging

from data.unified_normalization import UnifiedNormalizer, create_unified_normalizer
from data.static_feature_normalizer import StaticFeatureNormalizer, create_static_normalizer
from data.output_denormalizer import OutputDenormalizer, create_output_denormalizer

logger = logging.getLogger(__name__)


class MasterNormalizer:
    """
    Master Normalizer som kombinerar alla normalizers enligt Master POC.
    
    Hanterar komplett normalisering av:
    - 16 timeseries features → [-1, 1] med kliniska ranges
    - 6 static patient features → [-1, 1] med specifika formler
    - 8 output predictions → denormalisering till kliniska enheter
    """
    
    def __init__(self):
        """Initialisera Master Normalizer med alla sub-normalizers."""
        self.unified_normalizer = create_unified_normalizer()
        self.static_normalizer = create_static_normalizer()
        self.output_denormalizer = create_output_denormalizer()
        
        # Master POC feature ordning
        self.timeseries_features = [
            # Vital Signs (7 features)
            'HR', 'BP_SYS', 'BP_DIA', 'BP_MAP', 'SPO2', 'ETCO2', 'BIS',
            # Drug Infusions (3 features)
            'Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF',
            # Ventilator Settings (6 features)
            'TV', 'PEEP', 'FIO2', 'RR', 'etSEV', 'inSev'
        ]
        
        self.static_features = [
            'age', 'sex', 'height', 'weight', 'bmi', 'asa'
        ]
        
        self.output_predictions = [
            'Propofol_Predict', 'Remifentanil_Predict', 'Noradrenalin_Predict',
            'TV_Predict', 'PEEP_Predict', 'FIO2_Predict', 'RR_Predict', 'etSEV_Predict'
        ]
        
        logger.info("Master Normalizer initialiserad med Master POC specifikationer")
    
    def normalize_timeseries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalisera timeseries data med Unified Normalization enligt Master POC.
        
        Filtrera till ENDAST de 16 Master POC timeseries features och normalisera dessa.
        Alla andra features ignoreras för att säkerställa Master POC compliance.
        
        Args:
            df: DataFrame med timeseries features (kan innehålla många fler än 16)
            
        Returns:
            DataFrame med ENDAST 16 normaliserade Master POC timeseries features
        """
        # Filtrera till ENDAST Master POC timeseries features som finns i data
        available_features = [f for f in self.timeseries_features if f in df.columns]
        
        if not available_features:
            logger.warning("Inga Master POC timeseries features hittades i data")
            # Skapa tom DataFrame med Master POC features och default-värden
            empty_master_poc = pd.DataFrame(columns=self.timeseries_features)
            return empty_master_poc
        
        logger.info(f"Master POC Alternativ 2: Filtrerar från {len(df.columns)} kolumner till {len(available_features)} Master POC features")
        logger.info(f"Tillgängliga Master POC features: {available_features}")
        
        # Skapa DataFrame med ENDAST Master POC features
        master_poc_df = df[available_features].copy()
        
        # Normalisera ENDAST dessa Master POC features
        normalized_master_poc = self.unified_normalizer.normalize_dataframe(master_poc_df, available_features)
        
        logger.info(f"✅ Master POC normalisering klar: {len(normalized_master_poc.columns)} features normaliserade till [-1, 1]")
        
        # Returnera ENDAST de normaliserade Master POC features
        return normalized_master_poc
    
    def denormalize_timeseries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalisera timeseries data tillbaka till kliniska enheter enligt Master POC.
        
        Args:
            df: DataFrame med normaliserade Master POC timeseries features
            
        Returns:
            DataFrame med denormaliserade Master POC timeseries features
        """
        available_features = [f for f in self.timeseries_features if f in df.columns]
        
        if not available_features:
            logger.warning("Inga Master POC timeseries features hittades för denormalisering")
            return df.copy()
        
        logger.info(f"Denormaliserar {len(available_features)} Master POC timeseries features")
        
        return self.unified_normalizer.denormalize_dataframe(df, available_features)
    
    def normalize_static_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalisera static patient features med specifika Master POC formler.
        
        Args:
            df: DataFrame med static patient features
            
        Returns:
            DataFrame med normaliserade static features
        """
        available_features = [f for f in self.static_features if f in df.columns]
        
        if not available_features:
            logger.warning("Inga kända static features hittades")
            return df.copy()
        
        logger.info(f"Normaliserar {len(available_features)} static features med Master POC formler")
        
        return self.static_normalizer.normalize_static_features(df)
    
    def denormalize_static_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalisera static patient features tillbaka till ursprungliga enheter.
        
        Args:
            df: DataFrame med normaliserade static features
            
        Returns:
            DataFrame med denormaliserade static features
        """
        available_features = [f for f in self.static_features if f in df.columns]
        
        if not available_features:
            logger.warning("Inga kända static features hittades för denormalisering")
            return df.copy()
        
        logger.info(f"Denormaliserar {len(available_features)} static features")
        
        return self.static_normalizer.denormalize_static_features(df)
    
    def extract_and_normalize_static_features(self, clinical_df: Optional[pd.DataFrame]) -> np.ndarray:
        """
        Extrahera och normalisera static features från clinical DataFrame.
        
        Args:
            clinical_df: DataFrame med kliniska data
            
        Returns:
            Numpy array med 6 normaliserade static features [age, sex, height, weight, bmi, asa]
        """
        return self.static_normalizer.extract_and_normalize_static_features(clinical_df)
    
    def denormalize_predictions(self, predictions: Union[np.ndarray, Dict, pd.DataFrame]) -> Union[np.ndarray, Dict, pd.DataFrame]:
        """
        Denormalisera model predictions till kliniska enheter.
        
        Args:
            predictions: Normaliserade predictions i [-1, 1]
            
        Returns:
            Denormaliserade predictions i kliniska enheter
        """
        if isinstance(predictions, np.ndarray):
            logger.info(f"Denormaliserar predictions array med shape {predictions.shape}")
            return self.output_denormalizer.denormalize_predictions_array(predictions)
        elif isinstance(predictions, dict):
            logger.info(f"Denormaliserar predictions dictionary med {len(predictions)} predictions")
            return self.output_denormalizer.denormalize_predictions_dict(predictions)
        elif isinstance(predictions, pd.DataFrame):
            logger.info(f"Denormaliserar predictions DataFrame med shape {predictions.shape}")
            return self.output_denormalizer.denormalize_predictions_dataframe(predictions)
        else:
            raise ValueError(f"Unsupported predictions type: {type(predictions)}")
    
    def normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Normalisera target values för training.
        
        Args:
            targets: Target values i kliniska enheter
            
        Returns:
            Normaliserade targets i [-1, 1]
        """
        from data.output_denormalizer import normalize_target_outputs
        logger.info(f"Normaliserar targets med shape {targets.shape}")
        return normalize_target_outputs(targets)
    
    def process_complete_dataset(self, 
                                timeseries_df: pd.DataFrame, 
                                clinical_df: Optional[pd.DataFrame] = None,
                                targets_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
        """
        Komplett processning av ett dataset enligt Master POC.
        
        Args:
            timeseries_df: DataFrame med timeseries data
            clinical_df: DataFrame med kliniska data (för static features)
            targets_df: DataFrame med target values (optional)
            
        Returns:
            Tuple av (normaliserade timeseries, static features array, normaliserade targets)
        """
        logger.info("Startar komplett dataset-processning enligt Master POC")
        
        # Normalisera timeseries data
        normalized_timeseries = self.normalize_timeseries_data(timeseries_df)
        
        # Extrahera och normalisera static features
        static_features = self.extract_and_normalize_static_features(clinical_df)
        
        # Normalisera targets om de finns
        normalized_targets = None
        if targets_df is not None:
            # Konvertera targets till array och normalisera
            targets_array = targets_df[self.output_predictions].values if isinstance(targets_df, pd.DataFrame) else targets_df
            normalized_targets = self.normalize_targets(targets_array)
        
        logger.info(f"Dataset-processning klar: timeseries {normalized_timeseries.shape}, static {static_features.shape}")
        
        return normalized_timeseries, static_features, normalized_targets
    
    def validate_all_data(self, 
                         timeseries_df: pd.DataFrame, 
                         clinical_df: Optional[pd.DataFrame] = None,
                         predictions: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Dict[str, Dict]:
        """
        Validera all data enligt Master POC ranges.
        
        Args:
            timeseries_df: DataFrame med timeseries data
            clinical_df: DataFrame med kliniska data
            predictions: Model predictions att validera
            
        Returns:
            Dict med valideringsresultat för alla data-typer
        """
        validation_results = {
            'timeseries': {},
            'static': {},
            'predictions': {}
        }
        
        # Validera timeseries features
        timeseries_features = [f for f in self.timeseries_features if f in timeseries_df.columns]
        if timeseries_features:
            validation_results['timeseries'] = self.unified_normalizer.validate_ranges(timeseries_df, timeseries_features)
        
        # Validera static features
        if clinical_df is not None:
            static_features = [f for f in self.static_features if f in clinical_df.columns]
            if static_features:
                # Skapa temporär DataFrame för validering
                temp_df = clinical_df[static_features].copy()
                # Använd unified normalizer för grundläggande range-validering
                for feature in static_features:
                    if feature in self.unified_normalizer.feature_ranges:
                        feature_validation = self.unified_normalizer.validate_ranges(temp_df, [feature])
                        validation_results['static'].update(feature_validation)
        
        # Validera predictions
        if predictions is not None:
            validation_results['predictions'] = self.output_denormalizer.validate_predictions(predictions)
        
        return validation_results
    
    def get_feature_info(self) -> Dict[str, Dict]:
        """
        Hämta information om alla features.
        
        Returns:
            Dict med information om alla features
        """
        info = {
            'timeseries_features': {},
            'static_features': {},
            'output_predictions': {}
        }
        
        # Timeseries features info
        for feature in self.timeseries_features:
            info['timeseries_features'][feature] = self.unified_normalizer.get_feature_info(feature)
        
        # Static features info (basic info)
        for feature in self.static_features:
            info['static_features'][feature] = {
                'name': feature,
                'default': self.static_normalizer.default_values[feature],
                'range': self.static_normalizer.ranges[feature],
                'normalization': 'Master POC specific formula'
            }
        
        # Output predictions info
        for prediction in self.output_predictions:
            info['output_predictions'][prediction] = self.output_denormalizer.get_prediction_info(prediction)
        
        return info
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Hämta sammanfattning av Master Normalizer.
        
        Returns:
            Dict med sammanfattning
        """
        return {
            'target_range': (-1, 1),
            'timeseries_features': {
                'count': len(self.timeseries_features),
                'features': self.timeseries_features,
                'normalization': 'Unified Normalization with clinical ranges'
            },
            'static_features': {
                'count': len(self.static_features),
                'features': self.static_features,
                'normalization': 'Master POC specific formulas'
            },
            'output_predictions': {
                'count': len(self.output_predictions),
                'predictions': self.output_predictions,
                'denormalization': 'Master POC reverse formulas'
            },
            'total_input_features': len(self.timeseries_features) + len(self.static_features),
            'total_output_predictions': len(self.output_predictions)
        }


def create_master_normalizer() -> MasterNormalizer:
    """Factory function för att skapa Master Normalizer."""
    return MasterNormalizer()


# Convenience functions för enkel användning
def normalize_complete_data(timeseries_df: pd.DataFrame, 
                           clinical_df: Optional[pd.DataFrame] = None,
                           targets_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame], MasterNormalizer]:
    """
    Normalisera komplett dataset med Master POC standarder.
    
    Returns:
        Tuple av (normaliserade timeseries, static features, normaliserade targets, normalizer instance)
    """
    normalizer = create_master_normalizer()
    normalized_timeseries, static_features, normalized_targets = normalizer.process_complete_dataset(
        timeseries_df, clinical_df, targets_df
    )
    return normalized_timeseries, static_features, normalized_targets, normalizer
