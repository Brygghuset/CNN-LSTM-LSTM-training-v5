"""
Output Denormalization enligt Master POC specifikation.

Implementerar denormalisering av modell predictions tillbaka till kliniska enheter
enligt Master POC CNN-LSTM-LSTM dokumentet.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class OutputDenormalizer:
    """
    Denormalisering av modell outputs enligt Master POC specifikationer.
    
    Implementerar specifika reverse normalization formulas:
    - För ranges som börjar på 0: (norm + 1) × (max/2)
    - För ranges som inte börjar på 0: min + (norm + 1) × ((max-min)/2)
    """
    
    def __init__(self):
        """Initialisera Output Denormalizer."""
        # Master POC output ranges
        self.output_ranges = {
            # Drug Predictions (3 features)
            'Propofol_Predict': {'min': 0, 'max': 12, 'unit': 'mg/kg/h'},
            'Remifentanil_Predict': {'min': 0, 'max': 0.8, 'unit': 'mcg/kg/min'},
            'Noradrenalin_Predict': {'min': 0, 'max': 0.5, 'unit': 'mcg/kg/min'},
            
            # Ventilator Predictions (5 features)
            'TV_Predict': {'min': 0, 'max': 12, 'unit': 'ml/kg IBW'},
            'PEEP_Predict': {'min': 0, 'max': 30, 'unit': 'cmH2O'},
            'FIO2_Predict': {'min': 21, 'max': 100, 'unit': '%'},
            'RR_Predict': {'min': 6, 'max': 30, 'unit': 'breaths/min'},
            'etSEV_Predict': {'min': 0, 'max': 6, 'unit': 'kPa'}
        }
        
        logger.info("Output Denormalizer initialiserad med Master POC ranges")
    
    def denormalize_prediction(self, normalized_value: Union[float, np.ndarray], prediction_name: str) -> Union[float, np.ndarray]:
        """
        Denormalisera en enskild prediction enligt Master POC formler.
        
        Args:
            normalized_value: Normaliserat värde i [-1, 1]
            prediction_name: Namn på prediction
            
        Returns:
            Denormaliserat värde i kliniska enheter
        """
        if prediction_name not in self.output_ranges:
            raise ValueError(f"Okänd prediction: {prediction_name}. Tillgängliga: {list(self.output_ranges.keys())}")
        
        # Konvertera till numpy array
        if isinstance(normalized_value, (float, int)):
            normalized_value = np.array([normalized_value])
        elif not isinstance(normalized_value, np.ndarray):
            normalized_value = np.array(normalized_value)
        
        range_info = self.output_ranges[prediction_name]
        min_val = range_info['min']
        max_val = range_info['max']
        
        # Master POC reverse normalization formulas
        if min_val == 0:
            # För ranges som börjar på 0: (norm + 1) × (max/2)
            denormalized = (normalized_value + 1) * (max_val / 2)
        else:
            # För ranges som inte börjar på 0: min + (norm + 1) × ((max-min)/2)
            denormalized = min_val + (normalized_value + 1) * ((max_val - min_val) / 2)
        
        # Hantera NaN värden
        denormalized = np.where(np.isnan(normalized_value), np.nan, denormalized)
        
        # Clamp till giltiga ranges
        denormalized = np.clip(denormalized, min_val, max_val)
        
        return denormalized if len(denormalized) > 1 else denormalized[0]
    
    def normalize_prediction(self, value: Union[float, np.ndarray], prediction_name: str) -> Union[float, np.ndarray]:
        """
        Normalisera en prediction till [-1, 1] (för testing/validering).
        
        Args:
            value: Värde i kliniska enheter
            prediction_name: Namn på prediction
            
        Returns:
            Normaliserat värde i [-1, 1]
        """
        if prediction_name not in self.output_ranges:
            raise ValueError(f"Okänd prediction: {prediction_name}")
        
        # Konvertera till numpy array
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        
        range_info = self.output_ranges[prediction_name]
        min_val = range_info['min']
        max_val = range_info['max']
        
        # Forward normalization formulas (reverse av denormalization)
        if min_val == 0:
            # För ranges som börjar på 0: (value / (max/2)) - 1
            normalized = (value / (max_val / 2)) - 1
        else:
            # För ranges som inte börjar på 0: ((value - min) / ((max-min)/2)) - 1
            normalized = ((value - min_val) / ((max_val - min_val) / 2)) - 1
        
        # Hantera NaN värden
        normalized = np.where(np.isnan(value), np.nan, normalized)
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized if len(normalized) > 1 else normalized[0]
    
    def denormalize_predictions_array(self, predictions: np.ndarray) -> np.ndarray:
        """
        Denormalisera en array med alla 8 predictions.
        
        Args:
            predictions: Array med shape (..., 8) med normaliserade predictions
            
        Returns:
            Array med denormaliserade predictions i kliniska enheter
        """
        if predictions.shape[-1] != 8:
            raise ValueError(f"Förväntar 8 predictions, fick {predictions.shape[-1]}")
        
        # Lista med prediction-namn i ordning
        prediction_names = [
            'Propofol_Predict',      # 0
            'Remifentanil_Predict',  # 1
            'Noradrenalin_Predict',  # 2
            'TV_Predict',            # 3
            'PEEP_Predict',          # 4
            'FIO2_Predict',          # 5
            'RR_Predict',            # 6
            'etSEV_Predict'          # 7
        ]
        
        # Behåll original shape
        original_shape = predictions.shape
        predictions_flat = predictions.reshape(-1, 8)
        
        denormalized_flat = np.zeros_like(predictions_flat)
        
        for i, pred_name in enumerate(prediction_names):
            denormalized_flat[:, i] = self.denormalize_prediction(predictions_flat[:, i], pred_name)
        
        # Återställ original shape
        denormalized = denormalized_flat.reshape(original_shape)
        
        return denormalized
    
    def denormalize_predictions_dict(self, predictions_dict: Dict[str, Union[float, np.ndarray]]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Denormalisera predictions från en dictionary.
        
        Args:
            predictions_dict: Dict med prediction-namn som nycklar och normaliserade värden
            
        Returns:
            Dict med denormaliserade predictions
        """
        denormalized = {}
        
        for pred_name, values in predictions_dict.items():
            if pred_name in self.output_ranges:
                denormalized[pred_name] = self.denormalize_prediction(values, pred_name)
            else:
                logger.warning(f"Okänd prediction {pred_name}, hoppar över denormalisering")
                denormalized[pred_name] = values
        
        return denormalized
    
    def denormalize_predictions_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalisera predictions i en DataFrame.
        
        Args:
            df: DataFrame med normaliserade predictions
            
        Returns:
            DataFrame med denormaliserade predictions
        """
        result = df.copy()
        
        for col in df.columns:
            if col in self.output_ranges:
                result[col] = self.denormalize_prediction(df[col].values, col)
                logger.debug(f"Denormaliserade {col} till {self.output_ranges[col]['unit']}")
        
        return result
    
    def get_prediction_info(self, prediction_name: str) -> Dict[str, Any]:
        """
        Hämta information om en prediction.
        
        Args:
            prediction_name: Namn på prediction
            
        Returns:
            Dict med prediction-information
        """
        if prediction_name not in self.output_ranges:
            raise ValueError(f"Okänd prediction: {prediction_name}")
        
        range_info = self.output_ranges[prediction_name]
        return {
            'name': prediction_name,
            'min': range_info['min'],
            'max': range_info['max'],
            'unit': range_info['unit'],
            'range_type': 'starts_at_zero' if range_info['min'] == 0 else 'offset_range'
        }
    
    def get_all_predictions(self) -> list:
        """Hämta lista med alla tillgängliga predictions."""
        return list(self.output_ranges.keys())
    
    def validate_predictions(self, predictions: Union[np.ndarray, Dict, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validera att predictions ligger inom rimliga gränser.
        
        Args:
            predictions: Predictions att validera
            
        Returns:
            Dict med valideringsresultat per prediction
        """
        validation_results = {}
        
        if isinstance(predictions, np.ndarray):
            # Konvertera array till dict
            if predictions.shape[-1] != 8:
                logger.error(f"Förväntar 8 predictions, fick {predictions.shape[-1]}")
                return validation_results
            
            prediction_names = [
                'Propofol_Predict', 'Remifentanil_Predict', 'Noradrenalin_Predict',
                'TV_Predict', 'PEEP_Predict', 'FIO2_Predict', 'RR_Predict', 'etSEV_Predict'
            ]
            
            predictions_dict = {}
            for i, name in enumerate(prediction_names):
                predictions_dict[name] = predictions[..., i]
            
            predictions = predictions_dict
        
        elif isinstance(predictions, pd.DataFrame):
            predictions = predictions.to_dict('series')
        
        # Validera varje prediction
        for pred_name, values in predictions.items():
            if pred_name not in self.output_ranges:
                continue
            
            if isinstance(values, pd.Series):
                values = values.dropna().values
            elif isinstance(values, np.ndarray):
                values = values[~np.isnan(values)]
            
            if len(values) == 0:
                continue
            
            range_info = self.output_ranges[pred_name]
            min_val, max_val = values.min(), values.max()
            
            # Kontrollera om värden ligger utanför giltiga ranges
            below_min = (values < range_info['min']).sum()
            above_max = (values > range_info['max']).sum()
            
            validation_results[pred_name] = {
                'data_range': (min_val, max_val),
                'clinical_range': (range_info['min'], range_info['max']),
                'values_below_min': below_min,
                'values_above_max': above_max,
                'total_values': len(values),
                'unit': range_info['unit'],
                'within_range': below_min == 0 and above_max == 0
            }
        
        return validation_results


def create_output_denormalizer() -> OutputDenormalizer:
    """Factory function för att skapa Output Denormalizer."""
    return OutputDenormalizer()


# Convenience functions
def denormalize_model_outputs(predictions: np.ndarray) -> np.ndarray:
    """
    Denormalisera modell outputs med Master POC formler.
    
    Args:
        predictions: Array med normaliserade predictions (..., 8)
        
    Returns:
        Array med denormaliserade predictions i kliniska enheter
    """
    denormalizer = create_output_denormalizer()
    return denormalizer.denormalize_predictions_array(predictions)


def normalize_target_outputs(targets: np.ndarray) -> np.ndarray:
    """
    Normalisera target outputs för training (reverse av denormalization).
    
    Args:
        targets: Array med target values i kliniska enheter (..., 8)
        
    Returns:
        Array med normaliserade targets i [-1, 1]
    """
    denormalizer = create_output_denormalizer()
    
    if targets.shape[-1] != 8:
        raise ValueError(f"Förväntar 8 targets, fick {targets.shape[-1]}")
    
    prediction_names = [
        'Propofol_Predict', 'Remifentanil_Predict', 'Noradrenalin_Predict',
        'TV_Predict', 'PEEP_Predict', 'FIO2_Predict', 'RR_Predict', 'etSEV_Predict'
    ]
    
    # Behåll original shape
    original_shape = targets.shape
    targets_flat = targets.reshape(-1, 8)
    
    normalized_flat = np.zeros_like(targets_flat)
    
    for i, pred_name in enumerate(prediction_names):
        normalized_flat[:, i] = denormalizer.normalize_prediction(targets_flat[:, i], pred_name)
    
    # Återställ original shape
    normalized = normalized_flat.reshape(original_shape)
    
    return normalized
