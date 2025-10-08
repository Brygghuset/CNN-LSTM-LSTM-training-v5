"""
Data preprocessing-modul för CNN-LSTM-LSTM projektet.

Denna modul separerar olika ansvarsområden:
- ImputationMethod: Enum för olika imputeringsmetoder
- DataPreprocessor: Huvudklass med förenklad funktionalitet
- ImputationStrategy: Specifika imputeringsstrategier  
- ValidationProcessor: Fysiologisk validering
"""

from data.preprocessing.imputation_methods import ImputationMethod
from data.preprocessing.data_preprocessor import DataPreprocessor
from data.preprocessing.imputation_strategies import (
    ForwardFillStrategy, BackwardFillStrategy, LinearInterpolationStrategy,
    MeanImputationStrategy, MedianImputationStrategy, ZeroImputationStrategy
)

# Bakåtkompatibla funktioner
from data.preprocessing.data_preprocessor import impute_missing_values, get_missing_values_stats

__all__ = [
    # Enums
    'ImputationMethod',
    
    # Huvudklasser
    'DataPreprocessor',
    
    # Strategier
    'ForwardFillStrategy',
    'BackwardFillStrategy', 
    'LinearInterpolationStrategy',
    'MeanImputationStrategy',
    'MedianImputationStrategy',
    'ZeroImputationStrategy',
    
    # Bakåtkompatibla funktioner
    'impute_missing_values',
    'get_missing_values_stats'
] 