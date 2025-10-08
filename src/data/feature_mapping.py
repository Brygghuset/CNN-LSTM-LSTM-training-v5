"""
Feature mapping för VitalDB data.

Hanterar mappning från VitalDB-kolumnnamn till standardiserade namn
enligt klinisk standard.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from data.mappers.feature_mapper import FeatureMapper

# Import av interfaces och centraliserad konfiguration (eliminerar tight coupling)
from interfaces import IFeatureMappingService, MappingResult
from config import get_feature_mapping, get_safety_limits_dict

# Konfigurera logging
logger = logging.getLogger(__name__)

# DEPRECERAD: All konfiguration har flyttats till configs/safety_limits.yaml
# Använd get_feature_mapping() och get_safety_limits_dict() istället

# Bakåtkompatibilitet - ladda från centraliserad konfiguration
FEATURE_MAPPING = get_feature_mapping()
SAFETY_LIMITS = get_safety_limits_dict('vital')

# Skapa en global instans av service för backward compatibility (dependency injection)
from container import get_container
_container = get_container()
_mapping_service = _container.get(IFeatureMappingService)


def find_case_insensitive_match(target: str, candidates: List[str]) -> Optional[Tuple[str, str]]:
    """
    Hitta case-insensitive match mellan target och kandidatlista.
    
    Args:
        target: Kolumnnamn att matcha
        candidates: Lista med kandidatnamn att matcha mot
        
    Returns:
        Tuple med (matchande_kandidat, original_kolumn) om match hittas, None annars
    """
    # Delegera till service
    return _mapping_service.find_case_insensitive_match(target, candidates)


def map_incoming_data(raw_data: pd.DataFrame, include_source_columns: bool = False, enforce_numeric_types: bool = False) -> pd.DataFrame:
    """
    Mappa inkommande VitalDB data till standardiserade namn för vitalparametrar och ventilatorparametrar.
    Endast standardnamn inkluderas i resultatet om include_source_columns=False. Om True inkluderas även originalkolumnen (om den inte krockar).
    Loggning sker av vilken faktisk kolumn (och case-variant) som användes för varje parameter.
    
    Args:
        raw_data: DataFrame med rådata från VitalDB
        include_source_columns: Om True inkluderas även originalkolumnen i resultatet
        enforce_numeric_types: Om True konverteras alla medicinska parametrar till float64 för konsistent numerisk processing
        
    Returns:
        DataFrame med mappade data
    """
    # Använd nya domain service
    result: MappingResult = _mapping_service.map_data(raw_data, include_source_columns, enforce_numeric_types)
    
    # Logga traditionella meddelanden för bakåtkompatibilitet
    vital_params_mapped = sum(1 for param in ["HR", "NBP_SYS", "NBP_DIA", "MBP", "SPO2", "ETCO2", "BIS"] 
                             if param in result.mapping_details)
    vent_params_mapped = sum(1 for param in ["TV", "PEEP", "FiO2", "Sevoflurane"] 
                            if param in result.mapping_details)
    drug_params_mapped = sum(1 for param in ["Propofol_INF", "Remifentanil_INF", "Noradrenalin_INF"] 
                            if param in result.mapping_details)
    
    # Logga mappningsstatistik
    logger.info(f"Feature mapping slutförd: {result.mapped_columns}/{result.total_columns} kolumner mappade")
    logger.info(f"Vitalparametrar: {vital_params_mapped}/7, Ventilatorparametrar: {vent_params_mapped}/4, Läkemedelsparametrar: {drug_params_mapped}/3")
    
    if vital_params_mapped == 7:
        logger.info(f"Successfully mapped 7 vital parameters")
    
    # Logga detaljerad mappningsinformation
    for param, details in result.mapping_details.items():
        logger.info(f"{param}: {details['used_column']} ({details['match_type']}) - {details.get('data_type', 'N/A')}")
        logger.info(f"Mapped {param} from column '{details['used_column']}' (match type: {details['match_type']})")
    
    # Logga case conflicts om de finns
    if result.case_conflicts:
        logger.warning(f"Case variant conflicts detected: {result.case_conflicts}")
    
    return result.mapped_data


def validate_vital_parameters(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validera att vitalparametrar ligger inom kliniskt acceptabla gränser.
    Använder centraliserad validator.
    
    Args:
        df: DataFrame med vitalparametrar
        
    Returns:
        Dict med valideringsresultat för varje parameter
    """
    # Använd centraliserad validator
    from utils.validators import validate_vital_parameters as centralized_validator
    return centralized_validator(df)


def get_mapping_statistics() -> Dict[str, int]:
    """
    Hämta statistik över feature mapping.
    
    Returns:
        Dict med mappningsstatistik
    """
    return {
        'total_parameters': len(FEATURE_MAPPING),
        'vital_parameters': 7,
        'drug_parameters': 3,
        'ventilator_parameters': 4,
        'total_variants': sum(len(variants) for variants in FEATURE_MAPPING.values())
    }
