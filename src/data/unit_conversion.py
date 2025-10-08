"""
Enhetskonvertering för VitalDB data.

Hanterar konvertering från VitalDB-enheter till kliniska standardenheter
baserat på patientvikt och läkemedelskoncentrationer.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

# Import av interfaces och centraliserad konfiguration (eliminerar tight coupling)
from interfaces import IUnitConversionService, ConversionResult
from config import get_drug_concentrations, get_safety_limits_dict

# Konfigurera logging
logger = logging.getLogger(__name__)

# DEPRECERAD: All konfiguration har flyttats till configs/safety_limits.yaml
# Använd get_drug_concentrations() och get_safety_limits_dict() istället

# Bakåtkompatibilitet - ladda från centraliserad konfiguration
DRUG_CONCENTRATIONS = get_drug_concentrations()
DRUG_SAFETY_LIMITS = get_safety_limits_dict('drug')

# Skapa en global instans av service för backward compatibility (dependency injection)
from container import get_container
_container = get_container()
_conversion_service = _container.get(IUnitConversionService)


def convert_vitaldb_units(vitaldb_data: pd.DataFrame, patient_weight: float) -> pd.DataFrame:
    """
    Konvertera VitalDB-enheter till modellens standardenheter.
    Kan hantera både råa kolumnnamn (t.ex. Orchestra/PPF20_RATE) och standardnamn (t.ex. Propofol_INF).
    Om standardnamnet redan finns skrivs det över med konverterat värde.
    
    Args:
        vitaldb_data: DataFrame med rådata från VitalDB (kan vara mappad eller omappad)
        patient_weight: Patientens vikt i kg
    Returns:
        DataFrame med konverterade enheter
    """
    # Använd nya domain service
    result: ConversionResult = _conversion_service.convert_units(vitaldb_data, patient_weight)
    
    # Logga eventuella fel för bakåtkompatibilitet
    if result.failed_conversions:
        logger.warning(f"Failed conversions: {result.failed_conversions}")
    
    if result.safety_violations:
        logger.warning(f"Safety violations detected: {result.safety_violations}")
    
    return result.converted_data


def validate_drug_safety_limits(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validera att läkemedelsdoser ligger inom säkra kliniska gränser.
    Använder centraliserad validator.
    
    Args:
        df: DataFrame med konverterade läkemedelsdata
        
    Returns:
        Dict med valideringsresultat för varje läkemedel
    """
    # Använd centraliserad validator
    from utils.validators import validate_drug_safety_limits as centralized_validator
    return centralized_validator(df)


def get_conversion_statistics() -> Dict[str, Dict]:
    """
    Hämta statistik över enhetskonverteringar.
    
    Returns:
        Dict med konverteringsinformation för varje läkemedel
    """
    return {
        'concentrations': DRUG_CONCENTRATIONS.copy(),
        'safety_limits': DRUG_SAFETY_LIMITS.copy(),
        'conversion_formulas': {
            'Propofol_INF': 'mg/kg/h = (mL/h × 20 mg/mL) ÷ patient_weight_kg',
            'Remifentanil_INF': 'μg/kg/min = (mL/h × concentration_μg/mL) ÷ 60 min ÷ patient_weight_kg',
            'Noradrenalin_INF': 'μg/kg/min = (mL/h × 20 μg/mL) ÷ 60 min ÷ patient_weight_kg',
            'PEEP': 'cmH2O = mbar × 1.02',
            'Sevoflurane': '% = (kPa ÷ 101.325) × 100'
        }
    }


def calculate_drug_dose(ml_per_hour: float, concentration: float, patient_weight: float, 
                       drug_type: str) -> float:
    """
    Beräkna läkemedelsdos i kliniska enheter.
    
    Args:
        ml_per_hour: Infusionshastighet i mL/h
        concentration: Läkemedelskoncentration i mg/mL eller μg/mL
        patient_weight: Patientens vikt i kg
        drug_type: Typ av läkemedel ('propofol_inf', 'remifentanil_inf', 'noradrenalin_inf')
        
    Returns:
        Dos i kliniska enheter
    """
    if patient_weight <= 0:
        raise ValueError(f"Invalid patient weight: {patient_weight} kg")
    
    if drug_type.lower() in ['propofol', 'propofol_inf']:
        # Propofol_INF: mL/h → mg/kg/h
        mg_per_hour = ml_per_hour * concentration
        return mg_per_hour / patient_weight
    
    elif drug_type.lower() in ['remifentanil', 'remifentanil_inf', 'noradrenalin', 'noradrenalin_inf']:
        # Remifentanil_INF/Noradrenalin_INF: mL/h → μg/kg/min
        mcg_per_minute = (ml_per_hour * concentration) / 60
        return mcg_per_minute / patient_weight
    
    else:
        raise ValueError(f"Unknown drug type: {drug_type}")


def verify_conversion_accuracy(test_data: pd.DataFrame, expected_results: Dict[str, float], 
                              tolerance: float = 0.001) -> Dict[str, bool]:
    """
    Verifiera att enhetskonverteringar är korrekta.
    
    Args:
        test_data: DataFrame med konverterade data
        expected_results: Dict med förväntade värden
        tolerance: Tolerans för numerisk precision
        
    Returns:
        Dict med verifieringsresultat
    """
    verification_results = {}
    
    for drug, expected_value in expected_results.items():
        if drug in test_data.columns:
            actual_value = test_data[drug].iloc[0]
            difference = abs(actual_value - expected_value)
            verification_results[drug] = difference <= tolerance
            
            if not verification_results[drug]:
                logger.warning(
                    f"Conversion accuracy check failed for {drug}: "
                    f"expected {expected_value:.6f}, got {actual_value:.6f}, "
                    f"difference {difference:.6f} > tolerance {tolerance}"
                )
        else:
            verification_results[drug] = False
            logger.warning(f"Drug {drug} not found in test data")
    
    return verification_results


class UnitConverter:
    """Enhetskonverterare för VitalDB data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_units(self, vitaldb_data: pd.DataFrame, patient_weight: float) -> pd.DataFrame:
        """Konvertera VitalDB-enheter till standardenheter."""
        return convert_vitaldb_units(vitaldb_data, patient_weight)
    
    def validate_safety_limits(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validera säkerhetsgränser för läkemedel."""
        return validate_drug_safety_limits(df)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Hämta konverteringsstatistik."""
        return get_conversion_statistics()
