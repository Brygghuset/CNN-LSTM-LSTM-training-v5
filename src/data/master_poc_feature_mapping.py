#!/usr/bin/env python3
"""
Master POC Feature Mapping v5.0
===============================

Implementerar Master POC CNN-LSTM-LSTM v5.0 specifikation för feature mapping.
Baserat på Master_POC_CNN-LSTM-LSTM_v5.0.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Master POC Timeseries Features (16 features)
MASTER_POC_TIMESERIES_FEATURES = {
    # Vital Signs (7 features)
    'HR': {
        'order': 1,
        'unit': 'BPM',
        'vitaldb_source': ['Solar8000/HR'],
        'clinical_range': (20, 200),
        'default_value': 70,
        'imputation_strategy': 'default_forward_fill'
    },
    'BP_SYS': {
        'order': 2,
        'unit': 'mmHg',
        'vitaldb_source': ['Solar8000/ART_SBP', 'Solar8000/NIBP_SBP'],
        'clinical_range': (60, 250),
        'default_value': 140,
        'imputation_strategy': 'default_forward_fill'
    },
    'BP_DIA': {
        'order': 3,
        'unit': 'mmHg',
        'vitaldb_source': ['Solar8000/ART_DBP', 'Solar8000/NIBP_DBP'],
        'clinical_range': (30, 150),
        'default_value': 80,
        'imputation_strategy': 'default_forward_fill'
    },
    'BP_MAP': {
        'order': 4,
        'unit': 'mmHg',
        'vitaldb_source': ['Solar8000/ART_MBP', 'Solar8000/NIBP_MBP', 'EV1000/ART_MBP'],
        'clinical_range': (40, 180),
        'default_value': None,  # Beräknas från default BP_SYS och BP_DIA
        'imputation_strategy': 'default_forward_fill'
    },
    'SPO2': {
        'order': 5,
        'unit': '%',
        'vitaldb_source': ['Solar8000/PLETH_SPO2'],
        'clinical_range': (70, 100),
        'default_value': 96,
        'imputation_strategy': 'default_forward_fill'
    },
    'ETCO2': {
        'order': 6,
        'unit': 'kPa',
        'vitaldb_source': ['Solar8000/ETCO2'],
        'clinical_range': (2.0, 8.0),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    'BIS': {
        'order': 7,
        'unit': 'Numerisk',
        'vitaldb_source': ['BIS/BIS'],
        'clinical_range': (0, 100),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    
    # Drug Infusions (3 features)
    'Propofol_INF': {
        'order': 8,
        'unit': 'mg/kg/h',
        'vitaldb_source': ['Orchestra/PPF20_RATE'],
        'clinical_range': (0, 12),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero',
        'concentration': 20.0,  # mg/ml
        'conversion_factor': 'ml_h_to_mg_kg_h'
    },
    'Remifentanil_INF': {
        'order': 9,
        'unit': 'mcg/kg/min',
        'vitaldb_source': ['Orchestra/RFTN20_RATE'],
        'clinical_range': (0, 0.8),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero',
        'concentration': 20.0,  # mcg/ml
        'conversion_factor': 'ml_h_to_mcg_kg_min'
    },
    'Noradrenalin_INF': {
        'order': 10,
        'unit': 'mcg/kg/min',
        'vitaldb_source': ['Orchestra/NEPI_RATE'],
        'clinical_range': (0, 0.5),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero',
        'concentration': 0.1,  # mcg/ml (uppskattning för Noradrenalin)
        'conversion_factor': 'ml_h_to_mcg_kg_min'
    },
    
    # Ventilator Settings (6 features)
    'TV': {
        'order': 11,
        'unit': 'ml/kg IBW',
        'vitaldb_source': ['Solar8000/VENT_TV'],
        'clinical_range': (0, 12),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero',
        'conversion_factor': 'ml_to_ml_kg_ibw'
    },
    'PEEP': {
        'order': 12,
        'unit': 'cmH2O',
        'vitaldb_source': ['Solar8000/VENT_MEAS_PEEP'],
        'clinical_range': (0, 30),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    'FIO2': {
        'order': 13,
        'unit': '%',
        'vitaldb_source': ['Solar8000/FIO2'],
        'clinical_range': (21, 100),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    'RR': {
        'order': 14,
        'unit': 'breaths/min',
        'vitaldb_source': ['Solar8000/RR', 'Solar8000/RR_CO2', 'Primus/RR_CO2'],
        'clinical_range': (6, 30),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    'etSEV': {
        'order': 15,
        'unit': 'kPa',
        'vitaldb_source': ['Primus/EXP_SEVO'],
        'clinical_range': (0, 6),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    },
    'inSev': {
        'order': 16,
        'unit': 'kPa',
        'vitaldb_source': ['Primus/INSP_SEVO'],
        'clinical_range': (0, 8),
        'default_value': 0.0,  # Klinisk nolla
        'imputation_strategy': 'clinical_zero'
    }
}

# Master POC Static Features (6 features)
MASTER_POC_STATIC_FEATURES = {
    'age': {
        'order': 17,
        'unit': 'years',
        'default_value': 50,
        'range': (0, 120),
        'normalization_formula': 'age/120 × 2 - 1'
    },
    'sex': {
        'order': 18,
        'unit': 'F/M',
        'default_value': -1,  # Female
        'range': (-1, 1),  # -1 = Female, 1 = Male
        'normalization_formula': '1.0 eller -1.0'
    },
    'height': {
        'order': 19,
        'unit': 'cm',
        'default_value': 170,
        'range': (100, 230),
        'normalization_formula': '(height-100)/130 × 2 - 1'
    },
    'weight': {
        'order': 20,
        'unit': 'kg',
        'default_value': 70,
        'range': (20, 200),
        'normalization_formula': '(weight-20)/180 × 2 - 1'
    },
    'bmi': {
        'order': 21,
        'unit': 'kg/m²',
        'default_value': 24.2,
        'range': (10, 50),
        'normalization_formula': '(bmi-10)/40 × 2 - 1'
    },
    'asa': {
        'order': 22,
        'unit': '1-6',
        'default_value': 2,
        'range': (1, 6),
        'normalization_formula': '(asa-1)/5 × 2 - 1'
    }
}

# Master POC Output Features (8 features)
MASTER_POC_OUTPUT_FEATURES = {
    # Drug Output (3 predictions)
    'Propofol_Predict': {
        'order': 1,
        'unit': 'mg/kg/h',
        'range': (0, 12),
        'inverse_normalization_formula': '(norm + 1) × 6'
    },
    'Remifentanil_Predict': {
        'order': 2,
        'unit': 'mcg/kg/min',
        'range': (0, 0.8),
        'inverse_normalization_formula': '(norm + 1) × 0.4'
    },
    'Noradrenalin_Predict': {
        'order': 3,
        'unit': 'mcg/kg/min',
        'range': (0, 0.5),
        'inverse_normalization_formula': '(norm + 1) × 0.25'
    },
    
    # Ventilator Output (5 predictions)
    'TV_Predict': {
        'order': 4,
        'unit': 'ml/kg IBW',
        'range': (0, 12),
        'inverse_normalization_formula': '(norm + 1) × 6'
    },
    'PEEP_Predict': {
        'order': 5,
        'unit': 'cmH2O',
        'range': (0, 30),
        'inverse_normalization_formula': '(norm + 1) × 15'
    },
    'FIO2_Predict': {
        'order': 6,
        'unit': '%',
        'range': (21, 100),
        'inverse_normalization_formula': '21 + (norm + 1) × 39.5'
    },
    'RR_Predict': {
        'order': 7,
        'unit': 'breaths/min',
        'range': (6, 30),
        'inverse_normalization_formula': '6 + (norm + 1) × 12'
    },
    'etSEV_Predict': {
        'order': 8,
        'unit': 'kPa',
        'range': (0, 6),
        'inverse_normalization_formula': '(norm + 1) × 3'
    }
}

def get_master_poc_feature_count() -> Dict[str, int]:
    """Returnera antal features enligt Master POC specifikation."""
    return {
        'timeseries_features': len(MASTER_POC_TIMESERIES_FEATURES),
        'static_features': len(MASTER_POC_STATIC_FEATURES),
        'output_features': len(MASTER_POC_OUTPUT_FEATURES),
        'total_input_features': len(MASTER_POC_TIMESERIES_FEATURES) + len(MASTER_POC_STATIC_FEATURES)
    }

def get_timeseries_feature_order() -> List[str]:
    """Returnera timeseries features i korrekt ordning."""
    return sorted(MASTER_POC_TIMESERIES_FEATURES.keys(), 
                  key=lambda x: MASTER_POC_TIMESERIES_FEATURES[x]['order'])

def get_static_feature_order() -> List[str]:
    """Returnera static features i korrekt ordning."""
    return sorted(MASTER_POC_STATIC_FEATURES.keys(), 
                  key=lambda x: MASTER_POC_STATIC_FEATURES[x]['order'])

def get_output_feature_order() -> List[str]:
    """Returnera output features i korrekt ordning."""
    return sorted(MASTER_POC_OUTPUT_FEATURES.keys(), 
                  key=lambda x: MASTER_POC_OUTPUT_FEATURES[x]['order'])

def validate_master_poc_feature_mapping() -> bool:
    """Validera att feature mapping följer Master POC specifikation."""
    counts = get_master_poc_feature_count()
    
    # Validera antal features
    assert counts['timeseries_features'] == 16, f"Förväntade 16 timeseries features, fick {counts['timeseries_features']}"
    assert counts['static_features'] == 6, f"Förväntade 6 static features, fick {counts['static_features']}"
    assert counts['output_features'] == 8, f"Förväntade 8 output features, fick {counts['output_features']}"
    assert counts['total_input_features'] == 22, f"Förväntade 22 total input features, fick {counts['total_input_features']}"
    
    # Validera att alla features har korrekt ordning
    timeseries_order = get_timeseries_feature_order()
    for i, feature in enumerate(timeseries_order):
        expected_order = i + 1
        actual_order = MASTER_POC_TIMESERIES_FEATURES[feature]['order']
        assert actual_order == expected_order, f"Feature {feature} har fel ordning: förväntade {expected_order}, fick {actual_order}"
    
    logger.info("✅ Master POC feature mapping validerad")
    return True
