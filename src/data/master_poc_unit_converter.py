#!/usr/bin/env python3
"""
Master POC Unit Conversion v5.0
===============================

Implementerar unit conversions enligt Master POC CNN-LSTM-LSTM v5.0 specifikation.
Baserat på Master_POC_CNN-LSTM-LSTM_v5.0.md

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class MasterPOCUnitConverter:
    """Unit converter för Master POC preprocessing"""
    
    def __init__(self):
        # Drug concentrations enligt Master POC spec
        self.drug_concentrations = {
            'Propofol_INF': 20.0,  # mg/ml
            'Remifentanil_INF': 20.0,  # mcg/ml
            'Noradrenalin_INF': 0.1  # mcg/ml (uppskattning)
        }
        
        # Unit conversion factors
        self.conversion_factors = {
            'ml_h_to_mg_kg_h': 1.0,  # Will be calculated per case
            'ml_h_to_mcg_kg_min': 1.0,  # Will be calculated per case
            'ml_to_ml_kg_ibw': 1.0,  # Will be calculated per case
            'kpa_to_cmh2o': 10.197,  # 1 kPa = 10.197 cmH2O
            'cmh2o_to_kpa': 0.0981   # 1 cmH2O = 0.0981 kPa
        }
    
    def calculate_ideal_body_weight(self, height_cm: float, sex: int) -> float:
        """
        Beräkna Ideal Body Weight (IBW) med Devine formula.
        
        Args:
            height_cm: Höjd i centimeter
            sex: -1 (Female) eller 1 (Male)
            
        Returns:
            float: IBW i kg
        """
        if height_cm <= 0:
            logger.warning(f"Invalid height {height_cm} cm for IBW calculation")
            return 0.0
        
        height_inches = height_cm / 2.54
        
        if sex == 1:  # Male
            # Devine formula för män: IBW = 50 + 2.3 * (height_inches - 60)
            if height_inches >= 60:
                ibw = 50 + 2.3 * (height_inches - 60)
            else:
                ibw = 50  # Minimum IBW för män
        else:  # Female (sex == -1)
            # Devine formula för kvinnor: IBW = 45.5 + 2.3 * (height_inches - 60)
            if height_inches >= 60:
                ibw = 45.5 + 2.3 * (height_inches - 60)
            else:
                ibw = 45.5  # Minimum IBW för kvinnor
        
        return round(ibw, 1)
    
    def convert_propofol_ml_h_to_mg_kg_h(self, ml_h: float, weight_kg: float) -> float:
        """
        Konvertera Propofol från mL/h till mg/kg/h.
        
        Args:
            ml_h: Infusion rate i mL/h
            weight_kg: Patient vikt i kg
            
        Returns:
            float: Infusion rate i mg/kg/h
        """
        if weight_kg <= 0:
            logger.warning(f"Invalid weight {weight_kg} kg for Propofol conversion")
            return 0.0
        
        concentration_mg_ml = self.drug_concentrations['Propofol_INF']
        mg_h = ml_h * concentration_mg_ml
        mg_kg_h = mg_h / weight_kg
        
        return round(mg_kg_h, 3)
    
    def convert_remifentanil_ml_h_to_mcg_kg_min(self, ml_h: float, weight_kg: float) -> float:
        """
        Konvertera Remifentanil från mL/h till mcg/kg/min.
        
        Args:
            ml_h: Infusion rate i mL/h
            weight_kg: Patient vikt i kg
            
        Returns:
            float: Infusion rate i mcg/kg/min
        """
        if weight_kg <= 0:
            logger.warning(f"Invalid weight {weight_kg} kg for Remifentanil conversion")
            return 0.0
        
        concentration_mcg_ml = self.drug_concentrations['Remifentanil_INF']
        mcg_h = ml_h * concentration_mcg_ml
        mcg_min = mcg_h / 60  # Konvertera från h till min
        mcg_kg_min = mcg_min / weight_kg
        
        return round(mcg_kg_min, 4)
    
    def convert_noradrenalin_ml_h_to_mcg_kg_min(self, ml_h: float, weight_kg: float) -> float:
        """
        Konvertera Noradrenalin från mL/h till mcg/kg/min.
        
        Args:
            ml_h: Infusion rate i mL/h
            weight_kg: Patient vikt i kg
            
        Returns:
            float: Infusion rate i mcg/kg/min
        """
        if weight_kg <= 0:
            logger.warning(f"Invalid weight {weight_kg} kg for Noradrenalin conversion")
            return 0.0
        
        concentration_mcg_ml = self.drug_concentrations['Noradrenalin_INF']
        mcg_h = ml_h * concentration_mcg_ml
        mcg_min = mcg_h / 60  # Konvertera från h till min
        mcg_kg_min = mcg_min / weight_kg
        
        return round(mcg_kg_min, 4)
    
    def convert_tidal_volume_ml_to_ml_kg_ibw(self, tv_ml: float, height_cm: float, sex: int) -> float:
        """
        Konvertera Tidal Volume från ml till ml/kg IBW.
        
        Args:
            tv_ml: Tidal volume i ml
            height_cm: Patient höjd i cm
            sex: -1 (Female) eller 1 (Male)
            
        Returns:
            float: Tidal volume i ml/kg IBW
        """
        ibw_kg = self.calculate_ideal_body_weight(height_cm, sex)
        
        if ibw_kg <= 0:
            logger.warning(f"Invalid IBW {ibw_kg} kg for TV conversion")
            return 0.0
        
        tv_ml_kg_ibw = tv_ml / ibw_kg
        
        return round(tv_ml_kg_ibw, 2)
    
    def convert_kpa_to_cmh2o(self, kpa: float) -> float:
        """
        Konvertera från kPa till cmH2O.
        
        Args:
            kpa: Tryck i kPa
            
        Returns:
            float: Tryck i cmH2O
        """
        cmh2o = kpa * self.conversion_factors['kpa_to_cmh2o']
        return round(cmh2o, 2)
    
    def convert_cmh2o_to_kpa(self, cmh2o: float) -> float:
        """
        Konvertera från cmH2O till kPa.
        
        Args:
            cmh2o: Tryck i cmH2O
            
        Returns:
            float: Tryck i kPa
        """
        kpa = cmh2o * self.conversion_factors['cmh2o_to_kpa']
        return round(kpa, 3)
    
    def convert_drug_infusion(self, drug_name: str, ml_h: float, weight_kg: float) -> float:
        """
        Konvertera drug infusion baserat på drug name.
        
        Args:
            drug_name: Namn på drug (Propofol_INF, Remifentanil_INF, Noradrenalin_INF)
            ml_h: Infusion rate i mL/h
            weight_kg: Patient vikt i kg
            
        Returns:
            float: Konverterad infusion rate
        """
        if drug_name == 'Propofol_INF':
            return self.convert_propofol_ml_h_to_mg_kg_h(ml_h, weight_kg)
        elif drug_name == 'Remifentanil_INF':
            return self.convert_remifentanil_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        elif drug_name == 'Noradrenalin_INF':
            return self.convert_noradrenalin_ml_h_to_mcg_kg_min(ml_h, weight_kg)
        else:
            logger.error(f"Unknown drug name: {drug_name}")
            return 0.0
    
    def validate_conversion_ranges(self, converted_value: float, drug_name: str) -> bool:
        """
        Validera att konverterade värden ligger inom kliniska ranges.
        
        Args:
            converted_value: Konverterat värde
            drug_name: Namn på drug
            
        Returns:
            bool: True om inom range, False annars
        """
        clinical_ranges = {
            'Propofol_INF': (0, 12),  # mg/kg/h
            'Remifentanil_INF': (0, 0.8),  # mcg/kg/min
            'Noradrenalin_INF': (0, 0.5),  # mcg/kg/min
            'TV': (0, 12),  # ml/kg IBW
            'PEEP': (0, 30),  # cmH2O
            'etSEV': (0, 6),  # kPa
            'inSev': (0, 8)  # kPa
        }
        
        if drug_name in clinical_ranges:
            min_val, max_val = clinical_ranges[drug_name]
            return min_val <= converted_value <= max_val
        
        return True  # Okänd drug, antar att det är OK

def create_unit_converter() -> MasterPOCUnitConverter:
    """Skapa en ny MasterPOCUnitConverter instans."""
    return MasterPOCUnitConverter()
