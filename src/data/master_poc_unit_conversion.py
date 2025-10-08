"""
Master POC Unit Conversion Service för CNN-LSTM-LSTM modell.

Implementerar enhetskonvertering enligt Master POC specifikationer:
- Propofol: 20 mg/mL
- Remifentanil: 20 eller 50 μg/mL (beroende på input)
- Noradrenalin: 20 μg/mL
- Sevoflurane: Behåller kPa (1:1 till % om konvertering behövs)
- PEEP: mbar → cmH2O
- Fallback weight: 70 kg (Master POC default)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MasterPOCConversionResult:
    """Resultat från Master POC unit conversion."""
    converted_data: pd.DataFrame
    conversion_details: Dict[str, Dict]
    failed_conversions: List[str]
    patient_weight_used: float
    master_poc_compliant: bool


class MasterPOCUnitConverter:
    """
    Master POC Unit Converter för CNN-LSTM-LSTM modell.
    
    Implementerar enhetskonvertering enligt Master POC specifikationer.
    """
    
    def __init__(self):
        """Initialisera Master POC Unit Converter."""
        self.logger = logging.getLogger(__name__)
        
        # Master POC drug concentrations
        self.master_poc_concentrations = {
            'Propofol_INF': 20.0,      # mg/mL
            'Remifentanil_INF': 20.0,  # μg/mL (default)
            'Noradrenalin_INF': 20.0   # μg/mL
        }
        
        # Master POC fallback weight
        self.master_poc_default_weight = 70.0  # kg enligt Master POC
        
        # Master POC source column mappings
        self.master_poc_source_mappings = {
            'Propofol_INF': ['Orchestra/PPF20_RATE', 'Orchestra/PPF10_RATE', 'Orchestra/PPF_RATE'],
            'Remifentanil_INF': ['Orchestra/RFTN20_RATE', 'Orchestra/RFTN50_RATE', 'Orchestra/RFTN_RATE'],
            'Noradrenalin_INF': ['Orchestra/NEPI_RATE', 'Orchestra/NA_RATE'],
            'etSEV': ['Primus/EXP_SEVO'],
            'inSev': ['Primus/INSP_SEVO'],
            'PEEP': ['Solar8000/VENT_MEAS_PEEP', 'Primus/PEEP_MBAR'],
            'TV': ['Solar8000/VENT_TV', 'Primus/TV', 'TidalVolume']
        }
        
        self.logger.info("Master POC Unit Converter initialiserad")
        self.logger.info(f"Drug concentrations: {self.master_poc_concentrations}")
        self.logger.info(f"Default weight: {self.master_poc_default_weight} kg")
    
    def convert_units(self, vitaldb_data: pd.DataFrame, patient_weight: float) -> MasterPOCConversionResult:
        """
        Konvertera enheter enligt Master POC specifikationer.
        
        Args:
            vitaldb_data: DataFrame med rådata från VitalDB
            patient_weight: Patientens vikt i kg
            
        Returns:
            MasterPOCConversionResult med konverterad data
        """
        self.logger.info(f"Börjar Master POC unit conversion för patient_weight: {patient_weight} kg")
        
        # Skapa kopia för konverterad data
        converted_data = vitaldb_data.copy()
        
        # Kör alla konverteringar
        all_conversion_details = {}
        all_failed_conversions = []
        
        # Drug conversions
        drug_conversions = ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF']
        for drug in drug_conversions:
            conversion_details, failed = self._convert_drug(drug, vitaldb_data, converted_data, patient_weight)
            all_conversion_details.update(conversion_details)
            all_failed_conversions.extend(failed)
        
        # Sevoflurane conversions (behåller kPa)
        sevo_conversions = ['etSEV', 'inSev']
        for sevo in sevo_conversions:
            conversion_details, failed = self._convert_sevoflurane(sevo, vitaldb_data, converted_data)
            all_conversion_details.update(conversion_details)
            all_failed_conversions.extend(failed)
        
        # PEEP conversion
        conversion_details, failed = self._convert_peep(vitaldb_data, converted_data)
        all_conversion_details.update(conversion_details)
        all_failed_conversions.extend(failed)
        
        # TV conversion (ml → ml/kg IBW)
        conversion_details, failed = self._convert_tv(vitaldb_data, converted_data, patient_weight)
        all_conversion_details.update(conversion_details)
        all_failed_conversions.extend(failed)
        
        # Bestäm patient weight som användes
        actual_weight_used = patient_weight if patient_weight > 0 else self.master_poc_default_weight
        
        # Validera Master POC compliance
        master_poc_compliant = self._validate_master_poc_compliance(converted_data, all_conversion_details)
        
        self.logger.info(f"Master POC conversion slutförd: {len(all_conversion_details)} framgångsrika, {len(all_failed_conversions)} misslyckade")
        self.logger.info(f"Master POC compliant: {master_poc_compliant}")
        
        return MasterPOCConversionResult(
            converted_data=converted_data,
            conversion_details=all_conversion_details,
            failed_conversions=all_failed_conversions,
            patient_weight_used=actual_weight_used,
            master_poc_compliant=master_poc_compliant
        )
    
    def _convert_drug(self, drug_name: str, vitaldb_data: pd.DataFrame, 
                     converted_data: pd.DataFrame, patient_weight: float) -> Tuple[Dict, List[str]]:
        """Konvertera drug infusion enligt Master POC specifikation."""
        source_columns = self.master_poc_source_mappings.get(drug_name, [])
        conversion_details = {}
        failed_conversions = []
        
        # Hitta första tillgängliga kolumn
        source_col = None
        for col in source_columns:
            if col in vitaldb_data.columns:
                source_col = col
                break
        
        if source_col is None:
            self.logger.warning(f"Ingen källkolumn hittades för {drug_name}")
            failed_conversions.append(drug_name)
            return conversion_details, failed_conversions
        
        try:
            source_data = vitaldb_data[source_col]
            concentration = self.master_poc_concentrations[drug_name]
            
            # Special handling för Remifentanil (20 eller 50 μg/mL)
            if drug_name == 'Remifentanil_INF' and 'RFTN50' in source_col:
                concentration = 50.0  # μg/mL
            
            # Konvertera baserat på drug type
            if drug_name == 'Propofol_INF':
                # Propofol: mL/hr → mg/kg/h
                ml_per_hour = source_data
                mg_per_hour = ml_per_hour * concentration  # mg/h
                
                # Master POC fallback weight
                if patient_weight <= 0:
                    actual_weight = self.master_poc_default_weight
                else:
                    actual_weight = patient_weight
                
                mg_kg_h = mg_per_hour / actual_weight
                converted_data[drug_name] = mg_kg_h
                
                conversion_details[drug_name] = {
                    'from_value': ml_per_hour.iloc[0] if len(ml_per_hour) > 0 else 0.0,
                    'to_value': mg_kg_h.iloc[0] if len(mg_kg_h) > 0 else 0.0,
                    'from_unit': 'mL/h',
                    'to_unit': 'mg/kg/h',
                    'concentration': concentration,
                    'patient_weight': actual_weight,
                    'source_column': source_col
                }
                
            else:
                # Remifentanil och Noradrenalin: mL/hr → μg/kg/min
                ml_per_hour = source_data
                mcg_per_minute = (ml_per_hour * concentration) / 60  # μg/min
                
                # Master POC fallback weight
                if patient_weight <= 0:
                    actual_weight = self.master_poc_default_weight
                else:
                    actual_weight = patient_weight
                
                mcg_kg_min = mcg_per_minute / actual_weight
                converted_data[drug_name] = mcg_kg_min
                
                conversion_details[drug_name] = {
                    'from_value': ml_per_hour.iloc[0] if len(ml_per_hour) > 0 else 0.0,
                    'to_value': mcg_kg_min.iloc[0] if len(mcg_kg_min) > 0 else 0.0,
                    'from_unit': 'mL/h',
                    'to_unit': 'μg/kg/min',
                    'concentration': concentration,
                    'patient_weight': actual_weight,
                    'source_column': source_col
                }
            
            self.logger.info(f"Konverterade {drug_name}: {conversion_details[drug_name]['from_value']:.1f} {conversion_details[drug_name]['from_unit']} → {conversion_details[drug_name]['to_value']:.3f} {conversion_details[drug_name]['to_unit']}")
            
        except Exception as e:
            self.logger.error(f"Fel vid konvertering av {drug_name}: {e}")
            failed_conversions.append(drug_name)
        
        return conversion_details, failed_conversions
    
    def _convert_sevoflurane(self, sevo_name: str, vitaldb_data: pd.DataFrame, 
                           converted_data: pd.DataFrame) -> Tuple[Dict, List[str]]:
        """Konvertera Sevoflurane enligt Master POC specifikation (behåller kPa)."""
        source_columns = self.master_poc_source_mappings.get(sevo_name, [])
        conversion_details = {}
        failed_conversions = []
        
        # Hitta första tillgängliga kolumn
        source_col = None
        for col in source_columns:
            if col in vitaldb_data.columns:
                source_col = col
                break
        
        if source_col is None:
            self.logger.warning(f"Ingen källkolumn hittades för {sevo_name}")
            failed_conversions.append(sevo_name)
            return conversion_details, failed_conversions
        
        try:
            source_data = vitaldb_data[source_col]
            # Master POC specifikation: Behåll kPa (1:1 till % om konvertering behövs)
            converted_data[sevo_name] = source_data
            
            conversion_details[sevo_name] = {
                'from_value': source_data.iloc[0] if len(source_data) > 0 else 0.0,
                'to_value': source_data.iloc[0] if len(source_data) > 0 else 0.0,
                'from_unit': 'kPa',
                'to_unit': 'kPa',  # Master POC: behåller kPa
                'source_column': source_col,
                'note': 'Master POC: kPa behålls (1:1 till % om konvertering behövs)'
            }
            
            self.logger.info(f"Konverterade {sevo_name}: {conversion_details[sevo_name]['from_value']:.2f} {conversion_details[sevo_name]['from_unit']} → {conversion_details[sevo_name]['to_value']:.2f} {conversion_details[sevo_name]['to_unit']}")
            
        except Exception as e:
            self.logger.error(f"Fel vid konvertering av {sevo_name}: {e}")
            failed_conversions.append(sevo_name)
        
        return conversion_details, failed_conversions
    
    def _convert_peep(self, vitaldb_data: pd.DataFrame, converted_data: pd.DataFrame) -> Tuple[Dict, List[str]]:
        """Konvertera PEEP från mbar till cmH2O."""
        source_columns = self.master_poc_source_mappings.get('PEEP', [])
        conversion_details = {}
        failed_conversions = []
        
        # Hitta första tillgängliga kolumn
        source_col = None
        for col in source_columns:
            if col in vitaldb_data.columns:
                source_col = col
                break
        
        if source_col is None:
            self.logger.warning("Ingen källkolumn hittades för PEEP")
            failed_conversions.append('PEEP')
            return conversion_details, failed_conversions
        
        try:
            source_data = vitaldb_data[source_col]
            # PEEP: mbar → cmH2O (faktor 1.02)
            mbar = source_data
            cmh2o = mbar * 1.02
            converted_data['PEEP'] = cmh2o
            
            conversion_details['PEEP'] = {
                'from_value': mbar.iloc[0] if len(mbar) > 0 else 0.0,
                'to_value': cmh2o.iloc[0] if len(cmh2o) > 0 else 0.0,
                'from_unit': 'mbar',
                'to_unit': 'cmH2O',
                'source_column': source_col
            }
            
            self.logger.info(f"Konverterade PEEP: {conversion_details['PEEP']['from_value']:.1f} {conversion_details['PEEP']['from_unit']} → {conversion_details['PEEP']['to_value']:.1f} {conversion_details['PEEP']['to_unit']}")
            
        except Exception as e:
            self.logger.error(f"Fel vid konvertering av PEEP: {e}")
            failed_conversions.append('PEEP')
        
        return conversion_details, failed_conversions
    
    def _calculate_ibw(self, height_cm: float, sex: str) -> float:
        """
        Beräkna Ideal Body Weight (IBW) baserat på längd och kön.
        
        Args:
            height_cm: Längd i cm
            sex: Kön ('M' för man, 'F' för kvinna)
            
        Returns:
            IBW i kg
        """
        # Konvertera längd till meter
        height_m = height_cm / 100.0
        
        # IBW formler (Devine formula)
        if sex.upper() in ['M', 'MALE', '1']:
            # Män: IBW = 50 + 2.3 * (höjd i inches - 60)
            height_inches = height_cm / 2.54
            ibw = 50 + 2.3 * (height_inches - 60)
        else:
            # Kvinnor: IBW = 45.5 + 2.3 * (höjd i inches - 60)
            height_inches = height_cm / 2.54
            ibw = 45.5 + 2.3 * (height_inches - 60)
        
        # Säkerställ att IBW är inom rimliga gränser
        ibw = max(30.0, min(150.0, ibw))
        
        return ibw
    
    def _convert_tv(self, vitaldb_data: pd.DataFrame, converted_data: pd.DataFrame, 
                   patient_weight: float) -> Tuple[Dict, List[str]]:
        """
        Konvertera Tidal Volume från ml till ml/kg IBW.
        
        Args:
            vitaldb_data: DataFrame med rådata från VitalDB
            converted_data: DataFrame att skriva konverterad data till
            patient_weight: Patientens vikt i kg (används för fallback)
            
        Returns:
            Tuple med (conversion_details, failed_conversions)
        """
        source_columns = self.master_poc_source_mappings.get('TV', [])
        conversion_details = {}
        failed_conversions = []
        
        # Hitta första tillgängliga kolumn
        source_col = None
        for col in source_columns:
            if col in vitaldb_data.columns:
                source_col = col
                break
        
        if source_col is None:
            self.logger.warning("Ingen källkolumn hittades för TV")
            failed_conversions.append('TV')
            return conversion_details, failed_conversions
        
        try:
            source_data = vitaldb_data[source_col]
            
            # Försök att hämta patientdata för IBW-beräkning
            # Om klinisk data finns i vitaldb_data, använd den
            height_cm = None
            sex = None
            
            # Kolla om klinisk data finns i DataFrame
            if 'height' in vitaldb_data.columns:
                height_cm = vitaldb_data['height'].iloc[0] if len(vitaldb_data) > 0 else None
            if 'sex' in vitaldb_data.columns:
                sex = vitaldb_data['sex'].iloc[0] if len(vitaldb_data) > 0 else None
            
            # Om klinisk data saknas, använd fallback-värden
            if height_cm is None or pd.isna(height_cm) or height_cm <= 0:
                height_cm = 170.0  # Master POC default height
                self.logger.warning("Saknad längd, använder fallback: 170 cm")
            
            if sex is None or pd.isna(sex):
                sex = 'M'  # Master POC default sex
                self.logger.warning("Saknat kön, använder fallback: M")
            
            # Beräkna IBW
            ibw = self._calculate_ibw(height_cm, sex)
            
            # Konvertera TV: ml → ml/kg IBW
            ml_tv = source_data
            ml_kg_ibw = ml_tv / ibw
            converted_data['TV'] = ml_kg_ibw
            
            conversion_details['TV'] = {
                'from_value': ml_tv.iloc[0] if len(ml_tv) > 0 else 0.0,
                'to_value': ml_kg_ibw.iloc[0] if len(ml_kg_ibw) > 0 else 0.0,
                'from_unit': 'ml',
                'to_unit': 'ml/kg IBW',
                'source_column': source_col,
                'ibw_used': ibw,
                'height_cm': height_cm,
                'sex': sex
            }
            
            self.logger.info(f"Konverterade TV: {conversion_details['TV']['from_value']:.1f} {conversion_details['TV']['from_unit']} → {conversion_details['TV']['to_value']:.2f} {conversion_details['TV']['to_unit']} (IBW: {ibw:.1f} kg)")
            
        except Exception as e:
            self.logger.error(f"Fel vid konvertering av TV: {e}")
            failed_conversions.append('TV')
        
        return conversion_details, failed_conversions
    
    def _validate_master_poc_compliance(self, converted_data: pd.DataFrame, 
                                      conversion_details: Dict) -> bool:
        """Validera att konvertering följer Master POC specifikationer."""
        compliance = True
        
        # Kontrollera att alla Master POC features finns
        master_poc_features = ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF', 'etSEV', 'inSev', 'PEEP', 'TV']
        
        for feature in master_poc_features:
            if feature not in converted_data.columns:
                self.logger.warning(f"Master POC feature {feature} saknas efter konvertering")
                compliance = False
        
        # Kontrollera att konverteringar följer Master POC specifikationer
        for feature, details in conversion_details.items():
            if feature in ['Propofol_INF', 'Remifentanil_INF', 'Noradrenalin_INF']:
                # Kontrollera att patient weight fallback användes korrekt
                if details.get('patient_weight', 0) <= 0:
                    self.logger.warning(f"Master POC: Ogiltig patient weight för {feature}")
                    compliance = False
            
            elif feature in ['etSEV', 'inSev']:
                # Kontrollera att kPa behålls
                if details.get('to_unit') != 'kPa':
                    self.logger.warning(f"Master POC: {feature} ska behålla kPa, fick {details.get('to_unit')}")
                    compliance = False
        
        return compliance
    
    def get_master_poc_concentrations(self) -> Dict[str, float]:
        """Hämta Master POC drug concentrations."""
        return self.master_poc_concentrations.copy()
    
    def get_master_poc_default_weight(self) -> float:
        """Hämta Master POC default weight."""
        return self.master_poc_default_weight


def convert_vitaldb_units_master_poc(vitaldb_data: pd.DataFrame, patient_weight: float) -> MasterPOCConversionResult:
    """
    Konvertera VitalDB-enheter till Master POC standardenheter.
    
    Args:
        vitaldb_data: DataFrame med rådata från VitalDB
        patient_weight: Patientens vikt i kg
        
    Returns:
        MasterPOCConversionResult med konverterad data
    """
    converter = MasterPOCUnitConverter()
    return converter.convert_units(vitaldb_data, patient_weight)


# Convenience functions för bakåtkompatibilitet
def get_master_poc_drug_concentrations() -> Dict[str, float]:
    """Hämta Master POC drug concentrations."""
    converter = MasterPOCUnitConverter()
    return converter.get_master_poc_concentrations()


def get_master_poc_default_weight() -> float:
    """Hämta Master POC default weight."""
    converter = MasterPOCUnitConverter()
    return converter.get_master_poc_default_weight()
