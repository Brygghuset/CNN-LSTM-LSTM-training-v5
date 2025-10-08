"""
ClinicalValidator för validering av klinisk metadata.
Använder centraliserad konfiguration för värdeintervall och validering.
"""

import logging
from typing import Optional, List

import pandas as pd
from data.utils import handle_errors
from config import get_config


class ClinicalValidator:
    """Validator för klinisk metadata."""
    
    def __init__(self, config=None):
        """
        Initialisera ClinicalValidator med konfiguration.
        
        Args:
            config: ConfigManager instance (om None, använd global config)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or get_config()
        
        # Hämta konfigurationsvärden från rätt config-sektion
        if hasattr(self.config, 'data') and self.config.data:
            # Ny config-struktur (ConfigManager)
            self.required_columns = self.config.data.clinical_columns
            # Hämta från safety config om det finns
            if hasattr(self.config, 'safety') and self.config.safety:
                self.value_ranges = getattr(self.config.safety, 'clinical_value_ranges', {})
                self.valid_sex_values = getattr(self.config.safety, 'valid_sex_values', ['M', 'F'])
                # Försök hämta valid_asa_values från safety config
                self.valid_asa_values = getattr(self.config.safety, 'valid_asa_values', [1,2,3,4,5,6])
            else:
                self.value_ranges = {}
                self.valid_sex_values = ['M', 'F']
                self.valid_asa_values = [1,2,3,4,5,6]
        else:
            # Bakåtkompatibilitet för gammal config-struktur
            self.required_columns = getattr(self.config, 'clinical_columns', [
                'age', 'sex', 'height', 'weight', 'bmi', 'asa'
            ])
            self.value_ranges = getattr(self.config, 'clinical_value_ranges', {})
            self.valid_sex_values = getattr(self.config, 'valid_sex_values', ['M', 'F'])
            # Försök hämta valid_asa_values från clinical_value_ranges eller sätt default
            asa_range = self.value_ranges.get('asa', {})
            self.valid_asa_values = asa_range.get('valid_values', [1,2,3,4,5,6])
        # Dokumentation: valid_asa_values används för att validera ASA-klass enligt config eller default [1-6]
    
    @handle_errors(logger=None, default_return=False)
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validera att alla nödvändiga kolumner finns.
        
        Args:
            df: DataFrame att validera
            
        Returns:
            True om alla kolumner finns
        """
        if df is None or df.empty:
            self.logger.warning("DataFrame är None eller tom")
            return False
        
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            self.logger.warning(f"Saknade kolumner: {missing_columns}")
            return False
        
        return True
    
    @handle_errors(logger=None, default_return=False)
    def validate_patient_values(self, df: pd.DataFrame) -> bool:
        """
        Validera att patientvärden är rimliga.
        
        Args:
            df: DataFrame att validera
            
        Returns:
            True om alla värden är rimliga
        """
        if df is None or df.empty:
            return False
        
        # Validera varje rad (patient)
        for idx, row in df.iterrows():
            # Validera ålder
            if 'age' in row:
                age = self._safe_convert_to_float(row['age'])
                age_range = self.value_ranges.get('age', {})
                if not self._validate_numeric_range(age, age_range.get('min', 0), age_range.get('max', 120)):
                    self.logger.warning(f"Otillåten ålder: {age}")
                    return False
            
            # Validera kön
            if 'sex' in row:
                sex = str(row['sex']).strip()
                if sex not in self.valid_sex_values:
                    self.logger.warning(f"Otillåtet kön: {sex}")
                    return False
            
            # Validera längd
            if 'height' in row:
                height = self._safe_convert_to_float(row['height'])
                height_range = self.value_ranges.get('height', {})
                if not self._validate_numeric_range(height, height_range.get('min', 0), height_range.get('max', 250)):
                    self.logger.warning(f"Otillåten längd hittad")
                    return False
            
            # Validera vikt
            if 'weight' in row:
                weight = self._safe_convert_to_float(row['weight'])
                weight_range = self.value_ranges.get('weight', {})
                if not self._validate_numeric_range(weight, weight_range.get('min', 0), weight_range.get('max', 300)):
                    self.logger.warning(f"Otillåten vikt: {weight}")
                    return False
            
            # Validera BMI
            if 'bmi' in row:
                bmi = self._safe_convert_to_float(row['bmi'])
                bmi_range = self.value_ranges.get('bmi', {})
                if not self._validate_numeric_range(bmi, bmi_range.get('min', 0), bmi_range.get('max', 100)):
                    self.logger.warning(f"Otillåten BMI: {bmi}")
                    return False
            
            # Validera ASA-klass
            if 'asa' in row:
                asa = self._safe_convert_to_int(row['asa'])
                if asa not in self.valid_asa_values:
                    self.logger.warning(f"Otillåten ASA-klass: {asa}")
                    return False
        
        return True
    
    def _safe_convert_to_float(self, value) -> float:
        """Säker konvertering till float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return float('nan')
    
    def _safe_convert_to_int(self, value) -> int:
        """Säker konvertering till int."""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return -1  # Invalid value
    
    def _validate_numeric_range(self, value: float, min_val: float, max_val: float) -> bool:
        """Validera att numeriskt värde är inom intervall."""
        if pd.isna(value):
            return False
        return min_val <= value <= max_val 