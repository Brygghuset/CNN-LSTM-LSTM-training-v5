"""
Static Feature Normalization enligt Master POC specifikation.

Implementerar specifika normaliseringsformler för statiska patientfeatures
enligt Master POC CNN-LSTM-LSTM dokumentet.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class StaticFeatureNormalizer:
    """
    Normalisering av statiska patient features enligt Master POC specifikationer.
    
    Använder specifika formler för varje static feature enligt dokumentet:
    - age: age/120 × 2 - 1
    - sex: 1.0 eller -1.0 
    - height: (height-100)/130 × 2 - 1
    - weight: (weight-20)/180 × 2 - 1
    - bmi: (bmi-10)/40 × 2 - 1
    - asa: (asa-1)/5 × 2 - 1
    """
    
    def __init__(self):
        """Initialisera Static Feature Normalizer."""
        self.default_values = {
            'age': 50,
            'sex': -1,  # Default Female
            'height': 170,
            'weight': 70,
            'bmi': 24.2,
            'asa': 2
        }
        
        self.ranges = {
            'age': (0, 120),
            'sex': (-1, 1),
            'height': (100, 230),
            'weight': (20, 200),
            'bmi': (10, 50),
            'asa': (1, 6)
        }
        
        logger.info("Static Feature Normalizer initialiserad med Master POC formler")
    
    def normalize_age(self, age: Union[float, np.ndarray, pd.Series, str]) -> Union[float, np.ndarray]:
        """
        Normalisera ålder: age/120 × 2 - 1
        
        Args:
            age: Ålder i år (0-120) - kan vara string, float eller array
            
        Returns:
            Normaliserad ålder i [-1, 1]
        """
        # Konvertera till numpy array och hantera string-värden
        if isinstance(age, str):
            try:
                age = float(age)
            except (ValueError, TypeError):
                age = self.default_values['age']
        elif isinstance(age, (int, float)):
            age = float(age)
        elif isinstance(age, pd.Series):
            age = age.values
        elif not isinstance(age, np.ndarray):
            age = np.array(age)
        
        # Konvertera till float array
        age = np.array(age, dtype=float)
        
        # Hantera NaN värden med default
        age_filled = np.where(np.isnan(age), self.default_values['age'], age)
        
        # Master POC formula: age/120 × 2 - 1
        normalized = (age_filled / 120) * 2 - 1
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def denormalize_age(self, normalized_age: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Denormalisera ålder: (norm + 1) / 2 * 120
        
        Args:
            normalized_age: Normaliserad ålder i [-1, 1]
            
        Returns:
            Ålder i år
        """
        normalized_age = np.array(normalized_age) if not isinstance(normalized_age, np.ndarray) else normalized_age
        
        # Reverse formula: (norm + 1) / 2 * 120
        age = (normalized_age + 1) / 2 * 120
        
        return age
    
    def normalize_sex(self, sex: Union[str, float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Normalisera kön: -1 = Female, 1 = Male
        
        Args:
            sex: Kön ('F', 'M', 'Female', 'Male', 0, 1, -1, 1)
            
        Returns:
            Normaliserat kön (-1 eller 1)
        """
        if isinstance(sex, (str, float, int)):
            sex = np.array([sex])
        elif isinstance(sex, pd.Series):
            sex = sex.values
        elif not isinstance(sex, np.ndarray):
            sex = np.array(sex)
        
        normalized = np.full(sex.shape, self.default_values['sex'], dtype=float)
        
        for i, value in enumerate(sex):
            if pd.isna(value):
                normalized[i] = self.default_values['sex']  # Default Female
            elif isinstance(value, str):
                # Försök konvertera till numerisk värde först
                try:
                    numeric_val = float(value)
                    if numeric_val > 0:
                        normalized[i] = 1.0
                    else:
                        normalized[i] = -1.0
                except ValueError:
                    # Hantera som text-string
                    if value.upper() in ['M', 'MALE', 'MAN']:
                        normalized[i] = 1.0
                    elif value.upper() in ['F', 'FEMALE', 'WOMAN']:
                        normalized[i] = -1.0
                    else:
                        normalized[i] = self.default_values['sex']
            elif isinstance(value, (int, float, np.integer, np.floating)):
                if value > 0:
                    normalized[i] = 1.0  # Male
                else:
                    normalized[i] = -1.0  # Female
        
        return normalized if len(normalized) > 1 else normalized[0]
    
    def denormalize_sex(self, normalized_sex: Union[float, np.ndarray, pd.Series]) -> Union[str, np.ndarray]:
        """
        Denormalisera kön tillbaka till string representation.
        
        Args:
            normalized_sex: Normaliserat kön (-1 eller 1)
            
        Returns:
            Kön som string ('Female' eller 'Male')
        """
        normalized_sex = np.array(normalized_sex) if not isinstance(normalized_sex, np.ndarray) else normalized_sex
        
        sex_strings = np.where(normalized_sex > 0, 'Male', 'Female')
        
        return sex_strings if len(sex_strings) > 1 else sex_strings[0]
    
    def normalize_height(self, height: Union[float, np.ndarray, pd.Series, str]) -> Union[float, np.ndarray]:
        """
        Normalisera längd: (height-100)/130 × 2 - 1
        
        Args:
            height: Längd i cm (100-230) - kan vara string, float eller array
            
        Returns:
            Normaliserad längd i [-1, 1]
        """
        # Konvertera till numpy array och hantera string-värden
        if isinstance(height, str):
            try:
                height = float(height)
            except (ValueError, TypeError):
                height = self.default_values['height']
        elif isinstance(height, (int, float)):
            height = float(height)
        elif isinstance(height, pd.Series):
            height = height.values
        elif not isinstance(height, np.ndarray):
            height = np.array(height)
        
        # Konvertera till float array
        height = np.array(height, dtype=float)
        
        # Hantera NaN värden med default
        height_filled = np.where(np.isnan(height), self.default_values['height'], height)
        
        # Master POC formula: (height-100)/130 × 2 - 1
        normalized = ((height_filled - 100) / 130) * 2 - 1
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def denormalize_height(self, normalized_height: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Denormalisera längd: (norm + 1) / 2 * 130 + 100
        
        Args:
            normalized_height: Normaliserad längd i [-1, 1]
            
        Returns:
            Längd i cm
        """
        normalized_height = np.array(normalized_height) if not isinstance(normalized_height, np.ndarray) else normalized_height
        
        # Reverse formula: (norm + 1) / 2 * 130 + 100
        height = (normalized_height + 1) / 2 * 130 + 100
        
        return height
    
    def normalize_weight(self, weight: Union[float, np.ndarray, pd.Series, str]) -> Union[float, np.ndarray]:
        """
        Normalisera vikt: (weight-20)/180 × 2 - 1
        
        Args:
            weight: Vikt i kg (20-200) - kan vara string, float eller array
            
        Returns:
            Normaliserad vikt i [-1, 1]
        """
        # Konvertera till numpy array och hantera string-värden
        if isinstance(weight, str):
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                weight = self.default_values['weight']
        elif isinstance(weight, (int, float)):
            weight = float(weight)
        elif isinstance(weight, pd.Series):
            weight = weight.values
        elif not isinstance(weight, np.ndarray):
            weight = np.array(weight)
        
        # Konvertera till float array
        weight = np.array(weight, dtype=float)
        
        # Hantera NaN värden med default
        weight_filled = np.where(np.isnan(weight), self.default_values['weight'], weight)
        
        # Master POC formula: (weight-20)/180 × 2 - 1
        normalized = ((weight_filled - 20) / 180) * 2 - 1
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def denormalize_weight(self, normalized_weight: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Denormalisera vikt: (norm + 1) / 2 * 180 + 20
        
        Args:
            normalized_weight: Normaliserad vikt i [-1, 1]
            
        Returns:
            Vikt i kg
        """
        normalized_weight = np.array(normalized_weight) if not isinstance(normalized_weight, np.ndarray) else normalized_weight
        
        # Reverse formula: (norm + 1) / 2 * 180 + 20
        weight = (normalized_weight + 1) / 2 * 180 + 20
        
        return weight
    
    def normalize_bmi(self, bmi: Union[float, np.ndarray, pd.Series, str]) -> Union[float, np.ndarray]:
        """
        Normalisera BMI: (bmi-10)/40 × 2 - 1
        
        Args:
            bmi: BMI i kg/m² (10-50) - kan vara string, float eller array
            
        Returns:
            Normaliserat BMI i [-1, 1]
        """
        # Konvertera till numpy array och hantera string-värden
        if isinstance(bmi, str):
            try:
                bmi = float(bmi)
            except (ValueError, TypeError):
                bmi = self.default_values['bmi']
        elif isinstance(bmi, (int, float)):
            bmi = float(bmi)
        elif isinstance(bmi, pd.Series):
            bmi = bmi.values
        elif not isinstance(bmi, np.ndarray):
            bmi = np.array(bmi)
        
        # Konvertera till float array
        bmi = np.array(bmi, dtype=float)
        
        # Hantera NaN värden med default
        bmi_filled = np.where(np.isnan(bmi), self.default_values['bmi'], bmi)
        
        # Master POC formula: (bmi-10)/40 × 2 - 1
        normalized = ((bmi_filled - 10) / 40) * 2 - 1
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def denormalize_bmi(self, normalized_bmi: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Denormalisera BMI: (norm + 1) / 2 * 40 + 10
        
        Args:
            normalized_bmi: Normaliserat BMI i [-1, 1]
            
        Returns:
            BMI i kg/m²
        """
        normalized_bmi = np.array(normalized_bmi) if not isinstance(normalized_bmi, np.ndarray) else normalized_bmi
        
        # Reverse formula: (norm + 1) / 2 * 40 + 10
        bmi = (normalized_bmi + 1) / 2 * 40 + 10
        
        return bmi
    
    def normalize_asa(self, asa: Union[float, np.ndarray, pd.Series, str]) -> Union[float, np.ndarray]:
        """
        Normalisera ASA score: (asa-1)/5 × 2 - 1
        
        Args:
            asa: ASA score (1-6) - kan vara string, float eller array
            
        Returns:
            Normaliserat ASA i [-1, 1]
        """
        # Konvertera till numpy array och hantera string-värden
        if isinstance(asa, str):
            try:
                asa = float(asa)
            except (ValueError, TypeError):
                asa = self.default_values['asa']
        elif isinstance(asa, (int, float)):
            asa = float(asa)
        elif isinstance(asa, pd.Series):
            asa = asa.values
        elif not isinstance(asa, np.ndarray):
            asa = np.array(asa)
        
        # Konvertera till float array
        asa = np.array(asa, dtype=float)
        
        # Hantera NaN värden med default
        asa_filled = np.where(np.isnan(asa), self.default_values['asa'], asa)
        
        # Master POC formula: (asa-1)/5 × 2 - 1
        normalized = ((asa_filled - 1) / 5) * 2 - 1
        
        # Clamp till [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def denormalize_asa(self, normalized_asa: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Denormalisera ASA score: (norm + 1) / 2 * 5 + 1
        
        Args:
            normalized_asa: Normaliserat ASA i [-1, 1]
            
        Returns:
            ASA score (1-6)
        """
        normalized_asa = np.array(normalized_asa) if not isinstance(normalized_asa, np.ndarray) else normalized_asa
        
        # Reverse formula: (norm + 1) / 2 * 5 + 1
        asa = (normalized_asa + 1) / 2 * 5 + 1
        
        # Avrunda till närmaste heltal (ASA är diskret)
        asa = np.round(asa)
        
        return asa
    
    def normalize_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalisera alla statiska features i en DataFrame.
        
        Args:
            df: DataFrame med statiska features
            
        Returns:
            DataFrame med normaliserade statiska features
        """
        result = df.copy()
        
        normalization_methods = {
            'age': self.normalize_age,
            'sex': self.normalize_sex,
            'height': self.normalize_height,
            'weight': self.normalize_weight,
            'bmi': self.normalize_bmi,
            'asa': self.normalize_asa
        }
        
        for feature, method in normalization_methods.items():
            if feature in result.columns:
                logger.debug(f"Normaliserar {feature} med Master POC formel")
                result[feature] = method(result[feature])
        
        logger.info(f"Normaliserade {len([f for f in normalization_methods.keys() if f in df.columns])} statiska features")
        
        return result
    
    def denormalize_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalisera alla statiska features i en DataFrame.
        
        Args:
            df: DataFrame med normaliserade statiska features
            
        Returns:
            DataFrame med denormaliserade statiska features
        """
        result = df.copy()
        
        denormalization_methods = {
            'age': self.denormalize_age,
            'sex': self.denormalize_sex,
            'height': self.denormalize_height,
            'weight': self.denormalize_weight,
            'bmi': self.denormalize_bmi,
            'asa': self.denormalize_asa
        }
        
        for feature, method in denormalization_methods.items():
            if feature in result.columns:
                logger.debug(f"Denormaliserar {feature} med Master POC formel")
                result[feature] = method(result[feature])
        
        logger.info(f"Denormaliserade {len([f for f in denormalization_methods.keys() if f in df.columns])} statiska features")
        
        return result
    
    def extract_and_normalize_static_features(self, clinical_df: Optional[pd.DataFrame]) -> np.ndarray:
        """
        Extrahera och normalisera statiska features från clinical DataFrame.
        
        Args:
            clinical_df: DataFrame med kliniska data
            
        Returns:
            Numpy array med normaliserade statiska features [age, sex, height, weight, bmi, asa]
        """
        if clinical_df is None or clinical_df.empty:
            logger.warning("Ingen klinisk data tillgänglig, använder default-värden")
            # Returnera normaliserade default-värden
            return np.array([
                self.normalize_age(self.default_values['age']),
                self.normalize_sex(self.default_values['sex']),
                self.normalize_height(self.default_values['height']),
                self.normalize_weight(self.default_values['weight']),
                self.normalize_bmi(self.default_values['bmi']),
                self.normalize_asa(self.default_values['asa'])
            ])
        
        # Extrahera första raden (antar att alla rader har samma patientdata)
        patient_data = clinical_df.iloc[0] if len(clinical_df) > 0 else pd.Series()
        
        # Extrahera och normalisera varje feature
        static_features = []
        
        # Age
        age = patient_data.get('age', self.default_values['age'])
        static_features.append(self.normalize_age(age))
        
        # Sex
        sex = patient_data.get('sex', self.default_values['sex'])
        static_features.append(self.normalize_sex(sex))
        
        # Height
        height = patient_data.get('height', self.default_values['height'])
        static_features.append(self.normalize_height(height))
        
        # Weight
        weight = patient_data.get('weight', self.default_values['weight'])
        static_features.append(self.normalize_weight(weight))
        
        # BMI
        bmi = patient_data.get('bmi', self.default_values['bmi'])
        static_features.append(self.normalize_bmi(bmi))
        
        # ASA
        asa = patient_data.get('asa', self.default_values['asa'])
        static_features.append(self.normalize_asa(asa))
        
        return np.array(static_features, dtype=np.float32)


def create_static_normalizer() -> StaticFeatureNormalizer:
    """Factory function för att skapa Static Feature Normalizer."""
    return StaticFeatureNormalizer()
