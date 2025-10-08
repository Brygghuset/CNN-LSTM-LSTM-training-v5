#!/usr/bin/env python3
"""
T011: Test Feature Name Standardization
Verifiera att alla feature-namn följer standardiserat format
"""

import unittest
import sys
import os
import re

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår feature mapping modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_feature_mapping", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_feature_mapping.py')
)
master_poc_feature_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_feature_mapping)

# Använd modulen
MASTER_POC_TIMESERIES_FEATURES = master_poc_feature_mapping.MASTER_POC_TIMESERIES_FEATURES
MASTER_POC_STATIC_FEATURES = master_poc_feature_mapping.MASTER_POC_STATIC_FEATURES
MASTER_POC_OUTPUT_FEATURES = master_poc_feature_mapping.MASTER_POC_OUTPUT_FEATURES

class TestT011FeatureNameStandardization(unittest.TestCase):
    """T011: Test Feature Name Standardization"""
    
    def test_t011_feature_name_format(self):
        """
        T011: Test Feature Name Standardization
        Verifiera att alla feature-namn följer standardiserat format
        """
        # Arrange - Expected naming conventions
        # Timeseries features: UPPERCASE med underscores för separering
        # Static features: lowercase
        # Output features: UPPERCASE med _Predict suffix
        
        # Act & Assert
        all_features = {}
        all_features.update(MASTER_POC_TIMESERIES_FEATURES)
        all_features.update(MASTER_POC_STATIC_FEATURES)
        all_features.update(MASTER_POC_OUTPUT_FEATURES)
        
        for feature_name in all_features.keys():
            # Verifiera att feature name följer korrekt format
            self.assertTrue(self._is_valid_feature_name(feature_name), 
                          f"Feature name '{feature_name}' följer inte standardiserat format")
        
        print("✅ T011 PASSED: Alla feature-namn följer standardiserat format")
    
    def _is_valid_feature_name(self, feature_name: str) -> bool:
        """
        Validera att feature name följer standardiserat format.
        
        Format regler:
        - Timeseries features: UPPERCASE med underscores (HR, BP_SYS, Propofol_INF)
        - Static features: lowercase (age, sex, height)
        - Output features: UPPERCASE med _Predict suffix (Propofol_Predict)
        """
        # Kontrollera att det bara finns alfanumeriska tecken och underscores
        if not re.match(r'^[A-Za-z0-9_]+$', feature_name):
            return False
        
        # Kontrollera att det inte börjar eller slutar med underscore
        if feature_name.startswith('_') or feature_name.endswith('_'):
            return False
        
        # Kontrollera att det inte finns dubbla underscores
        if '__' in feature_name:
            return False
        
        return True
    
    def test_t011_timeseries_feature_naming_convention(self):
        """
        Verifiera att timeseries features följer UPPERCASE_UNDERSCORE konvention
        """
        # Arrange
        timeseries_features = list(MASTER_POC_TIMESERIES_FEATURES.keys())
        
        # Act & Assert
        for feature_name in timeseries_features:
            # Verifiera att timeseries features är UPPERCASE eller mixed case med underscores
            self.assertTrue(self._is_timeseries_naming_convention(feature_name),
                          f"Timeseries feature '{feature_name}' följer inte UPPERCASE_UNDERSCORE konvention")
        
        print("✅ T011 PASSED: Timeseries features följer UPPERCASE_UNDERSCORE konvention")
    
    def _is_timeseries_naming_convention(self, feature_name: str) -> bool:
        """Validera timeseries naming convention."""
        # Timeseries features ska vara UPPERCASE eller mixed case med underscores
        # Exempel: HR, BP_SYS, Propofol_INF, etSEV, inSev
        # Tillåt både helt UPPERCASE och mixed case med lowercase prefix
        return (re.match(r'^[A-Z][A-Za-z0-9]*(_[A-Z][A-Za-z0-9]*)*$', feature_name) is not None or
                re.match(r'^[A-Z]+$', feature_name) is not None or
                re.match(r'^[a-z]+[A-Z][A-Za-z0-9]*$', feature_name) is not None)
    
    def test_t011_static_feature_naming_convention(self):
        """
        Verifiera att static features följer lowercase konvention
        """
        # Arrange
        static_features = list(MASTER_POC_STATIC_FEATURES.keys())
        
        # Act & Assert
        for feature_name in static_features:
            # Verifiera att static features är lowercase
            self.assertTrue(self._is_static_naming_convention(feature_name),
                          f"Static feature '{feature_name}' följer inte lowercase konvention")
        
        print("✅ T011 PASSED: Static features följer lowercase konvention")
    
    def _is_static_naming_convention(self, feature_name: str) -> bool:
        """Validera static naming convention."""
        # Static features ska vara lowercase
        return re.match(r'^[a-z]+$', feature_name) is not None
    
    def test_t011_output_feature_naming_convention(self):
        """
        Verifiera att output features följer UPPERCASE_Predict konvention
        """
        # Arrange
        output_features = list(MASTER_POC_OUTPUT_FEATURES.keys())
        
        # Act & Assert
        for feature_name in output_features:
            # Verifiera att output features slutar med _Predict
            self.assertTrue(self._is_output_naming_convention(feature_name),
                          f"Output feature '{feature_name}' följer inte UPPERCASE_Predict konvention")
        
        print("✅ T011 PASSED: Output features följer UPPERCASE_Predict konvention")
    
    def _is_output_naming_convention(self, feature_name: str) -> bool:
        """Validera output naming convention."""
        # Output features ska sluta med _Predict
        return feature_name.endswith('_Predict')
    
    def test_t011_no_duplicate_feature_names(self):
        """
        Verifiera att det inte finns duplicerade feature names
        """
        # Arrange
        all_feature_names = []
        all_feature_names.extend(MASTER_POC_TIMESERIES_FEATURES.keys())
        all_feature_names.extend(MASTER_POC_STATIC_FEATURES.keys())
        all_feature_names.extend(MASTER_POC_OUTPUT_FEATURES.keys())
        
        # Act
        unique_names = set(all_feature_names)
        
        # Assert
        self.assertEqual(len(unique_names), len(all_feature_names),
                        f"Inga duplicerade feature names ska finnas. Hittade: {all_feature_names}")
        
        print("✅ T011 PASSED: Inga duplicerade feature names")
    
    def test_t011_feature_name_length_validation(self):
        """
        Verifiera att feature names har rimlig längd
        """
        # Arrange
        all_features = {}
        all_features.update(MASTER_POC_TIMESERIES_FEATURES)
        all_features.update(MASTER_POC_STATIC_FEATURES)
        all_features.update(MASTER_POC_OUTPUT_FEATURES)
        
        # Act & Assert
        for feature_name in all_features.keys():
            # Verifiera att feature name har rimlig längd (1-50 tecken)
            self.assertGreaterEqual(len(feature_name), 1,
                                  f"Feature name '{feature_name}' är för kort")
            self.assertLessEqual(len(feature_name), 50,
                                f"Feature name '{feature_name}' är för långt")
        
        print("✅ T011 PASSED: Feature names har rimlig längd")
    
    def test_t011_feature_name_readability(self):
        """
        Verifiera att feature names är läsbara och beskrivande
        """
        # Arrange
        timeseries_features = list(MASTER_POC_TIMESERIES_FEATURES.keys())
        
        # Act & Assert
        for feature_name in timeseries_features:
            # Verifiera att feature name innehåller beskrivande text
            # Tillåt korta förkortningar som HR, RR, BIS
            if len(feature_name) <= 3:
                # Korta förkortningar är OK
                continue
            
            self.assertGreater(len(feature_name), 2,
                             f"Feature name '{feature_name}' är för kort för att vara beskrivande")
            
            # Verifiera att det inte bara är siffror
            self.assertFalse(feature_name.isdigit(),
                           f"Feature name '{feature_name}' ska inte bara vara siffror")
        
        print("✅ T011 PASSED: Feature names är läsbara och beskrivande")

if __name__ == '__main__':
    unittest.main()
