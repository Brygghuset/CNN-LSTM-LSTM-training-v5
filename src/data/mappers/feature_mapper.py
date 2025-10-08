"""
FeatureMapper för mappning mellan VitalDB-kolumnnamn och standardiserade namn.
Använder centraliserad konfiguration istället för hårdkodade mappings.
"""

from typing import Dict, List, Set

from config import get_config


class FeatureMapper:
    """Hanterar mappning mellan VitalDB-kolumnnamn och standardiserade namn."""
    
    def __init__(self, config=None):
        """
        Initialisera FeatureMapper med konfiguration eller direkt mapping-dict.
        Args:
            config: dict (feature_mapping) eller config-objekt
        """
        if config is None:
            cfg = get_config()
            self.feature_mapping = cfg.feature_mapping
            self.clinical_columns = set(getattr(cfg, 'clinical_columns', []))
        elif isinstance(config, dict):
            self.feature_mapping = config
            self.clinical_columns = set()
        else:
            # config-objekt
            self.feature_mapping = getattr(config, 'feature_mapping', {})
            self.clinical_columns = set(getattr(config, 'clinical_columns', []))
        # Skapa set för snabbare uppslagning
        self.vitaldb_columns = set(self.feature_mapping.keys())
        self.standard_columns = set(self.feature_mapping.values())
    
    def map_to_standard(self, vitaldb_columns: List[str]) -> Dict[str, str]:
        """
        Mappa VitalDB-kolumnnamn till standardiserade namn.
        
        Args:
            vitaldb_columns: Lista med VitalDB-kolumnnamn
            
        Returns:
            Dict med mappning från VitalDB-namn till standardiserade namn
        """
        mapping = {}
        for col in vitaldb_columns:
            if col in self.feature_mapping:
                mapping[col] = self.feature_mapping[col]
        return mapping
    
    def is_relevant_track(self, track_name: str) -> bool:
        """
        Kontrollera om en track är relevant för vår modell.
        
        Args:
            track_name: Namnet på tracken
            
        Returns:
            True om tracken är relevant
        """
        return track_name in self.vitaldb_columns
    
    def get_expected_columns(self) -> Set[str]:
        """Hämta förväntade standardiserade kolumnnamn."""
        return self.standard_columns.copy()
    
    def get_vitaldb_columns(self) -> Set[str]:
        """Hämta VitalDB-kolumnnamn."""
        return self.vitaldb_columns.copy()
    
    def get_clinical_columns(self) -> Set[str]:
        """Hämta kliniska metadata kolumner."""
        return self.clinical_columns.copy()
    
    def apply_mapping(self, df, rename_columns: bool = True):
        """
        Applicera feature mapping på en DataFrame - minnesoptimerad.
        
        Args:
            df: DataFrame att mappa
            rename_columns: Om kolumner ska döpas om till standardiserade namn
            
        Returns:
            DataFrame med mappade kolumner
        """
        if rename_columns:
            # Skapa mappning för kolumner som finns i DataFrame
            rename_dict = {}
            for col in df.columns:
                if col in self.feature_mapping:
                    rename_dict[col] = self.feature_mapping[col]
            
            if rename_dict:
                # Använd inplace=True för att undvika kopiering
                df = df.rename(columns=rename_dict, inplace=False)  # Behåll False för att behålla return-värde
        
        # Ta bort Time-kolumn om den finns (för testkompatibilitet)
        if 'Time' in df.columns:
            # Använd drop med inplace=False men returnera resultatet
            df = df.drop(columns=['Time'], inplace=False)
        
        return df 