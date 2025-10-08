"""
ClinicalLoader för klinisk data med centraliserad felhantering.
"""

import logging
from typing import Optional

import pandas as pd
from data.utils import handle_errors, safe_pandas_read, validate_dataframe


class ClinicalLoader:
    """Loader för klinisk data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, filepath: str) -> bool:
        """Kontrollera om denna loader kan hantera filen."""
        filename = filepath.lower()
        return ('clinical' in filename or 'info' in filename or filename.endswith('clinical_data.csv'))
    
    def load(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Ladda klinisk data.
        
        Args:
            filepath: Sökväg till klinisk data-fil
            
        Returns:
            DataFrame eller None vid fel
        """
        # Undantag tillåts bubbla upp för TDD/testbarhet
        df = safe_pandas_read(filepath, logger=self.logger)
        
        if df is not None:
            # Validera att det är klinisk data
            if not validate_dataframe(df, min_rows=1, logger=self.logger):
                return None
            
            # Konvertera caseid till case_id för konsistens
            if 'caseid' in df.columns and 'case_id' not in df.columns:
                df = df.rename(columns={'caseid': 'case_id'})
        
        return df
    
    def get_case_data(self, df: pd.DataFrame, case_id) -> Optional[pd.DataFrame]:
        """
        Hämta data för en specifik case från klinisk data.
        
        Args:
            df: Klinisk data DataFrame
            case_id: ID för casen att hämta (kan vara sträng eller int)
            
        Returns:
            DataFrame för specifik case eller None
        """
        if df is None or df.empty:
            return None
        
        # Försök hitta case_id kolumn
        case_col = None
        for col in ['case_id', 'caseid', 'case']:
            if col in df.columns:
                case_col = col
                break
        
        if case_col is None:
            self.logger.warning("Ingen case ID kolumn hittades")
            return None
        
        # Konvertera case_id till int för matchning (hantera både "0001" och 1)
        try:
            if isinstance(case_id, str):
                # Ta bort ledande nollor och konvertera till int
                case_id_int = int(case_id.lstrip('0') or '0')
            else:
                case_id_int = int(case_id)
        except (ValueError, TypeError):
            self.logger.warning(f"Ogiltigt case_id format: {case_id}")
            return None
        
        # Filtrera för specifik case
        case_data = df[df[case_col] == case_id_int]
        
        if case_data.empty:
            self.logger.warning(f"Inga data hittades för case {case_id} (sökte efter {case_id_int})")
            return None
        
        return case_data 