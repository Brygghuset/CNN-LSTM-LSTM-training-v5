"""
CSVLoader för .csv filer med centraliserad felhantering.
"""

import logging
from typing import Optional

import pandas as pd
from data.utils import handle_errors, safe_pandas_read


class CSVLoader:
    """Loader för .csv filer."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, filepath: str) -> bool:
        """Kontrollera om denna loader kan hantera filen."""
        return filepath.lower().endswith('.csv')
    
    def load(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Ladda .csv fil.
        
        Args:
            filepath: Sökväg till .csv filen
            
        Returns:
            DataFrame eller None vid fel
        """
        # Undantag tillåts bubbla upp för TDD/testbarhet
        return safe_pandas_read(filepath, logger=self.logger) 