"""
VitalLoader för .vital filer med centraliserad felhantering.
"""

import logging
from typing import Optional

import pandas as pd
from data.utils import handle_errors, safe_pandas_read
from data import VITALDB_AVAILABLE


class VitalLoader:
    """Loader för .vital filer med fallback till CSV."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, filepath: str) -> bool:
        """Kontrollera om denna loader kan hantera filen."""
        return filepath.lower().endswith('.vital')
    
    def load(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Ladda .vital fil med fallback till CSV.
        
        Args:
            filepath: Sökväg till .vital filen
            
        Returns:
            DataFrame eller None vid fel
        """
        # Undantag tillåts bubbla upp för TDD/testbarhet
        # Försök med VitalDB-biblioteket först
        if VITALDB_AVAILABLE:
            df = self._load_with_vitaldb(filepath)
            if df is not None:
                return df
        # Fallback till CSV-läsning
        return self._load_as_csv(filepath)
    
    def _load_with_vitaldb(self, filepath: str) -> Optional[pd.DataFrame]:
        """Ladda med VitalDB-biblioteket."""
        try:
            import vitaldb as vdb
            
            # Läs .vital fil
            vf = vdb.read_vital(filepath)
            
            # Hämta alla tillgängliga track names
            track_names = vf.get_track_names()
            
            if not track_names:
                self.logger.warning(f"Inga tracks hittades i {filepath}")
                return None
            
            # Konvertera till pandas DataFrame med 1 sekunds intervall
            df = vf.to_pandas(track_names=track_names, interval=1.0)
            
            if df is None or df.empty:
                self.logger.warning(f"Tom DataFrame från {filepath}")
                return None
            
            # Konvertera datatyper för numeriska kolumner
            df = self._convert_to_numeric(df)
            
            # Filtrera bara numeriska tracks för prestanda
            numeric_columns = df.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) > 0:
                df = df[numeric_columns]
                self.logger.info(f"Laddade {len(df)} rader från {filepath} med VitalDB, {len(numeric_columns)} numeriska kolumner")
                return df
            else:
                self.logger.warning(f"Inga numeriska kolumner hittades i {filepath}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Kunde inte läsa {filepath} med VitalDB: {e}")
            return None
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Konvertera datatyper för numeriska kolumner."""
        # Först identifiera vilka kolumner som troligen är numeriska
        # genom att testa konvertering på första icke-null värdet
        numeric_convertible_cols = []
        
        for col in df.columns:
            # Hitta första icke-null värdet för att testa
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                test_value = non_null_values.iloc[0]
                try:
                    # Testa om värdet kan konverteras till numerisk
                    pd.to_numeric([test_value], errors='raise')
                    numeric_convertible_cols.append(col)
                except (ValueError, TypeError):
                    # Kolumnen är inte numerisk, skippa den
                    continue
        
        self.logger.info(f"Hittade {len(numeric_convertible_cols)} relevanta numeriska tracks")
        
        # Konvertera endast kolumner som kan vara numeriska
        for col in numeric_convertible_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info(f"Konverterade {len(numeric_convertible_cols)} kolumner till numeriska datatyper")
        
        # Ta bort kolumner som inte är numeriska
        df_numeric = df[numeric_convertible_cols]
        
        return df_numeric
    
    def _load_as_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Fallback: Ladda som CSV."""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Laddade {len(df)} rader från {filepath} som CSV")
            return df
        except Exception as e:
            self.logger.error(f"Kunde inte läsa {filepath} som CSV: {e}")
            return None 