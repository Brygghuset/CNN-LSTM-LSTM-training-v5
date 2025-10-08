"""
Error handling utilities för centraliserad felhantering.
Eliminerar kodduplicering i try-except blocks.
"""

import logging
import functools
from typing import Callable, Any, Optional, Union
from contextlib import contextmanager
import os

import pandas as pd

# VitalDB imports (optional)
try:
    import vitaldb as vdb
    VITALDB_AVAILABLE = True
except ImportError:
    VITALDB_AVAILABLE = False


def handle_errors(
    logger: Optional[logging.Logger] = None,
    default_return: Any = None,
    reraise: bool = False
):
    """
    Decorator för centraliserad felhantering.
    
    Args:
        logger: Logger att använda för felmeddelanden
        default_return: Vad att returnera vid fel
        reraise: Om fel ska kastas igen
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(func.__module__)
                log.error(f"Fel i {func.__name__}: {str(e)}")
                
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


@contextmanager
def safe_file_operation(
    filepath: str,
    logger: Optional[logging.Logger] = None,
    operation: str = "operation"
):
    """
    Context manager för säkra filoperationer.
    
    Args:
        filepath: Sökväg till filen
        logger: Logger att använda
        operation: Beskrivning av operationen
    """
    log = logger or logging.getLogger(__name__)
    
    try:
        log.debug(f"Startar {operation} på {filepath}")
        yield
        log.debug(f"Slutförde {operation} på {filepath}")
    except FileNotFoundError:
        log.warning(f"Filen hittades inte: {filepath}")
        raise
    except PermissionError:
        log.error(f"Behörighetsfel för fil: {filepath}")
        raise
    except Exception as e:
        log.error(f"Fel under {operation} på {filepath}: {str(e)}")
        raise


def safe_pandas_read(
    filepath: str,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Säker pandas read-operation med centraliserad felhantering.
    
    Args:
        filepath: Sökväg till filen
        logger: Logger att använda
        **kwargs: Argument till pd.read_csv eller liknande
    
    Returns:
        DataFrame eller None vid fel
    """
    log = logger or logging.getLogger(__name__)
    
    # Kontextbaserad felhantering
    from config import get_config
    config = get_config()
    
    try:
        if filepath.endswith('.csv'):
            return _read_csv_file(filepath, log, **kwargs)
        elif filepath.endswith('.vital'):
            return _read_vital_file(filepath, log, **kwargs)
        else:
            log.warning(f"Okänt filformat: {filepath}")
            return None
            
    except Exception as e:
        log.error(f"Korrupt fil vid läsning av {filepath}: {str(e)}")
        
        # Använd kontextbaserad felhantering för korrupta filer
        should_raise = config.should_raise_on_corrupted_files
        log.debug(f"Config says should_raise_on_corrupted_files: {should_raise}")
        if should_raise:
            log.debug(f"Raising exception for corrupted file: {filepath}")
            raise
        
        # Graceful handling: returnera None
        log.debug(f"Returning None for corrupted file: {filepath}")
        return None


def _read_csv_file(filepath: str, logger: logging.Logger, **kwargs) -> pd.DataFrame:
    """Läs CSV-fil."""
    return pd.read_csv(filepath, **kwargs)


def _read_vital_file(filepath: str, logger: logging.Logger, **kwargs) -> Optional[pd.DataFrame]:
    """Läs .vital-fil med VitalDB eller fallback till CSV."""
    from config import get_config
    config = get_config()
    
    # Försök med VitalDB först
    if VITALDB_AVAILABLE:
        try:
            df = _read_with_vitaldb(filepath, logger)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"VitalDB parsing misslyckades för {filepath}: {e}")
            # Fallback till CSV alltid, även i testmiljö
    # Fallback till CSV alltid, även om VitalDB misslyckas
    try:
        logger.info(f"Fallback till CSV för {filepath}")
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Korrupt fil: både VitalDB och CSV-läsning misslyckades för {filepath}: {e}")
        if config.should_raise_on_corrupted_files:
            raise
        return None


def _read_with_vitaldb(filepath: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Läs .vital-fil med VitalDB-biblioteket."""
    from config import get_config
    config = get_config()
    
    try:
        vf = vdb.read_vital(filepath)
        # Hämta alla tillgängliga track names
        track_names = vf.get_track_names()
        if not track_names:
            logger.warning(f"Inga tracks hittades i {filepath}")
            return None
        # Konvertera till pandas DataFrame med 1 sekunds intervall
        df = vf.to_pandas(track_names=track_names, interval=1.0)
        if df is not None and not df.empty:
            return _process_vitaldb_dataframe(df, logger)
        else:
            logger.warning(f"Tom DataFrame från {filepath}")
            return None
    except Exception as e:
        logger.warning(f"VitalDB parsing misslyckades för {filepath}: {e}")
        if config.should_raise_on_corrupted_files:
            raise
        return None


def _process_vitaldb_dataframe(df: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Processera VitalDB DataFrame för numeriska kolumner."""
    # Identifiera numeriska kolumner
    numeric_convertible_cols = _identify_numeric_columns(df)
    
    if not numeric_convertible_cols:
        logger.warning("Inga numeriska kolumner hittades")
        return None
    
    logger.info(f"Hittade {len(numeric_convertible_cols)} relevanta numeriska tracks")
    
    # Konvertera till numeriska datatyper
    for col in numeric_convertible_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Konverterade {len(numeric_convertible_cols)} kolumner till numeriska datatyper")
    
    # Returnera endast numeriska kolumner
    return df[numeric_convertible_cols]


def _identify_numeric_columns(df: pd.DataFrame) -> list:
    """Identifiera vilka kolumner som kan konverteras till numeriska."""
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
    
    return numeric_convertible_cols


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validera en DataFrame med centraliserad logik.
    
    Args:
        df: DataFrame att validera
        required_columns: Kolumner som måste finnas
        min_rows: Minsta antal rader
        logger: Logger att använda
    
    Returns:
        True om validering lyckas
    """
    log = logger or logging.getLogger(__name__)
    
    if df is None or df.empty:
        log.warning("DataFrame är None eller tom")
        return False
    
    if len(df) < min_rows:
        log.warning(f"DataFrame har för få rader: {len(df)} < {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            log.warning(f"Saknade kolumner: {missing_cols}")
            return False
    
    return True 