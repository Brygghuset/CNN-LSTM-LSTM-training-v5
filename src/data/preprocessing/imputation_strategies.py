"""
Imputeringsstrategier för saknade värden.

Varje strategi implementerar en specifik metod för att hantera saknade värden
enligt Strategy Pattern.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ImputationStrategy(ABC):
    """Abstract base class för imputeringsstrategier."""
    
    @abstractmethod
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera saknade värden i en serie."""
        pass


class ForwardFillStrategy(ImputationStrategy):
    """Forward fill strategi med smart och klassisk hantering."""
    
    def __init__(self, use_smart_fill: bool = True):
        """
        Args:
            use_smart_fill: Om True används smart fill med mean-imputation för isolerade NaN
        """
        self.use_smart_fill = use_smart_fill
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med forward fill."""
        if self.use_smart_fill:
            return self._smart_forward_fill(series, max_consecutive_nans)
        else:
            return self._classic_forward_fill(series, max_consecutive_nans)
    
    def _classic_forward_fill(self, series: pd.Series, 
                            max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Klassisk forward fill utan smart mean-imputation för isolerade NaN."""
        imputed = series.copy()
        
        # Klassisk forward fill
        if max_consecutive_nans is not None:
            imputed = imputed.ffill(limit=max_consecutive_nans)
        else:
            imputed = imputed.ffill()
        
        # Om det fortfarande finns NaN-värden (t.ex. initiala NaN), använd backward fill som fallback
        if imputed.isna().any():
            logger.info("Klassisk forward fill kunde inte hantera alla NaN-värden, använder backward fill fallback")
            if max_consecutive_nans is not None:
                imputed = imputed.bfill(limit=max_consecutive_nans)
            else:
                imputed = imputed.bfill()
        
        return imputed
    
    def _smart_forward_fill(self, series: pd.Series, 
                          max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Smart forward fill med mean-imputation för isolerade NaN-värden."""
        # Skapa kopia för att undvika att modifiera original
        imputed = series.copy()
        
        # Identifiera isolerade NaN (NaN mellan två giltiga värden)
        isolated_nans = []
        for i in range(1, len(series) - 1):
            if pd.isna(series.iloc[i]) and not pd.isna(series.iloc[i-1]) and not pd.isna(series.iloc[i+1]):
                isolated_nans.append(i)
        
        # Hantera isolerade NaN direkt med mean-imputation
        if isolated_nans:
            logger.info(f"Hittade {len(isolated_nans)} isolerade NaN-värden, använder mean-imputation")
            mean_value = series.mean()
            for idx in isolated_nans:
                imputed.iloc[idx] = mean_value
        
        # Först försök med forward fill för kvarvarande NaN
        if max_consecutive_nans is not None:
            imputed = imputed.ffill(limit=max_consecutive_nans)
        else:
            imputed = imputed.ffill()
        
        # Om det fortfarande finns NaN-värden (t.ex. initiala NaN), använd backward fill som fallback
        if imputed.isna().any():
            logger.info("Smart forward fill kunde inte hantera alla NaN-värden, använder backward fill fallback")
            if max_consecutive_nans is not None:
                imputed = imputed.bfill(limit=max_consecutive_nans)
            else:
                imputed = imputed.bfill()
        
        return imputed


class BackwardFillStrategy(ImputationStrategy):
    """Backward fill strategi."""
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med backward fill."""
        imputed = series.copy()
        
        if max_consecutive_nans is not None:
            imputed = imputed.bfill(limit=max_consecutive_nans)
        else:
            imputed = imputed.bfill()
        
        # Fallback till forward fill om backward fill inte räcker
        if imputed.isna().any():
            logger.info("Backward fill kunde inte hantera alla NaN-värden, använder forward fill fallback")
            if max_consecutive_nans is not None:
                imputed = imputed.ffill(limit=max_consecutive_nans)
            else:
                imputed = imputed.ffill()
        
        return imputed


class LinearInterpolationStrategy(ImputationStrategy):
    """Linjär interpolation strategi."""
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med linjär interpolation."""
        imputed = series.copy()
        
        # Använd pandas interpolate med limit om angiven
        if max_consecutive_nans is not None:
            imputed = imputed.interpolate(method='linear', limit=max_consecutive_nans)
        else:
            imputed = imputed.interpolate(method='linear')
        
        # Fallback till forward/backward fill för edge-värden
        if imputed.isna().any():
            logger.info("Linear interpolation kunde inte hantera alla NaN-värden, använder fill fallback")
            imputed = imputed.ffill().bfill()
        
        return imputed


class MeanImputationStrategy(ImputationStrategy):
    """Mean imputation strategi."""
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med medelvärde."""
        imputed = series.copy()
        mean_value = series.mean()
        
        if pd.isna(mean_value):
            logger.warning("Kan inte beräkna medelvärde för serie med endast NaN-värden")
            return imputed
        
        # Ersätt alla NaN med medelvärdet
        imputed = imputed.fillna(mean_value)
        
        return imputed


class MedianImputationStrategy(ImputationStrategy):
    """Median imputation strategi."""
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med medianvärde."""
        imputed = series.copy()
        median_value = series.median()
        
        if pd.isna(median_value):
            logger.warning("Kan inte beräkna median för serie med endast NaN-värden")
            return imputed
        
        # Ersätt alla NaN med medianvärdet
        imputed = imputed.fillna(median_value)
        
        return imputed


class ZeroImputationStrategy(ImputationStrategy):
    """Zero imputation strategi."""
    
    def impute(self, series: pd.Series, max_consecutive_nans: Optional[int] = None) -> pd.Series:
        """Imputera med nollor."""
        imputed = series.copy()
        
        # Ersätt alla NaN med 0
        imputed = imputed.fillna(0.0)
        
        return imputed 