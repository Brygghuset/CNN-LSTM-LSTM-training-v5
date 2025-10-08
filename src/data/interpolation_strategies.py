"""
Interpolation Strategies för databehandling enligt Strategy Pattern.

Denna modul implementerar olika interpolationsstrategier som kan användas
utbytbart i databehandlingspipelinen.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy.interpolate import interp1d
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InterpolationContext:
    """Kontext för interpolation med all nödvändig information."""
    original_times: np.ndarray
    original_values: np.ndarray
    new_timestamps: np.ndarray
    column_name: str
    validate_physiological: bool = True
    preserve_critical_events: bool = True


@dataclass
class InterpolationResult:
    """Resultat från interpolation."""
    values: np.ndarray
    success: bool
    method_used: str
    critical_events_preserved: int = 0
    validation_warnings: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InterpolationStrategy(ABC):
    """Abstract base class för interpolationsstrategier."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def interpolate(self, context: InterpolationContext) -> InterpolationResult:
        """
        Utför interpolation enligt strategin.
        
        Args:
            context: InterpolationContext med all nödvändig information
            
        Returns:
            InterpolationResult med resultat och metadata
        """
        pass
    
    def _prepare_data(self, times: np.ndarray, values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Förbered data för interpolation genom att ta bort NaN-värden."""
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 2:
            return None
        return times[valid_mask], values[valid_mask]
    
    def _validate_physiological(self, column: str, values: np.ndarray) -> Tuple[np.ndarray, int]:
        """Validera interpolerade värden mot fysiologiska gränser."""
        try:
            from utils.validators import CentralizedValidator
            validator = CentralizedValidator()
            
            warning_count = 0
            validated_values = values.copy()
            
            for i, value in enumerate(values):
                if not np.isnan(value):
                    result = validator.validate_parameter(column, value)
                    if not result.is_valid:
                        # Ersätt ogiltigt värde med NaN
                        validated_values[i] = np.nan
                        warning_count += 1
            
            return validated_values, warning_count
            
        except ImportError:
            self.logger.warning("CentralizedValidator inte tillgänglig, hoppar över fysiologisk validering")
            return values, 0


class LinearInterpolationStrategy(InterpolationStrategy):
    """Linjär interpolationsstrategi."""
    
    def __init__(self):
        super().__init__("LinearInterpolation")
    
    def interpolate(self, context: InterpolationContext) -> InterpolationResult:
        """Utför linjär interpolation."""
        try:
            # Förbered data
            prepared_data = self._prepare_data(context.original_times, context.original_values)
            if prepared_data is None:
                return InterpolationResult(
                    values=np.full(len(context.new_timestamps), np.nan),
                    success=False,
                    method_used=self.name,
                    metadata={'error': 'Otillräcklig data för interpolation'}
                )
            
            valid_times, valid_values = prepared_data
            
            # Skapa linjär interpolator
            interpolator = interp1d(
                valid_times, valid_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=np.nan
            )
            
            # Interpolera
            interpolated_values = interpolator(context.new_timestamps)
            
            # Fysiologisk validering om begärt
            validation_warnings = 0
            if context.validate_physiological:
                interpolated_values, validation_warnings = self._validate_physiological(
                    context.column_name, interpolated_values
                )
            
            metadata = {
                'original_points': len(valid_values),
                'interpolated_points': len(context.new_timestamps),
                'valid_interpolated_points': np.sum(~np.isnan(interpolated_values))
            }
            
            return InterpolationResult(
                values=interpolated_values,
                success=True,
                method_used=self.name,
                validation_warnings=validation_warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Linjär interpolation misslyckades för {context.column_name}: {e}")
            return InterpolationResult(
                values=np.full(len(context.new_timestamps), np.nan),
                success=False,
                method_used=self.name,
                metadata={'error': str(e)}
            )


class CubicInterpolationStrategy(InterpolationStrategy):
    """Kubisk interpolationsstrategi."""
    
    def __init__(self):
        super().__init__("CubicInterpolation")
    
    def interpolate(self, context: InterpolationContext) -> InterpolationResult:
        """Utför kubisk interpolation."""
        try:
            # Förbered data
            prepared_data = self._prepare_data(context.original_times, context.original_values)
            if prepared_data is None:
                return InterpolationResult(
                    values=np.full(len(context.new_timestamps), np.nan),
                    success=False,
                    method_used=self.name,
                    metadata={'error': 'Otillräcklig data för interpolation'}
                )
            
            valid_times, valid_values = prepared_data
            
            # Kubisk interpolation kräver minst 4 punkter
            if len(valid_values) < 4:
                self.logger.warning(f"Kubisk interpolation kräver minst 4 punkter, har {len(valid_values)}. Faller tillbaka till linjär.")
                # Fallback till linjär
                linear_strategy = LinearInterpolationStrategy()
                result = linear_strategy.interpolate(context)
                result.method_used = f"{self.name}_fallback_to_linear"
                return result
            
            # Skapa kubisk interpolator
            interpolator = interp1d(
                valid_times, valid_values, 
                kind='cubic', 
                bounds_error=False, 
                fill_value=np.nan
            )
            
            # Interpolera
            interpolated_values = interpolator(context.new_timestamps)
            
            # Fysiologisk validering om begärt
            validation_warnings = 0
            if context.validate_physiological:
                interpolated_values, validation_warnings = self._validate_physiological(
                    context.column_name, interpolated_values
                )
            
            metadata = {
                'original_points': len(valid_values),
                'interpolated_points': len(context.new_timestamps),
                'valid_interpolated_points': np.sum(~np.isnan(interpolated_values))
            }
            
            return InterpolationResult(
                values=interpolated_values,
                success=True,
                method_used=self.name,
                validation_warnings=validation_warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Kubisk interpolation misslyckades för {context.column_name}: {e}")
            # Fallback till linjär
            linear_strategy = LinearInterpolationStrategy()
            result = linear_strategy.interpolate(context)
            result.method_used = f"{self.name}_fallback_to_linear"
            result.metadata['fallback_reason'] = str(e)
            return result


class NearestInterpolationStrategy(InterpolationStrategy):
    """Nearest neighbor interpolationsstrategi."""
    
    def __init__(self):
        super().__init__("NearestInterpolation")
    
    def interpolate(self, context: InterpolationContext) -> InterpolationResult:
        """Utför nearest neighbor interpolation."""
        try:
            # Förbered data
            prepared_data = self._prepare_data(context.original_times, context.original_values)
            if prepared_data is None:
                return InterpolationResult(
                    values=np.full(len(context.new_timestamps), np.nan),
                    success=False,
                    method_used=self.name,
                    metadata={'error': 'Otillräcklig data för interpolation'}
                )
            
            valid_times, valid_values = prepared_data
            
            # Skapa nearest neighbor interpolator
            interpolator = interp1d(
                valid_times, valid_values, 
                kind='nearest', 
                bounds_error=False, 
                fill_value=np.nan
            )
            
            # Interpolera
            interpolated_values = interpolator(context.new_timestamps)
            
            # Fysiologisk validering om begärt
            validation_warnings = 0
            if context.validate_physiological:
                interpolated_values, validation_warnings = self._validate_physiological(
                    context.column_name, interpolated_values
                )
            
            metadata = {
                'original_points': len(valid_values),
                'interpolated_points': len(context.new_timestamps),
                'valid_interpolated_points': np.sum(~np.isnan(interpolated_values))
            }
            
            return InterpolationResult(
                values=interpolated_values,
                success=True,
                method_used=self.name,
                validation_warnings=validation_warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Nearest neighbor interpolation misslyckades för {context.column_name}: {e}")
            return InterpolationResult(
                values=np.full(len(context.new_timestamps), np.nan),
                success=False,
                method_used=self.name,
                metadata={'error': str(e)}
            )


class ForwardFillStrategy(InterpolationStrategy):
    """Forward fill strategi (carry forward)."""
    
    def __init__(self):
        super().__init__("ForwardFill")
    
    def interpolate(self, context: InterpolationContext) -> InterpolationResult:
        """Utför forward fill interpolation."""
        try:
            # Förbered data
            prepared_data = self._prepare_data(context.original_times, context.original_values)
            if prepared_data is None:
                return InterpolationResult(
                    values=np.full(len(context.new_timestamps), np.nan),
                    success=False,
                    method_used=self.name,
                    metadata={'error': 'Otillräcklig data för interpolation'}
                )
            
            valid_times, valid_values = prepared_data
            
            # Implementera forward fill
            interpolated_values = np.full(len(context.new_timestamps), np.nan)
            
            for i, target_time in enumerate(context.new_timestamps):
                # Hitta senaste tillgängliga värde före eller vid target_time
                valid_indices = valid_times <= target_time
                if np.any(valid_indices):
                    last_valid_idx = np.where(valid_indices)[0][-1]
                    interpolated_values[i] = valid_values[last_valid_idx]
            
            # Fysiologisk validering om begärt
            validation_warnings = 0
            if context.validate_physiological:
                interpolated_values, validation_warnings = self._validate_physiological(
                    context.column_name, interpolated_values
                )
            
            metadata = {
                'original_points': len(valid_values),
                'interpolated_points': len(context.new_timestamps),
                'valid_interpolated_points': np.sum(~np.isnan(interpolated_values))
            }
            
            return InterpolationResult(
                values=interpolated_values,
                success=True,
                method_used=self.name,
                validation_warnings=validation_warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Forward fill misslyckades för {context.column_name}: {e}")
            return InterpolationResult(
                values=np.full(len(context.new_timestamps), np.nan),
                success=False,
                method_used=self.name,
                metadata={'error': str(e)}
            )


class InterpolationStrategyFactory:
    """Factory för att skapa interpolationsstrategier."""
    
    _strategies = {
        'linear': LinearInterpolationStrategy,
        'cubic': CubicInterpolationStrategy,
        'nearest': NearestInterpolationStrategy,
        'forward_fill': ForwardFillStrategy
    }
    
    @classmethod
    def create_strategy(cls, method: str) -> InterpolationStrategy:
        """
        Skapa interpolationsstrategi baserat på metod.
        
        Args:
            method: Interpolationsmetod ('linear', 'cubic', 'nearest', 'forward_fill')
            
        Returns:
            InterpolationStrategy instans
        """
        if method not in cls._strategies:
            available_methods = list(cls._strategies.keys())
            raise ValueError(f"Okänd interpolationsmetod: {method}. Tillgängliga: {available_methods}")
        
        return cls._strategies[method]()
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Hämta lista över tillgängliga interpolationsmetoder."""
        return list(cls._strategies.keys()) 