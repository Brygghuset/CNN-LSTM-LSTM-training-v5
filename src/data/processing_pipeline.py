"""
Data Processing Pipeline med Strategy Pattern.

Denna modul implementerar en flexibel pipeline för databehandling med
separerade ansvarsområden och utbytbara strategier.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status för processingsteg."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Resultat från ett processingsteg."""
    data: pd.DataFrame
    status: ProcessingStatus
    metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    
    def is_successful(self) -> bool:
        """Kontrollera om processingen var framgångsrik."""
        return self.status == ProcessingStatus.COMPLETED
    
    def has_warnings(self) -> bool:
        """Kontrollera om det finns varningar."""
        return len(self.warnings) > 0
    
    def has_errors(self) -> bool:
        """Kontrollera om det finns fel."""
        return len(self.errors) > 0


@dataclass
class PipelineConfig:
    """Konfiguration för processing pipeline."""
    interpolation_method: str = 'linear'
    validate_physiological: bool = True
    target_frequency: float = 1.0
    tolerance: float = 0.001
    max_jump_threshold: float = 0.5
    enable_quality_assurance: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertera till dictionary."""
        return {
            'interpolation_method': self.interpolation_method,
            'validate_physiological': self.validate_physiological,
            'target_frequency': self.target_frequency,
            'tolerance': self.tolerance,
            'max_jump_threshold': self.max_jump_threshold,
            'enable_quality_assurance': self.enable_quality_assurance
        }


class ProcessingStep(ABC):
    """Abstract base class för alla processingsteg."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, config: PipelineConfig) -> ProcessingResult:
        """
        Utför processingsteg.
        
        Args:
            data: Input data
            config: Pipeline konfiguration
            
        Returns:
            ProcessingResult med resultat och metadata
        """
        pass
    
    def _create_result(self, data: pd.DataFrame, status: ProcessingStatus,
                      metadata: Dict[str, Any] = None, warnings: List[str] = None,
                      errors: List[str] = None) -> ProcessingResult:
        """Helper för att skapa ProcessingResult."""
        return ProcessingResult(
            data=data,
            status=status,
            metadata=metadata or {},
            warnings=warnings or [],
            errors=errors or []
        )


class ValidationStep(ProcessingStep):
    """Validerar input data och kontrollerar om processing behövs."""
    
    def __init__(self):
        super().__init__("DataValidation")
    
    def execute(self, data: pd.DataFrame, config: PipelineConfig) -> ProcessingResult:
        """Validera input data."""
        self.logger.info(f"Executing {self.name} step")
        
        warnings = []
        errors = []
        metadata = {
            'original_shape': data.shape,
            'columns': list(data.columns),
            'needs_processing': False
        }
        
        try:
            # Kontrollera att data inte är tom
            if data.empty:
                errors.append("Input data är tom")
                return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)
            
            # Kontrollera att Time-kolumn finns
            if 'Time' not in data.columns:
                warnings.append("Ingen Time-kolumn hittad")
                metadata['has_time_column'] = False
            else:
                metadata['has_time_column'] = True
                
                # Kontrollera temporal ordning
                if len(data) > 1:
                    time_diffs = np.diff(data['Time'].values)
                    if np.any(time_diffs <= 0):
                        warnings.append("Temporal ordning är inte monoton")
                        metadata['temporal_order_issues'] = True
                    else:
                        metadata['temporal_order_issues'] = False
                
                # Kontrollera samplingsfrekvens
                if len(data) > 1:
                    avg_interval = np.mean(time_diffs)
                    actual_frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0
                    frequency_deviation = abs(actual_frequency - config.target_frequency)
                    
                    metadata['actual_frequency'] = actual_frequency
                    metadata['target_frequency'] = config.target_frequency
                    metadata['frequency_deviation'] = frequency_deviation
                    
                    if frequency_deviation > config.tolerance:
                        metadata['needs_processing'] = True
                        self.logger.info(f"Synkronisering behövs: {actual_frequency:.3f} Hz → {config.target_frequency} Hz")
                    else:
                        self.logger.info(f"Data har redan {config.target_frequency} Hz, minimal processing behövs")
            
            # Kontrollera saknade värden
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            
            metadata['missing_values'] = {
                'total': int(total_missing),
                'by_column': missing_counts.to_dict()
            }
            
            if total_missing > 0:
                warnings.append(f"Hittade {total_missing} saknade värden")
                metadata['needs_processing'] = True
            
            # Kontrollera outliers (enkel kontroll)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_columns:
                if col != 'Time':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_info[col] = int(outliers)
                    
                    if outliers > 0:
                        metadata['needs_processing'] = True
            
            metadata['outliers'] = outlier_info
            total_outliers = sum(outlier_info.values())
            if total_outliers > 0:
                warnings.append(f"Hittade {total_outliers} potentiella outliers")
            
            self.logger.info(f"Validation slutförd: {len(warnings)} varningar, {len(errors)} fel")
            
            status = ProcessingStatus.COMPLETED if len(errors) == 0 else ProcessingStatus.FAILED
            return self._create_result(data, status, metadata, warnings, errors)
            
        except Exception as e:
            self.logger.error(f"Validation misslyckades: {e}")
            errors.append(f"Validation error: {str(e)}")
            return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)


class QualityAssuranceStep(ProcessingStep):
    """Kvalitetssäkring av processad data."""
    
    def __init__(self):
        super().__init__("QualityAssurance")
    
    def execute(self, data: pd.DataFrame, config: PipelineConfig) -> ProcessingResult:
        """Utför kvalitetssäkring."""
        if not config.enable_quality_assurance:
            self.logger.info("Quality assurance disabled, skipping")
            return self._create_result(data, ProcessingStatus.SKIPPED)
        
        self.logger.info(f"Executing {self.name} step")
        
        warnings = []
        errors = []
        metadata = {
            'final_shape': data.shape,
            'quality_checks': {}
        }
        
        try:
            # Kontrollera slutlig datakvalitet
            if data.empty:
                errors.append("Processad data är tom")
                return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)
            
            # Kontrollera kontinuitet
            if 'Time' in data.columns and len(data) > 1:
                time_diffs = np.diff(data['Time'].values)
                expected_interval = 1.0 / config.target_frequency
                
                # Kontrollera temporal kontinuitet
                large_gaps = np.abs(time_diffs - expected_interval) > (expected_interval * 0.1)
                gap_count = np.sum(large_gaps)
                
                metadata['quality_checks']['temporal_continuity'] = {
                    'large_gaps': int(gap_count),
                    'expected_interval': expected_interval,
                    'actual_intervals': {
                        'mean': float(np.mean(time_diffs)),
                        'std': float(np.std(time_diffs))
                    }
                }
                
                if gap_count > 0:
                    warnings.append(f"Hittade {gap_count} stora temporal gaps")
            
            # Kontrollera saknade värden efter processing
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            
            metadata['quality_checks']['missing_values'] = {
                'total': int(total_missing),
                'by_column': missing_counts.to_dict()
            }
            
            if total_missing > 0:
                warnings.append(f"Kvarvarande saknade värden: {total_missing}")
            
            # Kontrollera datatyper
            dtype_info = {}
            for col in data.columns:
                dtype_info[col] = str(data[col].dtype)
            
            metadata['quality_checks']['data_types'] = dtype_info
            
            # Kontrollera värdeområden för numeriska kolumner
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            value_ranges = {}
            
            for col in numeric_columns:
                if col != 'Time':
                    col_min = data[col].min()
                    col_max = data[col].max()
                    col_mean = data[col].mean()
                    col_std = data[col].std()
                    
                    value_ranges[col] = {
                        'min': float(col_min) if pd.notna(col_min) else None,
                        'max': float(col_max) if pd.notna(col_max) else None,
                        'mean': float(col_mean) if pd.notna(col_mean) else None,
                        'std': float(col_std) if pd.notna(col_std) else None
                    }
            
            metadata['quality_checks']['value_ranges'] = value_ranges
            
            self.logger.info(f"Quality assurance slutförd: {len(warnings)} varningar, {len(errors)} fel")
            
            status = ProcessingStatus.COMPLETED if len(errors) == 0 else ProcessingStatus.FAILED
            return self._create_result(data, status, metadata, warnings, errors)
            
        except Exception as e:
            self.logger.error(f"Quality assurance misslyckades: {e}")
            errors.append(f"Quality assurance error: {str(e)}")
            return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)


class InterpolationStep(ProcessingStep):
    """Interpolationssteg med Strategy Pattern."""
    
    def __init__(self):
        super().__init__("Interpolation")
    
    def execute(self, data: pd.DataFrame, config: PipelineConfig) -> ProcessingResult:
        """Utför interpolation med vald strategi."""
        self.logger.info(f"Executing {self.name} step with method: {config.interpolation_method}")
        
        warnings = []
        errors = []
        metadata = {
            'interpolation_method': config.interpolation_method,
            'columns_processed': [],
            'total_interpolated_points': 0
        }
        
        try:
            # Importera strategier
            from data.interpolation_strategies import InterpolationStrategyFactory, InterpolationContext
            
            # Skapa strategi
            strategy = InterpolationStrategyFactory.create_strategy(config.interpolation_method)
            
            # Kontrollera om Time-kolumn finns
            if 'Time' not in data.columns:
                warnings.append("Ingen Time-kolumn hittad för interpolation")
                return self._create_result(data, ProcessingStatus.SKIPPED, metadata, warnings, errors)
            
            # Skapa ny tidsaxel
            time_min = data['Time'].min()
            time_max = data['Time'].max()
            new_timestamps = np.arange(time_min, time_max + 1e-6, 1/config.target_frequency)
            
            # Skapa ny DataFrame med interpolerad data
            result_data = pd.DataFrame({'Time': new_timestamps})
            
            # Interpolera varje kolumn (utom Time)
            for column in data.columns:
                if column == 'Time':
                    continue
                
                # Skapa interpolationskontext
                context = InterpolationContext(
                    original_times=data['Time'].values,
                    original_values=data[column].values,
                    new_timestamps=new_timestamps,
                    column_name=column,
                    validate_physiological=config.validate_physiological
                )
                
                # Utför interpolation
                interpolation_result = strategy.interpolate(context)
                
                # Lägg till resultat
                result_data[column] = interpolation_result.values
                
                # Samla metadata
                metadata['columns_processed'].append({
                    'column': column,
                    'success': interpolation_result.success,
                    'method_used': interpolation_result.method_used,
                    'validation_warnings': interpolation_result.validation_warnings,
                    'metadata': interpolation_result.metadata
                })
                
                metadata['total_interpolated_points'] += interpolation_result.metadata.get('valid_interpolated_points', 0)
                
                if not interpolation_result.success:
                    warnings.append(f"Interpolation misslyckades för kolumn {column}")
                
                if interpolation_result.validation_warnings > 0:
                    warnings.append(f"Fysiologiska valideringsvarningar för {column}: {interpolation_result.validation_warnings}")
            
            self.logger.info(f"Interpolation slutförd: {len(metadata['columns_processed'])} kolumner processade")
            
            status = ProcessingStatus.COMPLETED if len(errors) == 0 else ProcessingStatus.FAILED
            return self._create_result(result_data, status, metadata, warnings, errors)
            
        except Exception as e:
            self.logger.error(f"Interpolation misslyckades: {e}")
            errors.append(f"Interpolation error: {str(e)}")
            return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)


class SynchronizationStep(ProcessingStep):
    """Temporal synkroniseringssteg."""
    
    def __init__(self):
        super().__init__("TemporalSynchronization")
    
    def execute(self, data: pd.DataFrame, config: PipelineConfig) -> ProcessingResult:
        """Utför temporal synkronisering."""
        self.logger.info(f"Executing {self.name} step")
        
        warnings = []
        errors = []
        metadata = {
            'target_frequency': config.target_frequency,
            'tolerance': config.tolerance,
            'synchronization_needed': False
        }
        
        try:
            # Kontrollera om synkronisering behövs
            if 'Time' not in data.columns:
                warnings.append("Ingen Time-kolumn hittad för synkronisering")
                return self._create_result(data, ProcessingStatus.SKIPPED, metadata, warnings, errors)
            
            if len(data) < 2:
                warnings.append("Inte tillräckligt med data för synkronisering")
                return self._create_result(data, ProcessingStatus.SKIPPED, metadata, warnings, errors)
            
            # Beräkna aktuell samplingsfrekvens
            time_diffs = np.diff(data['Time'].values)
            current_frequency = 1.0 / np.mean(time_diffs)
            frequency_deviation = abs(current_frequency - config.target_frequency)
            
            metadata['current_frequency'] = current_frequency
            metadata['frequency_deviation'] = frequency_deviation
            
            # Kontrollera om synkronisering behövs
            if frequency_deviation <= config.tolerance:
                self.logger.info(f"Data har redan {config.target_frequency} Hz, ingen synkronisering behövs")
                metadata['synchronization_needed'] = False
                return self._create_result(data, ProcessingStatus.SKIPPED, metadata, warnings, errors)
            
            metadata['synchronization_needed'] = True
            self.logger.info(f"Synkronisering behövs: {current_frequency:.3f} Hz → {config.target_frequency} Hz")
            
            # Kontrollera temporal ordning
            if np.any(time_diffs <= 0):
                warnings.append("Temporal ordning är inte monoton, sorterar data")
                data = data.sort_values('Time').reset_index(drop=True)
                metadata['sorted_data'] = True
            else:
                metadata['sorted_data'] = False
            
            # Utför interpolation för synkronisering
            interpolation_step = InterpolationStep()
            interpolation_result = interpolation_step.execute(data, config)
            
            if not interpolation_result.is_successful():
                errors.extend(interpolation_result.errors)
                warnings.extend(interpolation_result.warnings)
                return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)
            
            # Verifiera slutresultat
            synchronized_data = interpolation_result.data
            if len(synchronized_data) > 1:
                final_time_diffs = np.diff(synchronized_data['Time'].values)
                final_frequency = 1.0 / np.mean(final_time_diffs)
                metadata['final_frequency'] = final_frequency
                metadata['final_deviation'] = abs(final_frequency - config.target_frequency)
                
                if metadata['final_deviation'] <= config.tolerance:
                    self.logger.info(f"Synkronisering slutförd: {final_frequency:.6f} Hz")
                else:
                    warnings.append(f"Synkronisering inte perfekt: {final_frequency:.6f} Hz (mål: {config.target_frequency} Hz)")
            
            # Sammanfoga metadata från interpolation
            metadata['interpolation_metadata'] = interpolation_result.metadata
            warnings.extend(interpolation_result.warnings)
            
            self.logger.info(f"Temporal synkronisering slutförd")
            
            status = ProcessingStatus.COMPLETED if len(errors) == 0 else ProcessingStatus.FAILED
            return self._create_result(synchronized_data, status, metadata, warnings, errors)
            
        except Exception as e:
            self.logger.error(f"Temporal synkronisering misslyckades: {e}")
            errors.append(f"Synchronization error: {str(e)}")
            return self._create_result(data, ProcessingStatus.FAILED, metadata, warnings, errors)


class DataProcessingPipeline:
    """Huvudklass för databehandlingspipeline med Strategy Pattern."""
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialisera pipeline.
        
        Args:
            config: Pipeline konfiguration. Om None skapas default config.
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialisera alla steg
        self.steps = [
            ValidationStep(),
            SynchronizationStep(),
            QualityAssuranceStep()
        ]
    
    def process(self, data: pd.DataFrame) -> ProcessingResult:
        """
        Kör hela pipeline på data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ProcessingResult med slutresultat
        """
        self.logger.info(f"Starting data processing pipeline with {len(self.steps)} steps")
        
        current_data = data.copy()
        all_warnings = []
        all_errors = []
        all_metadata = {
            'pipeline_config': self.config.to_dict(),
            'steps': []
        }
        
        # Kör alla steg
        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                result = step.execute(current_data, self.config)
                
                # Samla metadata
                step_metadata = {
                    'step_name': step.name,
                    'status': result.status.value,
                    'warnings_count': len(result.warnings),
                    'errors_count': len(result.errors),
                    'metadata': result.metadata
                }
                all_metadata['steps'].append(step_metadata)
                
                # Samla varningar och fel
                all_warnings.extend(result.warnings)
                all_errors.extend(result.errors)
                
                # Kontrollera status
                if result.status == ProcessingStatus.FAILED:
                    self.logger.error(f"Step {step.name} failed, stopping pipeline")
                    return ProcessingResult(
                        data=current_data,
                        status=ProcessingStatus.FAILED,
                        metadata=all_metadata,
                        warnings=all_warnings,
                        errors=all_errors
                    )
                elif result.status == ProcessingStatus.SKIPPED:
                    self.logger.info(f"Step {step.name} skipped")
                    continue
                else:
                    # Uppdatera data för nästa steg
                    current_data = result.data
                    
            except Exception as e:
                self.logger.error(f"Step {step.name} raised exception: {e}")
                all_errors.append(f"Step {step.name} exception: {str(e)}")
                return ProcessingResult(
                    data=current_data,
                    status=ProcessingStatus.FAILED,
                    metadata=all_metadata,
                    warnings=all_warnings,
                    errors=all_errors
                )
        
        # Pipeline slutförd framgångsrikt
        all_metadata['final_shape'] = current_data.shape
        
        self.logger.info(f"Pipeline slutförd: {len(all_warnings)} varningar, {len(all_errors)} fel")
        
        final_status = ProcessingStatus.COMPLETED if len(all_errors) == 0 else ProcessingStatus.FAILED
        
        return ProcessingResult(
            data=current_data,
            status=final_status,
            metadata=all_metadata,
            warnings=all_warnings,
            errors=all_errors
        )
    
    def add_step(self, step: ProcessingStep, position: Optional[int] = None):
        """
        Lägg till ett steg i pipeline.
        
        Args:
            step: ProcessingStep att lägga till
            position: Position att infoga på (None = sist)
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        self.logger.info(f"Added step {step.name} at position {position or len(self.steps)-1}")
    
    def remove_step(self, step_name: str) -> bool:
        """
        Ta bort ett steg från pipeline.
        
        Args:
            step_name: Namn på steg att ta bort
            
        Returns:
            True om steget togs bort, False om det inte hittades
        """
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                removed_step = self.steps.pop(i)
                self.logger.info(f"Removed step {removed_step.name}")
                return True
        
        self.logger.warning(f"Step {step_name} not found for removal")
        return False
    
    def get_step_names(self) -> List[str]:
        """Hämta namn på alla steg i pipeline."""
        return [step.name for step in self.steps] 