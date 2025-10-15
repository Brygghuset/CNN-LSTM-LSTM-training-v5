"""
Temporal synkronisering för VitalDB data.

Hanterar synkronisering av data med oregelbunden samplingsfrekvens till standardiserad 1 Hz
för modell-input med linjär interpolation och säkerhetsvalidering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from dataclasses import dataclass

# Centraliserad validering med loose coupling
# from utils.validators import  # KOMMENTERAD BORT - inte kritisk för AWS körning CentralizedValidator, ValidationCategory
# from interfaces import IValidator  # KOMMENTERAD BORT - inte kritisk för AWS körning

logger = logging.getLogger(__name__)


@dataclass
class InterpolationData:
    """Data container för interpolationsprocess."""
    original_times: np.ndarray
    original_values: np.ndarray
    new_timestamps: np.ndarray
    column_name: str
    method: str
    validate_physiological: bool


@dataclass
class InterpolationResult:
    """Resultat från interpolationsprocess."""
    values: np.ndarray
    success: bool
    critical_events_preserved: int = 0
    validation_warnings: int = 0


class SynchronizationStep(ABC):
    """Abstract base class för synkroniseringssteg."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Processera data i ett synkroniseringssteg."""
        pass


class FrequencyValidator(SynchronizationStep):
    """Validerar och kontrollerar om synkronisering behövs."""
    
    def __init__(self, target_frequency: float = 1.0, tolerance: float = 0.001):
        self.target_frequency = target_frequency
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
    def needs_synchronization(self, df: pd.DataFrame) -> bool:
        """Kontrollera om data behöver synkronisering."""
        if 'Time' not in df.columns or len(df) < 2:
            return False
            
        # Beräkna aktuell samplingsfrekvens
        time_diffs = np.diff(df['Time'].values)
        current_frequency = 1.0 / np.mean(time_diffs)
        
        # Kontrollera om frekvensen avviker från målet
        frequency_deviation = abs(current_frequency - self.target_frequency)
        
        return frequency_deviation > self.tolerance
    
    def process(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Validera och returnera data om synkronisering inte behövs."""
        if 'Time' not in df.columns:
            self.logger.warning("Ingen Time-kolumn hittad, returnerar original data")
            config['skip_sync'] = True
            return df.copy()
        
        if len(df) < 2:
            self.logger.warning("Inte tillräckligt med data för synkronisering")
            config['skip_sync'] = True
            return df.copy()
        
        if not self.needs_synchronization(df):
            self.logger.info(f"Data har redan {self.target_frequency} Hz, ingen synkronisering behövs")
            config['skip_sync'] = True
            return df.copy()
        
        self.logger.info(f"Synkronisering behövs - fortsätter med interpolation")
        config['skip_sync'] = False
        return df  # Fortsätt till nästa steg


class TimeAxisGenerator(SynchronizationStep):
    """Genererar ny tidsaxel med uniform sampling."""
    
    def __init__(self, target_frequency: float = 1.0):
        self.target_frequency = target_frequency
        self.logger = logging.getLogger(__name__)
    
    def process(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Skapa ny DataFrame med uniform tidsaxel."""
        if config.get('skip_sync', False):
            return df
            
        time_min = df['Time'].min()
        time_max = df['Time'].max()
        new_timestamps = np.arange(time_min, time_max + 1e-6, 1/self.target_frequency)
        
        # Skapa ny DataFrame med uniform tidsaxel
        synchronized_df = pd.DataFrame({'Time': new_timestamps})
        
        # Lagra original data i config för nästa steg
        config['original_df'] = df
        config['new_timestamps'] = new_timestamps
        
        self.logger.info(f"Genererad ny tidsaxel: {len(new_timestamps)} sampel")
        
        return synchronized_df


class CriticalEventDetector:
    """Detekterar kritiska medicinska event för preservation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clinical_thresholds = self._initialize_clinical_thresholds()
    
    def _initialize_clinical_thresholds(self) -> Dict[str, Dict]:
        """Initialisera kliniska trösklar för event-detection."""
        return {
            'Orchestra/PPF20_RATE': {'peak': 5.0, 'valley': 0.0},  # Propofol bolus
            'Solar8000/HR': {'peak': 100, 'valley': 50},  # Hjärtfrekvens
            'Solar8000/NIBP_SBP': {'peak': 160, 'valley': 90},  # Systoliskt blodtryck
            'Solar8000/PLETH_SPO2': {'peak': 100, 'valley': 95},  # Syresättning
        }
    
    def detect_critical_events(self, times: np.ndarray, values: np.ndarray, column: str) -> List[Dict]:
        """Detektera kritiska medicinska event."""
        events = []
        
        if len(values) < 3:
            return events
        
        # Detektera peaks och valleys
        events.extend(self._detect_peaks(times, values, column))
        events.extend(self._detect_valleys(times, values, column))
        
        return events
    
    def _detect_peaks(self, times: np.ndarray, values: np.ndarray, column: str) -> List[Dict]:
        """Detektera peaks (lokala maxima)."""
        events = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                if self._is_clinically_significant(values[i], column, 'peak'):
                    events.append({
                        'type': 'peak',
                        'time': times[i],
                        'value': values[i],
                        'index': i
                    })
        
        return events
    
    def _detect_valleys(self, times: np.ndarray, values: np.ndarray, column: str) -> List[Dict]:
        """Detektera valleys (lokala minima)."""
        events = []
        
        for i in range(1, len(values) - 1):
            if values[i] < values[i-1] and values[i] < values[i+1]:
                if self._is_clinically_significant(values[i], column, 'valley'):
                    events.append({
                        'type': 'valley',
                        'time': times[i],
                        'value': values[i],
                        'index': i
                    })
        
        return events
    
    def _is_clinically_significant(self, value: float, column: str, event_type: str) -> bool:
        """Kontrollera om ett event är kliniskt signifikant."""
        if column not in self.clinical_thresholds:
            return True  # Om ingen tröskel definierad, anta signifikant
        
        threshold = self.clinical_thresholds[column][event_type]
        
        if event_type == 'peak':
            return value >= threshold
        else:  # valley
            return value <= threshold


class InterpolationEngine:
    """Ren interpolationsmotor utan sidoeffekter."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_interpolator(self, times: np.ndarray, values: np.ndarray, method: str):
        """Skapa interpolator baserat på metod."""
        if method == 'linear':
            return interp1d(times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            return interp1d(times, values, kind='cubic', bounds_error=False, fill_value='extrapolate')
        elif method == 'nearest':
            return interp1d(times, values, kind='nearest', bounds_error=False, fill_value='extrapolate')
        else:
            self.logger.warning(f"Okänd interpolationsmetod: {method}, använder linear")
            return interp1d(times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    def interpolate_values(self, interpolator, new_timestamps: np.ndarray) -> np.ndarray:
        """Utför själva interpolationen."""
        try:
            return interpolator(new_timestamps)
        except Exception as e:
            self.logger.error(f"Interpolation misslyckades: {e}")
            return np.full(len(new_timestamps), np.nan)


class EventPreserver:
    """Bevarar kritiska medicinska event i interpolerad data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preserve_events(self, new_times: np.ndarray, interpolated_values: np.ndarray, 
                       critical_events: List[Dict]) -> Tuple[np.ndarray, int]:
        """
        Bevara kritiska event i interpolerad data.
        
        Returns:
            Tuple av (preserved_values, antal_bevarade_events)
        """
        if not critical_events:
            return interpolated_values, 0
        
        preserved_values = interpolated_values.copy()
        preserved_count = 0
        
        for event in critical_events:
            # Hitta närmaste tidpunkt i nya tidsaxeln
            time_distances = np.abs(new_times - event['time'])
            closest_idx = np.argmin(time_distances)
            
            # Säkerställ att event-värdet bevaras
            preserved_values[closest_idx] = event['value']
            preserved_count += 1
        
        self.logger.debug(f"Bevarade {preserved_count} kritiska event")
        return preserved_values, preserved_count


class PhysiologicalValidator:
    """Validerar interpolerade värden mot fysiologiska gränser."""
    
    def __init__(self, validator=None):
        # Använd dependency injection för validator
        if validator is None:
            self.validator = CentralizedValidator()
        else:
            self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    def validate_values(self, column: str, interpolated_values: np.ndarray, 
                       original_values: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Validera interpolerade värden mot fysiologiska gränser.
            
        Returns:
            Tuple av (validated_values, antal_warnings)
        """
        # Använd centraliserad validator
        validation_results = []
        warning_count = 0
        
        for value in interpolated_values:
            if not np.isnan(value):
                result = self.validator.validate_parameter(column, value)
                validation_results.append(result.is_valid)
                if not result.is_valid:
                    warning_count += 1
            else:
                validation_results.append(True)  # NaN är OK
        
        # Om validering misslyckas, använd original värden som referens
        if not all(validation_results):
            self.logger.warning(f"Validering misslyckades för {column}: {warning_count} ogiltiga värden")
            # Begränsa till ursprungliga min/max värden
            original_min = np.nanmin(original_values)
            original_max = np.nanmax(original_values)
            validated_values = np.clip(interpolated_values, original_min, original_max)
        else:
            validated_values = interpolated_values
        
        return validated_values, warning_count


class ColumnInterpolator:
    """Orchestrator för att interpolera en enskild kolumn."""
    
    def __init__(self):
        self.engine = InterpolationEngine()
        self.event_detector = CriticalEventDetector()
        self.event_preserver = EventPreserver()
        self.validator = PhysiologicalValidator()
        self.logger = logging.getLogger(__name__)
    
    def interpolate_column(self, data: InterpolationData) -> InterpolationResult:
        """
        Interpolera en enskild kolumn med alla steg.
        
        Returns:
            InterpolationResult med resultatet och metadata
        """
        try:
            # Steg 1: Förbered data
            valid_data = self._prepare_data(data.original_times, data.original_values)
            if valid_data is None:
                return InterpolationResult(
                    values=np.full(len(data.new_timestamps), np.nan),
                    success=False
                )
            
            valid_times, valid_values = valid_data
            
            # Steg 2: Detektera kritiska event
            critical_events = self.event_detector.detect_critical_events(
                valid_times, valid_values, data.column_name
            )
            
            # Steg 3: Skapa interpolator och interpolera
            interpolator = self.engine.create_interpolator(
                valid_times, valid_values, data.method
            )
            interpolated_values = self.engine.interpolate_values(
                interpolator, data.new_timestamps
            )
            
            # Steg 4: Bevara kritiska event
            preserved_values, events_preserved = self.event_preserver.preserve_events(
                data.new_timestamps, interpolated_values, critical_events
            )
            
            # Steg 5: Validera fysiologiskt om begärt
            warnings = 0
            if data.validate_physiological:
                preserved_values, warnings = self.validator.validate_values(
                    data.column_name, preserved_values, data.original_values
                )
            
            return InterpolationResult(
                values=preserved_values,
                success=True,
                critical_events_preserved=events_preserved,
                validation_warnings=warnings
            )
            
        except Exception as e:
            self.logger.warning(f"Interpolation misslyckades för {data.column_name}: {e}")
            return InterpolationResult(
                values=np.full(len(data.new_timestamps), np.nan),
                success=False
            )
    
    def _prepare_data(self, times: np.ndarray, values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Förbered data för interpolation genom att ta bort NaN-värden."""
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 2:
            return None
        
        return times[valid_mask], values[valid_mask]


class DataInterpolator(SynchronizationStep):
    """Hanterar interpolation av alla kolumner med refaktorerad struktur."""
    
    def __init__(self, validator=None):
        self.logger = logging.getLogger(__name__)
        self.column_interpolator = ColumnInterpolator()
        # Använd dependency injection för validator
        if validator is not None:
            self.column_interpolator.validator = PhysiologicalValidator(validator)
    
    def process(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Interpolera alla kolumner till ny tidsaxel."""
        if config.get('skip_sync', False):
            return df
            
        original_df = config['original_df']
        new_timestamps = config['new_timestamps']
        interpolation_method = config.get('interpolation_method', 'linear')
        validate_physiological = config.get('validate_physiological', True)
        
        # Interpolera varje kolumn
        total_events_preserved = 0
        total_warnings = 0
        
        for column in original_df.columns:
            if column == 'Time':
                continue
            
            # Skapa interpolationsdata
            interpolation_data = InterpolationData(
                original_times=original_df['Time'].values,
                original_values=original_df[column].values,
                new_timestamps=new_timestamps,
                column_name=column,
                method=interpolation_method,
                validate_physiological=validate_physiological
            )
            
            # Interpolera kolumnen
            result = self.column_interpolator.interpolate_column(interpolation_data)
            
            # Lägg till resultat till DataFrame
            df[column] = result.values
            
            # Samla statistik
            if result.success:
                total_events_preserved += result.critical_events_preserved
                total_warnings += result.validation_warnings
        
        self.logger.info(f"Interpolation slutförd: {total_events_preserved} event bevarade, "
                        f"{total_warnings} valideringsvarningar")
        
        return df


class TemporalSynchronizer:
    """
    Refaktorerad temporal synchronizer med separerade ansvarsområden.
    
    Använder Chain of Responsibility pattern för synkroniseringsprocessen.
    """
    
    def __init__(self, target_frequency: float = 1.0, tolerance: float = 0.001):
        self.target_frequency = target_frequency
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
        # Initialisera synkroniseringssteg
        self.steps = [
            FrequencyValidator(target_frequency, tolerance),
            TimeAxisGenerator(target_frequency),
            DataInterpolator()
        ]
    
    def synchronize_data(self, df: pd.DataFrame, 
                        interpolation_method: str = 'linear',
                        validate_physiological: bool = True) -> pd.DataFrame:
        """
        Synkronisera data till målfrekvens med refaktorerad pipeline.
        
        Args:
            df: DataFrame med Time-kolumn och data
            interpolation_method: Interpolationsmetod ('linear', 'cubic', 'nearest')
            validate_physiological: Om True valideras interpolerade värden
            
        Returns:
            Synkroniserad DataFrame med bevarade kritiska event
        """
        self.logger.info(f"Börjar temporal synkronisering för {len(df)} rader")
        
        # Konfiguration för alla steg
        config = {
            'interpolation_method': interpolation_method,
            'validate_physiological': validate_physiological
        }
        
        current_df = df.copy()
        
        # Kör alla synkroniseringssteg
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            self.logger.debug(f"Kör steg {i+1}/{len(self.steps)}: {step_name}")
            
            result_df = step.process(current_df, config)
            
            # Om första steget indikerar att synkronisering ska skippas
            if i == 0 and config.get('skip_sync', False):
                self.logger.info("Ingen synkronisering behövdes")
                return result_df
            
            current_df = result_df
        
        # Verifiera slutresultat
        actual_frequency = self._verify_sampling_frequency(current_df)
        self.logger.info(f"Synkronisering slutförd: {actual_frequency:.6f} Hz")
        
        return current_df
    
    def _verify_sampling_frequency(self, df: pd.DataFrame) -> float:
        """Verifiera den faktiska samplingsfrekvensen."""
        if 'Time' not in df.columns or len(df) < 2:
            return 0.0
        
        time_diffs = np.diff(df['Time'].values)
        avg_interval = np.mean(time_diffs)
        actual_frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        return actual_frequency
    
    # Befintliga metoder behålls för bakåtkompatibilitet
    def needs_synchronization(self, df: pd.DataFrame) -> bool:
        """Bakåtkompatibel metod."""
        validator = FrequencyValidator(self.target_frequency, self.tolerance)
        return validator.needs_synchronization(df)
    
    def check_temporal_continuity(self, df: pd.DataFrame, 
                                max_jump_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Kontrollera temporal kontinuitet för att undvika interpolation-artefakter.
        
        Args:
            df: DataFrame att kontrollera
            max_jump_threshold: Maximal tillåten förändring mellan sampel (som fraktion av medelvärdet)
            
        Returns:
            Dictionary med kontinuitetsresultat
        """
        continuity_results = {}
        
        for column in df.columns:
            if column == 'Time':
                continue
                
            data = df[column].dropna()
            if len(data) < 2:
                continuity_results[column] = {
                    'continuous': True,
                    'large_jumps': 0,
                    'max_jump': 0.0
                }
                continue
            
            # Beräkna förändringar mellan sampel
            diffs = np.abs(np.diff(data))
            mean_value = data.mean()
            max_allowed_jump = mean_value * max_jump_threshold
            
            # Hitta stora hopp
            large_jumps = diffs > max_allowed_jump
            large_jump_count = np.sum(large_jumps)
            
            continuity_results[column] = {
                'continuous': large_jump_count == 0,
                'large_jumps': large_jump_count,
                'max_jump': np.max(diffs) if len(diffs) > 0 else 0.0,
                'max_allowed_jump': max_allowed_jump
            }
            
            if large_jump_count > 0:
                logger.warning(f"Temporal diskontinuitet hittad i {column}: "
                             f"{large_jump_count} stora hopp (max: {np.max(diffs):.2f})")
        
        return continuity_results


# Bakåtkompatibel funktion för befintlig kod
def synchronize_data(df: pd.DataFrame, target_frequency: float = 1.0, 
                    interpolation_method: str = 'linear',
                    validate_physiological: bool = True) -> pd.DataFrame:
    """
    Bakåtkompatibel wrapper för refaktorerad synchronizer.
    
    DEPRECATED: Använd TemporalSynchronizer direkt för nya implementationer.
    """
    synchronizer = TemporalSynchronizer(target_frequency)
    return synchronizer.synchronize_data(df, interpolation_method, validate_physiological) 