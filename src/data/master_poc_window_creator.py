#!/usr/bin/env python3
"""
Master POC Window Creation Module
Implementerar sliding window creation enligt Master POC specifikation
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WindowConfig:
    """Konfiguration för window creation."""
    window_size_seconds: int = 300  # 5 minuter
    step_size_seconds: int = 30     # 10% overlap
    timeseries_features: int = 16  # Antal timeseries features
    static_features: int = 6       # Antal static features
    sampling_rate_hz: float = 1.0  # 1 Hz sampling rate

class MasterPOCWindowCreator:
    """
    Master POC Window Creator enligt specifikation.
    
    Window Size: 300 sekunder (5 min)
    Step Size: 30 sekunder (10% overlap)
    Window Shape: [300, 16] för timeseries
    """
    
    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()
        logger.info(f"MasterPOCWindowCreator initialiserad:")
        logger.info(f"   Window Size: {self.config.window_size_seconds}s")
        logger.info(f"   Step Size: {self.config.step_size_seconds}s")
        logger.info(f"   Timeseries Features: {self.config.timeseries_features}")
        logger.info(f"   Static Features: {self.config.static_features}")
    
    def create_sliding_windows(self, 
                             timeseries_data: np.ndarray, 
                             static_data: np.ndarray,
                             timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Skapa sliding windows från timeseries och static data.
        
        Args:
            timeseries_data: Shape [time_steps, 16] - timeseries features
            static_data: Shape [6] - static features
            timestamps: Optional timestamps för varje time step
            
        Returns:
            Tuple of:
            - windows: Shape [n_windows, 300, 16] - sliding windows
            - static_windows: Shape [n_windows, 6] - repeated static features
            - metadata: List of window metadata
        """
        if timeseries_data.shape[1] != self.config.timeseries_features:
            raise ValueError(f"Timeseries data måste ha {self.config.timeseries_features} features, fick {timeseries_data.shape[1]}")
        
        if static_data.shape[0] != self.config.static_features:
            raise ValueError(f"Static data måste ha {self.config.static_features} features, fick {static_data.shape[0]}")
        
        time_steps = timeseries_data.shape[0]
        window_size_steps = self.config.window_size_seconds  # 1 Hz sampling rate
        step_size_steps = self.config.step_size_seconds
        
        # Beräkna antal windows
        if time_steps < window_size_steps:
            logger.warning(f"Otillräcklig data: {time_steps} steg < {window_size_steps} steg (300s)")
            return np.array([]).reshape(0, window_size_steps, self.config.timeseries_features), \
                   np.array([]).reshape(0, self.config.static_features), \
                   []
        
        # Beräkna start indices för windows
        start_indices = list(range(0, time_steps - window_size_steps + 1, step_size_steps))
        
        if not start_indices:
            logger.warning(f"Ingen window kan skapas från {time_steps} steg med window size {window_size_steps}")
            return np.array([]).reshape(0, window_size_steps, self.config.timeseries_features), \
                   np.array([]).reshape(0, self.config.static_features), \
                   []
        
        # Skapa windows
        windows = []
        static_windows = []
        metadata = []
        
        for i, start_idx in enumerate(start_indices):
            end_idx = start_idx + window_size_steps
            
            # Extrahera window
            window = timeseries_data[start_idx:end_idx]
            windows.append(window)
            
            # Replikera static features för varje window
            static_windows.append(static_data.copy())
            
            # Skapa metadata
            window_metadata = {
                'window_index': i,
                'start_time_step': start_idx,
                'end_time_step': end_idx,
                'start_timestamp': timestamps[start_idx] if timestamps is not None else start_idx,
                'end_timestamp': timestamps[end_idx-1] if timestamps is not None else end_idx-1,
                'duration_seconds': window_size_steps,
                'step_size_seconds': step_size_steps
            }
            metadata.append(window_metadata)
        
        # Konvertera till numpy arrays
        windows_array = np.array(windows)
        static_windows_array = np.array(static_windows)
        
        logger.info(f"Skapade {len(windows)} windows från {time_steps} time steps")
        logger.info(f"Window shape: {windows_array.shape}")
        logger.info(f"Static windows shape: {static_windows_array.shape}")
        
        return windows_array, static_windows_array, metadata
    
    def calculate_expected_window_count(self, time_steps: int) -> int:
        """
        Beräkna förväntat antal windows från antal time steps.
        
        Args:
            time_steps: Antal time steps i data
            
        Returns:
            Antal förväntade windows
        """
        window_size_steps = self.config.window_size_seconds
        step_size_steps = self.config.step_size_seconds
        
        if time_steps < window_size_steps:
            return 0
        
        # Beräkna antal windows med sliding window
        window_count = (time_steps - window_size_steps) // step_size_steps + 1
        return max(0, window_count)
    
    def validate_window_shape(self, windows: np.ndarray) -> bool:
        """
        Validera att windows har korrekt shape enligt Master POC spec.
        
        Args:
            windows: Window array
            
        Returns:
            True om shape är korrekt
        """
        expected_shape = (None, self.config.window_size_seconds, self.config.timeseries_features)
        
        if len(windows.shape) != 3:
            logger.error(f"Windows måste ha 3 dimensioner, fick {len(windows.shape)}")
            return False
        
        if windows.shape[1] != self.config.window_size_seconds:
            logger.error(f"Window size måste vara {self.config.window_size_seconds}, fick {windows.shape[1]}")
            return False
        
        if windows.shape[2] != self.config.timeseries_features:
            logger.error(f"Timeseries features måste vara {self.config.timeseries_features}, fick {windows.shape[2]}")
            return False
        
        return True
    
    def validate_step_size(self, metadata: List[Dict[str, Any]]) -> bool:
        """
        Validera att step size är korrekt mellan windows.
        
        Args:
            metadata: Lista med window metadata
            
        Returns:
            True om step size är korrekt
        """
        if len(metadata) < 2:
            return True  # Ingen validering möjlig med < 2 windows
        
        expected_step = self.config.step_size_seconds
        
        for i in range(1, len(metadata)):
            actual_step = metadata[i]['start_time_step'] - metadata[i-1]['start_time_step']
            if actual_step != expected_step:
                logger.error(f"Step size mellan window {i-1} och {i} är {actual_step}, förväntat {expected_step}")
                return False
        
        return True
    
    def get_window_overlap_percentage(self) -> float:
        """
        Beräkna overlap procent mellan windows.
        
        Returns:
            Overlap procent
        """
        overlap_steps = self.config.window_size_seconds - self.config.step_size_seconds
        overlap_percentage = (overlap_steps / self.config.window_size_seconds) * 100
        return overlap_percentage
    
    def create_windows_from_dataframe(self, 
                                    df: pd.DataFrame, 
                                    timeseries_columns: List[str],
                                    static_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Skapa windows från pandas DataFrame.
        
        Args:
            df: DataFrame med timeseries data
            timeseries_columns: Lista med timeseries kolumnnamn
            static_data: Static features array
            
        Returns:
            Tuple of windows, static_windows, metadata
        """
        if len(timeseries_columns) != self.config.timeseries_features:
            raise ValueError(f"Måste ha exakt {self.config.timeseries_features} timeseries kolumner")
        
        # Extrahera timeseries data
        timeseries_data = df[timeseries_columns].values
        
        # Använd timestamps om tillgängliga
        timestamps = None
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
        elif df.index.name == 'timestamp':
            timestamps = df.index.values
        
        return self.create_sliding_windows(timeseries_data, static_data, timestamps)

def create_master_poc_window_creator(config: Optional[WindowConfig] = None) -> MasterPOCWindowCreator:
    """Factory function för att skapa MasterPOCWindowCreator."""
    return MasterPOCWindowCreator(config)
