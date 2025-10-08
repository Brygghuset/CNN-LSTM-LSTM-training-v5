"""
FileFinder utility för centraliserad filsökning.
Eliminerar kodduplicering i filsökning och path-hantering.
"""

import os
import glob
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import pandas as pd


class FileFinder:
    """Centraliserad klass för filsökning och path-hantering."""
    
    def __init__(self, data_dir: str, logger: Optional[logging.Logger] = None, clinical_data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.clinical_data_dir = Path(clinical_data_dir) if clinical_data_dir else self.data_dir.parent
        self.logger = logger or logging.getLogger(__name__)
    
    def find_file(self, filename: str) -> Optional[str]:
        """Hitta en specifik fil i data_dir."""
        filepath = self.data_dir / filename
        return str(filepath) if filepath.exists() else None
    
    def find_files_by_pattern(self, pattern: str) -> List[str]:
        """Hitta filer som matchar ett glob-pattern."""
        search_pattern = str(self.data_dir / pattern)
        return glob.glob(search_pattern)
    
    def find_all_files(self) -> List[str]:
        """Hitta alla filer i data_dir."""
        return self.find_files_by_pattern("*")
    
    def find_case_files(self, case_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Hitta timeseries och klinisk data för ett specifikt case."""
        # Konvertera case_id till sträng och hantera olika format
        case_id_str = str(case_id)
        
        # Möjliga timeseries filnamn (prioritera zero-padded format)
        # Hantera både zero-padded och icke-zero-padded format
        ts_patterns = []
        
        # Försök först med zero-padded format om case_id är numeriskt
        try:
            case_num = int(case_id_str)
            ts_patterns.extend([
                f"{case_num:04d}.vital",  # 0001.vital (högst prioritet)
                f"{case_num:04d}.csv",    # 0001.csv (hög prioritet)
            ])
        except ValueError:
            pass
        
        # Sedan med icke-zero-padded format
        ts_patterns.extend([
            f"{case_id_str}.vital",       # 1.vital
            f"{case_id_str}.csv",         # 1.csv
            f"case_{case_id_str}.vital",  # case_1.vital
            f"case_{case_id_str}.csv",    # case_1.csv
            f"Case{case_id_str}.vital",   # Case1.vital
            f"Case{case_id_str}.csv"      # Case1.csv
        ])
        
        # Möjliga kliniska filnamn
        cl_patterns = [
            f"{case_id_str}_clinical.csv",
            f"case_{case_id_str}_clinical.csv",
            f"case {case_id_str} Clinical info.csv",
            f"Case{case_id_str}Clinical.csv",
            f"Case{case_id_str}_clinical.csv"
        ]
        
        # Hitta timeseries fil
        ts_file = None
        for pattern in ts_patterns:
            found = self.find_file(pattern)
            if found:
                ts_file = found
                break
        
        # Hitta klinisk fil
        cl_file = None
        for pattern in cl_patterns:
            found = self.find_file(pattern)
            if found:
                cl_file = found
                break
        
        # Fallback till central clinical_data.csv (i clinical_data_dir)
        if not cl_file:
            central_clinical = self.clinical_data_dir / "clinical_data.csv"
            self.logger.info(f"Söker efter klinisk data i: {central_clinical}")
            if central_clinical.exists():
                cl_file = str(central_clinical)
                self.logger.info(f"Hittade klinisk data: {cl_file}")
            else:
                self.logger.warning(f"Klinisk data inte hittad: {central_clinical}")
        
        return ts_file, cl_file
    
    def get_file_info(self, filepath: str) -> dict:
        """Hämta information om en fil."""
        path = Path(filepath)
        if not path.exists():
            return {}
        
        return {
            'name': path.name,
            'size': path.stat().st_size,
            'modified': path.stat().st_mtime,
            'extension': path.suffix,
            'exists': True
        }
    
    def create_path(self, *parts: str) -> str:
        """Skapa en path med os.path.join."""
        return str(self.data_dir.joinpath(*parts))
    
    def ensure_dir(self, subdir: str = None) -> str:
        """Säkerställ att en katalog finns."""
        if subdir:
            dir_path = self.data_dir / subdir
        else:
            dir_path = self.data_dir
        
        dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path) 