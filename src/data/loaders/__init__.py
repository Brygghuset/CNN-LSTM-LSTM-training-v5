"""
Loaders module för olika filtyper.
Centraliserade imports för att eliminera kodduplicering.
"""

from data.loaders.vital_loader import VitalLoader
from data.loaders.csv_loader import CSVLoader
from data.loaders.clinical_loader import ClinicalLoader

__all__ = ['VitalLoader', 'CSVLoader', 'ClinicalLoader'] 