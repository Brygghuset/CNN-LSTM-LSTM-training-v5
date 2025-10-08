import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Hanterar flera VitalDB datakällor och väljer automatiskt den bästa tillgängliga.
    """
    
    def __init__(self, config, force_test_mode=False, test_data_dir=None):
        self.config = config
        self.force_test_mode = force_test_mode
        self.test_data_dir = test_data_dir
        
        # Hantera både Dict och Config-objekt
        if hasattr(config, 'get') and callable(config.get):
            # Dict-format
            self.vitaldb_config = config.get('data', {}).get('vitaldb', {})
        else:
            # Config-objekt format - använd centraliserad konfiguration
            try:
                from config import get_config_manager
                config_manager = get_config_manager()
                data_config = config_manager.data
                
                # Försök hämta vitaldb-konfiguration
                if hasattr(data_config, 'vitaldb'):
                    self.vitaldb_config = data_config.vitaldb
                else:
                    # Fallback: försök hämta från config.yaml direkt
                    import yaml
                    config_path = "configs/config.yaml"
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            full_config = yaml.safe_load(f)
                        self.vitaldb_config = full_config.get('data', {}).get('vitaldb', {})
                    else:
                        self.vitaldb_config = {}
            except Exception as e:
                logger.error(f"Fel vid laddning av vitaldb-konfiguration: {e}")
                # Fallback till tom config
                self.vitaldb_config = {}
        
        # Uppdatera dataset_sources med S3 fallback-hierarki
        self.dataset_sources = [
            {
                'name': 's3_full_dataset',
                'path': 's3://master-poc-v1.0/raw-data/vital-files',
                'priority': 1,  # Högsta prioritet
                'description': 'Fullständigt VitalDB dataset på AWS S3 (Master POC)',
                'enabled': True,
                'type': 's3'
            },
            {
                'name': 'full_dataset_ssd',
                'path': '/Volumes/SSD T5 Max C/VitalDB Dataset/vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0/vital_files',
                'priority': 2,
                'description': 'Fullständigt VitalDB dataset på extern SSD',
                'enabled': True,
                'type': 'local'
            },
            {
                'name': 'local_dataset',
                'path': 'data/raw/vitaldb',
                'priority': 3,
                'description': 'Lokal VitalDB dataset med 5 cases',
                'enabled': True,
                'type': 'local'
            },
            {
                'name': 'fallback_dataset',
                'path': 'tests/fixtures',
                'priority': 4,
                'description': 'Mock-data för tester',
                'enabled': True,
                'type': 'local'
            }
        ]
        
        self.auto_select = True
        self.preferred_dataset = 's3_full_dataset'  # Uppdatera preferred till S3
        self.selected_source = None
        
        # S3 manager för AWS-operationer
        self.s3_manager = None
        self.s3_available = False
        
        # Om test-mode är tvingat, använd test_data_dir
        if self.force_test_mode and self.test_data_dir:
            self.selected_source = {
                'name': 'test_mode',
                'path': self.test_data_dir,
                'priority': 0,  # Högsta prioritet
                'description': 'Test mode - forced directory',
                'enabled': True,
                'type': 'local'
            }
            logger.info(f"Test mode aktiverat: använder {self.test_data_dir}")
        else:
            # Initialisera S3 manager
            self._initialize_s3_manager()
            self._validate_and_select_dataset()
    
    def _initialize_s3_manager(self):
        """Initialisera S3 manager för AWS-operationer."""
        try:
            from data.sagemaker_preprocessing import S3DataManager
            self.s3_manager = S3DataManager('master-poc-v1.0')
            self.s3_available = True
            logger.info("S3 manager initialiserad")
        except Exception as e:
            logger.warning(f"S3 manager kunde inte initialiseras: {e}")
            self.s3_manager = None
            self.s3_available = False
    
    def _validate_and_select_dataset(self):
        """Validera och välj den bästa tillgängliga datakällan med S3-prioritet."""
        available_sources = []
        
        for source in self.dataset_sources:
            if not source.get('enabled', True):
                logger.debug(f"Datakälla '{source['name']}' inaktiverad")
                continue
            
            # Validera baserat på typ
            if source.get('type') == 's3':
                if self._validate_s3_source(source):
                    available_sources.append(source)
                    logger.info(f"S3 datakälla '{source['name']}' tillgänglig: {source['path']}")
                else:
                    logger.warning(f"S3 datakälla '{source['name']}' inte tillgänglig: {source['path']}")
            else:
                # Lokal validering
                path = source['path']
                if self._validate_dataset_path(path):
                    available_sources.append(source)
                    logger.info(f"Lokal datakälla '{source['name']}' tillgänglig: {path}")
                else:
                    logger.warning(f"Lokal datakälla '{source['name']}' inte tillgänglig: {path}")
        
        if not available_sources:
            raise RuntimeError("Ingen tillgänglig VitalDB datakälla hittades!")
        
        # Sortera efter prioritet (lägre nummer = högre prioritet)
        available_sources.sort(key=lambda x: x.get('priority', 999))
        
        # Välj källa
        if self.auto_select:
            self.selected_source = available_sources[0]
            logger.info(f"Automatiskt vald datakälla: {self.selected_source['name']}")
        else:
            # Försök hitta preferred dataset
            preferred = next((s for s in available_sources if s['name'] == self.preferred_dataset), None)
            if preferred:
                self.selected_source = preferred
                logger.info(f"Använder preferred datakälla: {self.selected_source['name']}")
            else:
                self.selected_source = available_sources[0]
                logger.warning(f"Preferred datakälla '{self.preferred_dataset}' inte tillgänglig, använder: {self.selected_source['name']}")
    
    def _validate_dataset_path(self, path: str) -> bool:
        """Validera om en dataset-sökväg är tillgänglig och innehåller data"""
        if not os.path.exists(path):
            return False
        
        # Kontrollera om det finns .vital eller .csv filer
        vital_files = glob.glob(os.path.join(path, "*.vital"))
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        # För fullständigt dataset, kräv minst 10 cases
        # För lokalt dataset och test-mode, kräv minst 1 case
        min_cases = 1 if ('local_dataset' in path or 'test_mode' in path or self.force_test_mode or 'data/raw' in path) else 10
        
        if len(vital_files) >= min_cases or len(csv_files) >= min_cases:
            return True
        
        # Kontrollera om det finns undermappar med cases
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            vital_files = glob.glob(os.path.join(subdir_path, "*.vital"))
            csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
            if len(vital_files) >= min_cases or len(csv_files) >= min_cases:
                return True
        
        return False
    
    def _validate_s3_source(self, source: dict) -> bool:
        """Validera S3-datakälla."""
        if not self.s3_available or not self.s3_manager:
            return False
            
        try:
            # Lista filer i S3-bucket med prefix
            files = self.s3_manager.list_files(prefix="raw-data/vital-files")
            
            # Kräv minst 10 .vital filer för full dataset
            vital_files = [f for f in files if f.endswith('.vital')]
            return len(vital_files) >= 10
            
        except Exception as e:
            logger.warning(f"S3-validering misslyckades för {source['name']}: {e}")
            return False
    
    def get_dataset_path(self) -> str:
        """Hämta sökvägen till den valda datakällan"""
        if not self.selected_source:
            raise RuntimeError("Ingen datakälla vald!")
        return self.selected_source['path']
    
    def get_dataset_info(self) -> Dict:
        """Hämta information om den valda datakällan"""
        if not self.selected_source:
            return {}
        
        path = self.selected_source['path']
        vital_files = glob.glob(os.path.join(path, "*.vital"))
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        return {
            'name': self.selected_source['name'],
            'path': path,
            'description': self.selected_source['description'],
            'vital_files': len(vital_files),
            'csv_files': len(csv_files),
            'total_files': len(vital_files) + len(csv_files)
        }
    
    def get_available_cases(self, max_cases: Optional[int] = None) -> List[str]:
        """Hämta lista över tillgängliga cases"""
        if not self.selected_source:
            return []
        
        path = self.selected_source['path']
        cases = []
        
        # Hitta .vital filer
        vital_files = glob.glob(os.path.join(path, "*.vital"))
        for file in vital_files:
            case_id = os.path.splitext(os.path.basename(file))[0]
            # Filtrera bort kliniska filer
            if not self._is_clinical_file(case_id):
                cases.append(case_id)
        
        # Hitta .csv filer
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        for file in csv_files:
            case_id = os.path.splitext(os.path.basename(file))[0]
            # Filtrera bort kliniska filer och duplicerade cases
            if not self._is_clinical_file(case_id) and case_id not in cases:
                cases.append(case_id)
        
        # Sortera cases numeriskt (hantera 0001, 0002, etc.)
        def sort_key(case_id):
            return str(case_id)
        cases.sort(key=sort_key)
        
        # Begränsa antal cases om specificerat
        if max_cases:
            cases = cases[:max_cases]
        
        return cases
    
    def _is_clinical_file(self, case_id: str) -> bool:
        """Kontrollera om en fil är en klinisk datafil"""
        clinical_patterns = [
            'clinical_data', 'lab_data', 'track_names', 
            'clinical_parameters', 'lab_parameters', 'clinical info'
        ]
        case_id_lower = case_id.lower()
        return any(pattern in case_id_lower for pattern in clinical_patterns)
    
    def get_case_path(self, case_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Hämta sökvägar till vital och csv filer för ett case.
        Returnerar (vital_path, csv_path) där None betyder att filen inte finns.
        """
        if not self.selected_source:
            return None, None
        
        path = self.selected_source['path']
        vital_path = os.path.join(path, f"{case_id}.vital")
        csv_path = os.path.join(path, f"{case_id}.csv")
        
        return (
            vital_path if os.path.exists(vital_path) else None,
            csv_path if os.path.exists(csv_path) else None
        )
    
    def is_full_dataset(self) -> bool:
        """Kontrollera om vi använder det fullständiga datasetet"""
        return self.selected_source and self.selected_source['name'] == 'full_dataset'
    
    def is_test_dataset(self) -> bool:
        """Kontrollera om vi använder test-datasetet"""
        return self.selected_source and self.selected_source['name'] in ['test_mode', 'fallback_dataset']
    
    def get_dataset_stats(self) -> Dict:
        """Hämta statistik om datasetet"""
        if not self.selected_source:
            return {
                'name': None,
                'path': None,
                'total_cases': 0,
                'available_cases': [],
                'description': '',
                'is_full_dataset': False,
                'is_test_dataset': False
            }
        
        path = self.selected_source['path']
        cases = self.get_available_cases()
        return {
            'name': self.selected_source['name'],
            'path': path,
            'total_cases': len(cases),
            'available_cases': cases[:10],  # Visa första 10 cases
            'description': self.selected_source['description'],
            'is_full_dataset': self.is_full_dataset(),
            'is_test_dataset': self.is_test_dataset()
        } 