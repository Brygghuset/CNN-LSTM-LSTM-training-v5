"""
Bakåtkompatibel preprocessing-fil som importerar från nya separerade moduler.

DEPRECATED: Använd direkt imports från src.data.preprocessing.data_preprocessor
och src.data.preprocessing.imputation_strategies för nya implementationer.

Denna fil behålls för bakåtkompatibilitet.
"""

# Import från de nya separerade modulerna
from data.preprocessing.imputation_methods import ImputationMethod
from data.preprocessing.data_preprocessor import DataPreprocessor, impute_missing_values, get_missing_values_stats

# Exportera allt för bakåtkompatibilitet
__all__ = [
    'ImputationMethod',
    'DataPreprocessor',
    'impute_missing_values',
    'get_missing_values_stats'
]

class DataPreprocessor:
    """Main preprocessor class that orchestrates all preprocessing steps."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Sätt upp validator
        self.validator = DataValidator(config.get('validation', {}))
        
        # Sätt upp pipeline
        from data.processing_pipeline import DataProcessingPipeline, PipelineConfig
        
        # Skapa pipeline config från befintlig config
        pipeline_config = PipelineConfig(
            interpolation_method=config.get('interpolation_method', 'linear'),
            validate_physiological=config.get('validate_physiological', True),
            target_frequency=config.get('target_frequency', 1.0),
            tolerance=config.get('tolerance', 0.001),
            max_jump_threshold=config.get('max_jump_threshold', 0.5),
            enable_quality_assurance=config.get('enable_quality_assurance', True)
        )
        
        self.pipeline = DataProcessingPipeline(pipeline_config)
        
        # Sätt upp enhetskonverterare
        self.unit_converter = UnitConverter()
        
        # Sätt upp feature mapper
        self.feature_mapper = FeatureMapper(config.get('feature_mapping', {}))
        
        # Sätt upp temporal synchronizer (fallback för legacy support)
        self.temporal_sync = TemporalSynchronizer(config.get('temporal_sync', {}))
        
        self.logger.info("DataPreprocessor initialized with new pipeline")
    
    def preprocess(self, data: pd.DataFrame, case_id: str = None) -> pd.DataFrame:
        """
        Preprocess data using the new pipeline.
        
        Args:
            data: Input DataFrame
            case_id: Optional case identifier for logging
            
        Returns:
            Preprocessed DataFrame
        """
        if case_id:
            self.logger.info(f"Starting preprocessing for case {case_id}")
        else:
            self.logger.info("Starting preprocessing")
        
        try:
            # Kör nya pipeline
            result = self.pipeline.process(data)
            
            if not result.is_successful():
                self.logger.error(f"Pipeline failed with {len(result.errors)} errors")
                for error in result.errors:
                    self.logger.error(f"  - {error}")
                
                if len(result.warnings) > 0:
                    self.logger.warning(f"Pipeline had {len(result.warnings)} warnings")
                    for warning in result.warnings:
                        self.logger.warning(f"  - {warning}")
                
                # Returnera original data vid fel
                return data
            
            # Logga varningar om de finns
            if result.has_warnings():
                self.logger.warning(f"Pipeline completed with {len(result.warnings)} warnings")
                for warning in result.warnings:
                    self.logger.warning(f"  - {warning}")
            
            # Logga pipeline metadata
            if 'steps' in result.metadata:
                for step_info in result.metadata['steps']:
                    step_name = step_info['step_name']
                    step_status = step_info['status']
                    self.logger.info(f"  Step {step_name}: {step_status}")
            
            processed_data = result.data
            
            # Utför ytterligare preprocessing steg som inte ingår i pipeline
            processed_data = self._apply_additional_preprocessing(processed_data, case_id)
            
            if case_id:
                self.logger.info(f"Preprocessing completed for case {case_id}: {data.shape} → {processed_data.shape}")
            else:
                self.logger.info(f"Preprocessing completed: {data.shape} → {processed_data.shape}")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            # Fallback till legacy preprocessing
            return self._legacy_preprocess(data, case_id)
    
    def _apply_additional_preprocessing(self, data: pd.DataFrame, case_id: str = None) -> pd.DataFrame:
        """
        Utför ytterligare preprocessing som inte ingår i huvudpipeline.
        
        Args:
            data: Data från pipeline
            case_id: Optional case identifier
            
        Returns:
            Fully preprocessed DataFrame
        """
        try:
            # Enhetskonvertering
            if hasattr(self.unit_converter, 'convert_units'):
                data = self.unit_converter.convert_units(data)
            
            # Feature mapping
            if hasattr(self.feature_mapper, 'map_features'):
                data = self.feature_mapper.map_features(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Additional preprocessing failed: {e}")
            return data
    
    def _legacy_preprocess(self, data: pd.DataFrame, case_id: str = None) -> pd.DataFrame:
        """
        Fallback till legacy preprocessing om nya pipeline misslyckas.
        
        Args:
            data: Input DataFrame
            case_id: Optional case identifier
            
        Returns:
            Preprocessed DataFrame using legacy methods
        """
        self.logger.warning("Falling back to legacy preprocessing")
        
        try:
            # Använd gamla temporal synchronizer
            if hasattr(self.temporal_sync, 'synchronize'):
                data = self.temporal_sync.synchronize(data)
            
            # Validering
            if hasattr(self.validator, 'validate'):
                validation_result = self.validator.validate(data)
                if not validation_result.get('is_valid', True):
                    self.logger.warning(f"Legacy validation warnings: {validation_result.get('warnings', [])}")
            
            # Enhetskonvertering
            if hasattr(self.unit_converter, 'convert_units'):
                data = self.unit_converter.convert_units(data)
            
            # Feature mapping
            if hasattr(self.feature_mapper, 'map_features'):
                data = self.feature_mapper.map_features(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Legacy preprocessing also failed: {e}")
            return data
    
    def get_pipeline_info(self) -> dict:
        """
        Hämta information om pipeline konfiguration.
        
        Returns:
            Dictionary med pipeline information
        """
        return {
            'pipeline_steps': self.pipeline.get_step_names(),
            'config': self.pipeline.config.to_dict(),
            'available_interpolation_methods': [
                'linear', 'cubic', 'nearest', 'forward_fill'
            ]
        } 