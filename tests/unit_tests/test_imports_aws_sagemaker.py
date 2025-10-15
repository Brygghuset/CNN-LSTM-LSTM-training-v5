#!/usr/bin/env python3
"""
Test att imports fungerar korrekt både lokalt och i AWS SageMaker miljön.
"""
import sys
import os
import pytest

def test_imports_without_sys_path_manipulation():
    """Verifiera att imports fungerar utan sys.path manipulation."""
    # Simulera SageMaker environment där src/ innehåll är i /opt/ml/code/
    
    # Dessa imports ska fungera direkt
    from checkpoint_manager import MasterPOCCheckpointManager
    from memory_efficient_batch_processor import MemoryEfficientBatchProcessor
    from data.master_poc_preprocessing_orchestrator import MasterPOCPreprocessingOrchestrator
    from data.master_poc_preprocessing_pipeline import MasterPOCPreprocessingPipeline
    from utils.case_range_parser import parse_case_range
    
    assert MasterPOCCheckpointManager is not None
    assert MemoryEfficientBatchProcessor is not None
    assert MasterPOCPreprocessingOrchestrator is not None
    assert MasterPOCPreprocessingPipeline is not None
    assert parse_case_range is not None

def test_orchestrator_creation():
    """Verifiera att orchestrator kan skapas."""
    from data.master_poc_preprocessing_orchestrator import create_master_poc_orchestrator
    
    orchestrator = create_master_poc_orchestrator(
        s3_bucket='test-bucket',
        checkpoint_interval=10,
        batch_size=5,
        enable_s3=False  # Disable S3 för test
    )
    
    assert orchestrator is not None
    assert orchestrator.batch_size == 5
    assert orchestrator.checkpoint_interval == 10

def test_checkpoint_manager_creation():
    """Verifiera att checkpoint manager kan skapas."""
    from checkpoint_manager import create_checkpoint_manager, get_memory_usage_mb
    
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_path='test-checkpoint-path',
        enable_checkpoints=True,
        checkpoint_interval=10
    )
    
    assert checkpoint_manager is not None
    assert checkpoint_manager.enable_checkpoints is True
    assert checkpoint_manager.checkpoint_interval == 10
    
    # Test memory usage function
    memory_mb = get_memory_usage_mb()
    assert isinstance(memory_mb, float)
    assert memory_mb > 0

def test_case_range_parser():
    """Verifiera att case range parser fungerar."""
    from utils.case_range_parser import parse_case_range
    
    # Test range format
    cases = parse_case_range("1-5")
    assert cases == ["0001", "0002", "0003", "0004", "0005"]
    
    # Test comma-separated format
    cases = parse_case_range("1,3,5")
    assert cases == ["0001", "0003", "0005"]
    
    # Test mixed format
    cases = parse_case_range("1-3,5,7")
    assert cases == ["0001", "0002", "0003", "0005", "0007"]

def test_tfrecord_creator():
    """Verifiera att TFRecord creator kan skapas."""
    from data.master_poc_tfrecord_creator import MasterPOCTFRecordCreator
    
    creator = MasterPOCTFRecordCreator()
    assert creator is not None
    assert creator.config.timeseries_shape == (300, 16)
    assert creator.config.static_shape == (6,)
    assert creator.config.target_shape == (8,)

def test_s3_parallel_loader():
    """Verifiera att S3 parallel loader kan skapas."""
    from data.s3_parallel_loader import S3ParallelLoader
    
    loader = S3ParallelLoader(
        s3_bucket='test-bucket',
        max_workers=2,
        retry_attempts=1
    )
    
    assert loader is not None
    assert loader.s3_bucket == 'test-bucket'
    assert loader.max_workers == 2
    assert loader.retry_attempts == 1

def test_master_poc_pipeline():
    """Verifiera att Master POC pipeline kan skapas."""
    from data.master_poc_preprocessing_pipeline import MasterPOCPreprocessingPipeline, MasterPOCPreprocessingConfig
    
    config = MasterPOCPreprocessingConfig()
    pipeline = MasterPOCPreprocessingPipeline(config, enable_s3=False)
    
    assert pipeline is not None
    assert pipeline.config.timeseries_features == 16
    assert pipeline.config.static_features == 6
    assert pipeline.config.output_features == 8

if __name__ == "__main__":
    pytest.main([__file__])
