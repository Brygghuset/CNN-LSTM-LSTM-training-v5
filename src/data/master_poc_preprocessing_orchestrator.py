#!/usr/bin/env python3
"""
Master POC Preprocessing Orchestrator
====================================

Wrapper som kombinerar MasterPOCPreprocessingPipeline med befintlig infrastruktur:
- TFRecord writing och streaming
- S3 integration
- Checkpoint management
- Memory-efficient batch processing
- SageMaker environment integration
- Error handling och recovery

Denna orchestrator behåller alla beprövade infrastrukturfunktioner medan
MasterPOCPreprocessingPipeline hanterar core preprocessing logic.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import gc
import tempfile
import shutil

# Import Master POC core preprocessing
from data.master_poc_preprocessing_pipeline import MasterPOCPreprocessingPipeline, MasterPOCPreprocessingConfig

# Import befintlig infrastruktur
from checkpoint_manager import PreprocessingCheckpointManager, get_memory_usage_mb, create_checkpoint_manager
from memory_efficient_batch_processor import MemoryEfficientBatchProcessor
from data.tfrecord_standard_format import TFRecordStandardFormat

# Import konfiguration
from config import get_config

logger = logging.getLogger(__name__)


class MasterPOCPreprocessingOrchestrator:
    """
    Orchestrator som kombinerar MasterPOCPreprocessingPipeline med befintlig infrastruktur.
    
    Funktioner:
    - Core preprocessing med MasterPOCPreprocessingPipeline
    - TFRecord writing och streaming
    - S3 integration för temp storage
    - Checkpoint management för långa jobb
    - Memory-efficient batch processing
    - SageMaker environment integration
    - Error handling och recovery
    """
    
    def __init__(self, 
                 config: Optional[MasterPOCPreprocessingConfig] = None,
                 s3_bucket: str = 'anestesi-ai-631158448906-pilot',
                 checkpoint_interval: int = 10,
                 batch_size: int = 50,
                 enable_streaming: bool = True,
                 enable_s3: bool = True,
                 memory_efficient: bool = True,
                 enable_checkpoints: bool = False,
                 cloud_mode: bool = False):
        """
        Initiera Master POC Preprocessing Orchestrator.
        
        Args:
            config: MasterPOCPreprocessingConfig (auto-genereras om None)
            s3_bucket: S3 bucket för temp storage
            checkpoint_interval: Checkpoint var N:e case
            batch_size: Batch size för memory-efficient processing
            enable_streaming: Enable streaming TFRecord writing
            enable_s3: Om S3-support ska aktiveras för vital data loading
            memory_efficient: Om memory-efficient processing ska användas
            enable_checkpoints: Om checkpointing ska aktiveras
            cloud_mode: Om körning sker i molnet (SageMaker)
        """
        self.logger = logging.getLogger(__name__)
        
        # Core preprocessing engine
        if config is None:
            config = MasterPOCPreprocessingConfig(
                timeseries_features=16,
                static_features=6,
                total_input_features=22,
                output_features=8,
                window_size=300,
                step_size=30,
                target_range=(-1, 1),
                imputation_method="master_poc_smart_forward_fill"
            )
        
        self.master_poc_pipeline = MasterPOCPreprocessingPipeline(config, enable_s3=enable_s3, s3_bucket=s3_bucket)
        self.config = config
        self.enable_s3 = enable_s3
        self.memory_efficient = memory_efficient
        self.enable_checkpoints = enable_checkpoints
        self.cloud_mode = cloud_mode
        
        # Infrastructure components
        self.s3_bucket = s3_bucket
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.enable_streaming = enable_streaming
        
        # Initialize infrastructure
        self._initialize_infrastructure()
        
        # Stats
        self.processed_cases = 0
        self.successful_cases = []
        self.failed_cases = []
        self.total_windows = 0
        
        self.logger.info("🚀 Master POC Preprocessing Orchestrator initialized")
        self.logger.info(f"   Master POC Pipeline: {self.master_poc_pipeline.__class__.__name__}")
        self.logger.info(f"   S3 Bucket: {s3_bucket}")
        self.logger.info(f"   S3 Support: {enable_s3}")
        self.logger.info(f"   Checkpoint Interval: {checkpoint_interval}")
        self.logger.info(f"   Batch Size: {batch_size}")
        self.logger.info(f"   Streaming: {enable_streaming}")
    
    def _initialize_infrastructure(self):
        """Initiera infrastrukturkomponenter."""
        try:
            # Checkpoint manager (endast om enable_checkpoints=True)
            if self.enable_checkpoints:
                self.checkpoint_manager = create_checkpoint_manager(
                    job_id=f"master-poc-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    s3_bucket=self.s3_bucket,
                    checkpoint_interval=self.checkpoint_interval
                )
            else:
                self.checkpoint_manager = None
            
            # Memory-efficient batch processor (endast om memory_efficient=True)
            if self.memory_efficient:
                self.batch_processor = MemoryEfficientBatchProcessor(
                    batch_size=self.batch_size,
                    checkpoint_interval=self.checkpoint_interval,
                    s3_bucket=self.s3_bucket,
                    enable_streaming=self.enable_streaming
                )
            else:
                self.batch_processor = None
            
            # TFRecord writer
            self.tfrecord_writer = TFRecordStandardFormat()
            
            self.logger.info("✅ Infrastructure components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize infrastructure: {e}")
            raise
    
    def load_case_data(self, case_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Ladda case data med automatisk fallback-hierarki (S3 -> SSD -> Local -> Mock).
        
        Args:
            case_id: Case identifier (t.ex. "0001")
            
        Returns:
            Tuple med (timeseries_df, clinical_df)
        """
        return self.master_poc_pipeline.load_case_data(case_id)
    
    def preprocess_case(self, case_id: str, timeseries_df: pd.DataFrame, clinical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Processa case med Master POC pipeline (kompatibilitet med entry point).
        
        Args:
            case_id: Case identifier
            timeseries_df: Timeseries data
            clinical_df: Clinical data
            
        Returns:
            Dict med preprocessing resultat
        """
        return self.process_case(case_id, timeseries_df, clinical_df)
    
    def load_raw_case_data(self, case_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Ladda raw case data för unit conversion (utan feature mapping).
        
        Args:
            case_id: Case identifier (t.ex. "0001")
            
        Returns:
            Tuple med (raw_timeseries_df, clinical_df)
        """
        return self.master_poc_pipeline.load_raw_case_data(case_id)
    
    def process_case(self, 
                   case_id: str, 
                   timeseries_df: pd.DataFrame, 
                   clinical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Processa en enskild case med Master POC pipeline och infrastruktur.
        
        Args:
            case_id: Case identifier
            timeseries_df: Timeseries data från VitalDB
            clinical_df: Clinical data från VitalDB
            
        Returns:
            Dict med processing result och metadata
        """
        self.logger.info(f"🔄 Processing case {case_id} with Master POC orchestrator")
        start_time = datetime.now()
        
        try:
            # Steg 1: Core preprocessing med MasterPOCPreprocessingPipeline
            timeseries_windows, static_features, metadata = self.master_poc_pipeline.preprocess_case(
                case_id, timeseries_df, clinical_df
            )
            
            if timeseries_windows is None:
                raise ValueError(f"Master POC preprocessing failed for case {case_id}")
            
            # Steg 2: Validera Master POC compliance (utan targets)
            validation_result = self.master_poc_pipeline.validate_master_poc_compliance(
                timeseries_windows, None, static_features
            )
            
            if not validation_result['compliance']:
                self.logger.warning(f"⚠️ Case {case_id} har compliance-problem: {validation_result['issues']}")
            
            # Steg 3: Förbered för TFRecord writing (utan targets)
            tfrecord_data = self._prepare_tfrecord_data(
                timeseries_windows, static_features, case_id
            )
            
            # Steg 4: Memory cleanup
            del timeseries_windows, static_features
            gc.collect()
            
            # Steg 5: Uppdatera stats
            self.processed_cases += 1
            self.successful_cases.append(case_id)
            self.total_windows += len(tfrecord_data['timeseries_windows'])
            
            # Steg 6: Spara checkpoint om nödvändigt (endast om checkpoints aktiverade)
            if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(self.processed_cases):
                self._save_checkpoint()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'case_id': case_id,
                'status': 'success',
                'windows_count': len(tfrecord_data['timeseries_windows']),
                'timeseries_windows_count': len(tfrecord_data['timeseries_windows']),
                'processing_time_seconds': processing_time,
                'master_poc_compliance': validation_result['compliance'],
                'compliance_issues': validation_result.get('issues', []),
                'tfrecord_data': tfrecord_data,
                'metadata': metadata
            }
            
            self.logger.info(f"✅ Case {case_id} processed successfully: {len(tfrecord_data['timeseries_windows'])} timeseries windows in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Case {case_id} processing failed: {e}")
            self.processed_cases += 1  # Uppdatera counter även vid fel
            self.failed_cases.append(case_id)
            
            return {
                'case_id': case_id,
                'status': 'failed',
                'error': str(e),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _prepare_tfrecord_data(self, 
                              timeseries_windows: np.ndarray, 
                              static_features: np.ndarray, 
                              case_id: str) -> Dict[str, Any]:
        """
        Förbered data för TFRecord writing.
        
        Args:
            timeseries_windows: Preprocessade timeseries windows (n_windows, 300, 16)
            static_features: Static patient features (6,)
            case_id: Case identifier
            
        Returns:
            Dict med TFRecord-ready data (utan targets - de skapas av modellen)
        """
        try:
            # Validera shapes enligt Master POC
            expected_window_shape = (self.config.window_size, self.config.timeseries_features)
            expected_static_shape = (self.config.static_features,)
            
            if timeseries_windows.shape[1:] != expected_window_shape:
                raise ValueError(f"Timeseries window shape mismatch: {timeseries_windows.shape[1:]} != {expected_window_shape}")
            
            if static_features.shape != expected_static_shape:
                raise ValueError(f"Static features shape mismatch: {static_features.shape} != {expected_static_shape}")
            
            # Förbered TFRecord data (Master POC: endast input features, inga targets)
            tfrecord_data = {
                'timeseries_windows': timeseries_windows.astype(np.float32),
                'static_features': np.tile(static_features.astype(np.float32), (len(timeseries_windows), 1)),  # Replikera för varje window
                'case_ids': [case_id] * len(timeseries_windows),  # En case_id per window
                'case_id': case_id,  # Original case_id för kompatibilitet
                'timeseries_window_shape': expected_window_shape,
                'static_shape': expected_static_shape
            }
            
            self.logger.debug(f"TFRecord data prepared for case {case_id}: {len(timeseries_windows)} timeseries windows")
            return tfrecord_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare TFRecord data for case {case_id}: {e}")
            raise
    
    def process_batch(self, 
                     case_batch: List[Dict[str, Any]], 
                     output_path: str) -> Dict[str, Any]:
        """
        Processa en batch av cases med memory-efficient processing.
        
        Args:
            case_batch: Lista av case data (case_id, timeseries_df, clinical_df)
            output_path: Output path för TFRecord files
            
        Returns:
            Dict med batch processing result
        """
        self.logger.info(f"📦 Processing batch of {len(case_batch)} cases")
        batch_start_time = datetime.now()
        
        try:
            # Processa cases i batch
            batch_results = []
            batch_timeseries_windows = []
            batch_static_features = []
            batch_case_ids = []
            
            for case_data in case_batch:
                case_id = case_data['case_id']
                timeseries_df = case_data['timeseries_df']
                clinical_df = case_data['clinical_df']
                
                # Processa case
                result = self.process_case(case_id, timeseries_df, clinical_df)
                batch_results.append(result)
                
                if result['status'] == 'success':
                    tfrecord_data = result['tfrecord_data']
                    batch_timeseries_windows.extend(tfrecord_data['timeseries_windows'])
                    batch_static_features.extend(tfrecord_data['static_features'])
                    batch_case_ids.extend([case_id] * len(tfrecord_data['timeseries_windows']))
            
            # Skriv TFRecord batch
            if batch_timeseries_windows:
                tfrecord_path = self._write_tfrecord_batch(
                    batch_timeseries_windows, batch_static_features, 
                    batch_case_ids, output_path
                )
                
                batch_processing_time = (datetime.now() - batch_start_time).total_seconds()
                
                result = {
                    'batch_size': len(case_batch),
                    'successful_cases': len([r for r in batch_results if r['status'] == 'success']),
                    'failed_cases': len([r for r in batch_results if r['status'] == 'failed']),
                    'total_timeseries_windows': len(batch_timeseries_windows),
                    'tfrecord_path': tfrecord_path,
                    'processing_time_seconds': batch_processing_time,
                    'case_results': batch_results
                }
                
                self.logger.info(f"✅ Batch processed: {result['successful_cases']}/{result['batch_size']} cases, {result['total_timeseries_windows']} timeseries windows")
                return result
            else:
                raise ValueError("No successful cases in batch")
                
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return {
                'batch_size': len(case_batch),
                'successful_cases': 0,
                'failed_cases': len(case_batch),
                'total_timeseries_windows': 0,
                'error': str(e),
                'processing_time_seconds': (datetime.now() - batch_start_time).total_seconds()
            }
    
    def _write_tfrecord_batch(self, 
                            timeseries_windows: List[np.ndarray], 
                            static_features: List[np.ndarray], 
                            case_ids: List[str], 
                            output_path: str) -> str:
        """
        Skriv TFRecord batch med Master POC TFRecordStandardFormat v2.0.
        
        Args:
            timeseries_windows: Lista av timeseries window arrays
            static_features: Lista av static feature arrays
            case_ids: Lista av case IDs
            output_path: Output directory
            
        Returns:
            Path till skapad TFRecord fil
        """
        try:
            # Skapa output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Generera unikt filnamn
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tfrecord_filename = f"master_poc_batch_{timestamp}.tfrecord"
            tfrecord_path = os.path.join(output_path, tfrecord_filename)
            
            # Skriv TFRecord direkt (Master POC v2.0: utan targets)
            self._write_tfrecord_direct(
                timeseries_windows=timeseries_windows,
                static_features=static_features,
                case_ids=case_ids,
                output_path=tfrecord_path
            )
            
            self.logger.info(f"📄 TFRecord batch written: {tfrecord_path}")
            return tfrecord_path
            
        except Exception as e:
            self.logger.error(f"Failed to write TFRecord batch: {e}")
            raise
    
    def _write_tfrecord_direct(self, 
                              timeseries_windows: List[np.ndarray], 
                              static_features: List[np.ndarray], 
                              case_ids: List[str], 
                              output_path: str) -> None:
        """
        Skriv TFRecord direkt med Master POC format v2.0 (utan targets).
        """
        import tensorflow as tf
        from data.tfrecord_standard_format import TFRecordStandardFormat
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for i, (timeseries_window, static_feat, case_id) in enumerate(zip(timeseries_windows, static_features, case_ids)):
                # Serialize data enligt Master POC standardformat v2.0
                timeseries_bytes = tf.io.serialize_tensor(tf.constant(timeseries_window, dtype=tf.float32)).numpy()
                static_bytes = tf.constant(static_feat, dtype=tf.float32).numpy()
                
                # Create feature dict enligt Master POC v2.0 (utan targets)
                feature_dict = {
                    'timeseries_windows': tf.train.Feature(bytes_list=tf.train.BytesList(value=[timeseries_bytes])),
                    'static_features': tf.train.Feature(float_list=tf.train.FloatList(value=static_bytes)),
                    'timeseries_window_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(timeseries_window.shape))),
                    'static_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(static_feat.shape))),
                    'case_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[case_id.encode()]))
                }
                
                # Create example (utan targets - de skapas av modellen)
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
    
    def _save_checkpoint(self):
        """Spara checkpoint state."""
        if not self.checkpoint_manager:
            return
            
        try:
            checkpoint_saved = self.checkpoint_manager.save_checkpoint(
                processed_cases=self.processed_cases,
                successful_cases=self.successful_cases,
                failed_cases=self.failed_cases,
                processing_stats={
                    'total_windows': self.total_windows,
                    'master_poc_compliance': True,
                    'pipeline_version': 'master_poc_orchestrator_v1.0'
                },
                tfrecord_stats={
                    'files_written': len(self.successful_cases),
                    'total_windows': self.total_windows
                },
                memory_usage_mb=get_memory_usage_mb()
            )
            
            if checkpoint_saved:
                self.logger.info(f"💾 Checkpoint saved: {self.processed_cases} cases processed")
            else:
                self.logger.warning("⚠️ Failed to save checkpoint")
                
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Hämta processing statistik."""
        return {
            'processed_cases': self.processed_cases,
            'successful_cases': len(self.successful_cases),
            'failed_cases': len(self.failed_cases),
            'total_windows': self.total_windows,
            'success_rate': len(self.successful_cases) / max(self.processed_cases, 1),
            'master_poc_compliance': True,
            'pipeline_version': 'master_poc_orchestrator_v1.0'
        }
    
    def cleanup(self):
        """Rensa upp resurser."""
        try:
            # Memory cleanup
            gc.collect()
            
            # Spara final checkpoint
            if self.processed_cases > 0:
                self._save_checkpoint()
            
            self.logger.info("🧹 Master POC Orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def create_master_poc_orchestrator(config: Optional[MasterPOCPreprocessingConfig] = None,
                                 s3_bucket: str = 'anestesi-ai-631158448906-pilot',
                                 checkpoint_interval: int = 10,
                                 batch_size: int = 50,
                                 enable_s3: bool = True) -> MasterPOCPreprocessingOrchestrator:
    """
    Factory function för att skapa Master POC Preprocessing Orchestrator.
    
    Args:
        config: MasterPOCPreprocessingConfig
        s3_bucket: S3 bucket för temp storage
        checkpoint_interval: Checkpoint interval
        batch_size: Batch size för memory-efficient processing
        enable_s3: Om S3-support ska aktiveras för vital data loading
        
    Returns:
        MasterPOCPreprocessingOrchestrator instance
    """
    return MasterPOCPreprocessingOrchestrator(
        config=config,
        s3_bucket=s3_bucket,
        checkpoint_interval=checkpoint_interval,
        batch_size=batch_size,
        enable_streaming=True,
        enable_s3=enable_s3
    )
