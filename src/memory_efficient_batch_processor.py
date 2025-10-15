#!/usr/bin/env python3
"""
Memory-Efficient Batch Processor för AWS SageMaker
Enligt AWS SageMaker Preprocessing Guide v4.1

Hanterar 100+ cases med memory-efficient processing, streaming TFRecord writing,
och chunked case processing för optimal resource utilization.
"""

import os
import gc
import logging
import tempfile
import time
import shutil
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from checkpoint_manager import MasterPOCCheckpointManager, get_memory_usage_mb

# Setup logging
logger = logging.getLogger(__name__)

# Konstanter enligt AWS SageMaker Preprocessing Guide v4.1
TARGET_FEATURES = [7, 8, 9, 10, 11, 12, 13, 14]  # 3 drugs + 5 vent = 8 output features enligt Master POC

def normalize_data_pytorch(data: np.ndarray, range_bounds: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """DEPRECATED: Legacy normalisering - använd Master POC normalizers istället"""
    min_val, max_val = range_bounds
    
    # Standard min-max normalization till [-1, 1]
    data_min = np.nanmin(data, axis=0, keepdims=True)
    data_max = np.nanmax(data, axis=0, keepdims=True)
    
    # Undvik division med noll
    data_range = data_max - data_min
    data_range = np.where(data_range == 0, 1, data_range)
    
    # Normalisera till [0, 1] först
    normalized = (data - data_min) / data_range
    
    # Skala till önskad range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized.astype(np.float32)

class MemoryEfficientBatchProcessor:
    """
    Memory-Efficient Batch Processor för stora datasets (100+ cases)
    
    Features:
    - Chunked case processing för memory efficiency
    - Streaming TFRecord writing utan memory accumulation  
    - Batch-level checkpointing
    - Automatic memory cleanup
    - S3 integration för temp storage
    """
    
    def __init__(self, 
                 batch_size: int = 20,
                 checkpoint_interval: int = 5,
                 temp_storage_path: str = None,
                 s3_bucket: str = "anestesi-ai-631158448906-pilot",
                 enable_streaming: bool = True):
        """
        Initialisera Memory-Efficient Batch Processor
        
        Args:
            batch_size: Antal cases per batch (default: 20)
            checkpoint_interval: Checkpoint var N:e batch (default: 5)
            temp_storage_path: Path för temporär lagring (auto-detect SageMaker vs local)
            s3_bucket: S3 bucket för outputs
            enable_streaming: Enable streaming TFRecord writing
        """
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        # Auto-detect appropriate temp storage path
        if temp_storage_path is None:
            if os.path.exists('/opt/ml'):
                # SageMaker environment - use same filesystem as output
                self.temp_storage_path = '/opt/ml/temp'
            else:
                # Local development
                self.temp_storage_path = '/tmp'
        else:
            self.temp_storage_path = temp_storage_path
            
        self.s3_bucket = s3_bucket
        self.enable_streaming = enable_streaming
        
        # Ensure temp directory exists
        os.makedirs(self.temp_storage_path, exist_ok=True)
        
        # Initialize stats
        self.processed_batches = 0
        self.total_windows = 0
        self.failed_cases = []
        self.successful_cases = []
        
        logger.info(f"🚀 Memory-Efficient Batch Processor initialized")
        logger.info(f"   Batch Size: {batch_size} cases")
        logger.info(f"   Checkpoint Interval: {checkpoint_interval} batches")
        logger.info(f"   Streaming Enabled: {enable_streaming}")
        logger.info(f"   Temp Storage: {temp_storage_path}")
        
    def _create_case_batches(self, case_ids: List[str]) -> List[List[str]]:
        """
        Dela upp cases i batches för memory-efficient processing
        
        Args:
            case_ids: Lista av case IDs
            
        Returns:
            Lista av case batches
        """
        batches = []
        for i in range(0, len(case_ids), self.batch_size):
            batch = case_ids[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"📦 Created {len(batches)} batches from {len(case_ids)} cases")
        logger.info(f"   Batch sizes: {[len(batch) for batch in batches]}")
        
        return batches
    
    def _process_single_case_memory_efficient(self, case_id: str, 
                                            case_processor_func,
                                            **kwargs) -> Optional[Dict]:
        """
        Processa en enskild case med memory management
        
        Args:
            case_id: Case ID att processa
            case_processor_func: Funktion för case processing
            **kwargs: Additional arguments för case processor
            
        Returns:
            Case processing result eller None vid fel
        """
        try:
            # Process case med befintlig funktion
            result = case_processor_func(case_id, **kwargs)
            
            # Immediate memory cleanup
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Case {case_id} processing failed: {e}")
            self.failed_cases.append(case_id)
            return None
    
    def _process_batch(self, case_batch: List[str], 
                      case_processor_func,
                      **kwargs) -> List[Dict]:
        """
        Processa en batch av cases med memory management
        
        Args:
            case_batch: Lista av case IDs i denna batch
            case_processor_func: Funktion för case processing
            **kwargs: Additional arguments för case processor
            
        Returns:
            Lista av case processing results
        """
        logger.info(f"🔄 Processing batch med {len(case_batch)} cases")
        start_time = time.time()
        
        batch_results = []
        
        for i, case_id in enumerate(case_batch):
            logger.info(f"   Processing case {case_id} ({i+1}/{len(case_batch)})")
            
            # Process single case
            case_result = self._process_single_case_memory_efficient(
                case_id, case_processor_func, **kwargs
            )
            
            if case_result is not None:
                batch_results.append(case_result)
                self.successful_cases.append(case_id)
                
                # Log progress
                windows_count = len(case_result.get('windows', []))
                self.total_windows += windows_count
                logger.info(f"   ✅ Case {case_id}: {windows_count} windows")
            
            # Memory check
            memory_usage = get_memory_usage_mb()
            if memory_usage > 8000:  # 8GB threshold
                logger.warning(f"⚠️ High memory usage: {memory_usage:.1f} MB")
                gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Batch completed in {processing_time:.1f}s")
        logger.info(f"   Successful: {len(batch_results)}/{len(case_batch)} cases")
        logger.info(f"   Total windows: {sum(len(r.get('windows', [])) for r in batch_results)}")
        
        return batch_results
    
    def _stream_to_tfrecord(self, batch_results: List[Dict], 
                           batch_idx: int,
                           output_dir: str) -> Dict[str, int]:
        """
        Stream batch results direkt till TFRecord utan memory accumulation
        
        Args:
            batch_results: Lista av case processing results
            batch_idx: Batch index för filnamn
            output_dir: Output directory för TFRecord filer
            
        Returns:
            Dictionary med antal samples per split
        """
        if not self.enable_streaming:
            logger.info("📄 Streaming disabled, returning batch results")
            return self._combine_batch_results(batch_results)
        
        logger.info(f"📝 Streaming batch {batch_idx} till TFRecord")
        
        # Combine all data från batch
        all_windows = []
        all_targets = []
        all_static_features = []
        all_case_ids = []
        
        for case_result in batch_results:
            windows = case_result.get('windows', [])
            targets = case_result.get('targets', [])
            static_features = case_result.get('static_features', [])
            case_id = case_result.get('case_id', '')
            
            all_windows.extend(windows)
            all_targets.extend(targets)
            all_static_features.extend(static_features)
            all_case_ids.extend([case_id] * len(windows))
        
        if not all_windows:
            logger.warning(f"⚠️ No data in batch {batch_idx}")
            return {'train': 0, 'validation': 0, 'test': 0}
        
        # Convert till numpy arrays
        windows_array = np.array(all_windows)
        targets_array = np.array(all_targets)
        static_array = np.array(all_static_features)
        
        logger.info(f"   Combined data: {windows_array.shape[0]} samples")
        
        # Stratified split by case
        train_indices, val_indices, test_indices = self._stratified_split_by_case(
            all_case_ids, 0.7, 0.15, 0.15
        )
        
        # Write TFRecord files
        batch_stats = {}
        
        for split_name, indices in [
            ('train', train_indices),
            ('validation', val_indices), 
            ('test', test_indices)
        ]:
            if len(indices) == 0:
                batch_stats[split_name] = 0
                continue
                
            # Create temp TFRecord file
            temp_file = os.path.join(
                self.temp_storage_path, 
                f"batch_{batch_idx}_{split_name}.tfrecord"
            )
            
            # Write samples
            with tf.io.TFRecordWriter(temp_file) as writer:
                for idx in indices:
                    self._write_tfrecord_example(
                        writer, 
                        windows_array[idx],
                        targets_array[idx], 
                        static_array[idx],
                        all_case_ids[idx] # Pass case_id for legacy fallback
                    )
            
            # Move to output directory
            final_path = os.path.join(output_dir, f"batch_{batch_idx}_{split_name}.tfrecord")
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.move(temp_file, final_path)
            
            batch_stats[split_name] = len(indices)
            logger.info(f"   📄 {split_name}: {len(indices)} samples -> {final_path}")
        
        # Memory cleanup
        del all_windows, all_targets, all_static_features, all_case_ids
        del windows_array, targets_array, static_array
        gc.collect()
        
        return batch_stats
    
    def _stratified_split_by_case(self, case_ids: List[str],
                                 train_ratio: float, 
                                 validation_ratio: float,
                                 test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
        """
        Stratified split by case (samma som befintlig implementation)
        """
        unique_cases = list(set(case_ids))
        np.random.shuffle(unique_cases)
        
        train_end = int(len(unique_cases) * train_ratio)
        val_end = int(len(unique_cases) * (train_ratio + validation_ratio))
        
        train_cases = set(unique_cases[:train_end])
        val_cases = set(unique_cases[train_end:val_end])
        test_cases = set(unique_cases[val_end:])
        
        train_indices = [i for i, case_id in enumerate(case_ids) if case_id in train_cases]
        val_indices = [i for i, case_id in enumerate(case_ids) if case_id in val_cases]
        test_indices = [i for i, case_id in enumerate(case_ids) if case_id in test_cases]
        
        return train_indices, val_indices, test_indices
    
    def _write_tfrecord_example(self, writer, window, target, static_features, case_id="unknown"):
        """
        Skriv TFRecord example med Standard Format v1.0 för kompatibilitet
        """
        try:
            # Import TFRecord Standard Format (standalone version för SageMaker kompatibilitet)
            try:
                from data.tfrecord_standard_format_standalone import TFRecordStandardFormat
            except ImportError:
                # Fallback för lokala tester
                from data.tfrecord_standard_format_standalone import TFRecordStandardFormat
            
            # Använd Standard Format v1.0 för garanterad kompatibilitet
            example = TFRecordStandardFormat.serialize_example(
                window=window.astype(np.float32),
                target=target.astype(np.float32),
                static=static_features.astype(np.float32),
                case_id=case_id
            )
            
            writer.write(example.SerializeToString())
            
        except Exception as e:
            logger.error(f"❌ Failed to write TFRecord example with Standard Format v1.0: {e}")
            # Fallback till legacy format om Standard Format v1.0 failar
            logger.warning("⚠️ Falling back to legacy format")
            self._write_legacy_tfrecord_example(writer, window, target, static_features)
    
    def _write_legacy_tfrecord_example(self, writer, window, target, static_features):
        """
        Legacy TFRecord format (endast som fallback)
        """
        try:
            # Serialize tensors
            window_bytes = tf.io.serialize_tensor(tf.constant(window, dtype=tf.float32)).numpy()
            target_bytes = tf.io.serialize_tensor(tf.constant(target, dtype=tf.float32)).numpy()
            
            # Create feature dict
            feature = {
                'window': tf.train.Feature(bytes_list=tf.train.BytesList(value=[window_bytes])),
                'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_bytes])),
                'static': tf.train.Feature(float_list=tf.train.FloatList(value=static_features.tolist()))
            }
            
            # Create example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
        except Exception as e:
            logger.error(f"❌ Failed to write legacy TFRecord example: {e}")
    
    def _combine_batch_results(self, batch_results: List[Dict]) -> Dict[str, int]:
        """
        Combine batch results utan streaming (fallback)
        """
        total_samples = sum(len(r.get('windows', [])) for r in batch_results)
        
        # Simple split approximation
        train_samples = int(total_samples * 0.7)
        val_samples = int(total_samples * 0.15)
        test_samples = total_samples - train_samples - val_samples
        
        return {
            'train': train_samples,
            'validation': val_samples, 
            'test': test_samples
        }
    
    def _cleanup_memory(self):
        """
        Aggressiv memory cleanup
        """
        gc.collect()
        memory_usage = get_memory_usage_mb()
        logger.info(f"🧹 Memory cleanup: {memory_usage:.1f} MB")
    
    def _should_checkpoint(self, batch_idx: int) -> bool:
        """
        Kontrollera om checkpoint ska sparas
        """
        return (batch_idx + 1) % self.checkpoint_interval == 0
    
    def process_large_dataset(self, case_ids: List[str],
                             case_processor_func,
                             checkpoint_manager: MasterPOCCheckpointManager,
                             output_dir: str,
                             **kwargs) -> Dict:
        """
        Huvudfunktion för memory-efficient processing av stora datasets
        
        Args:
            case_ids: Lista av alla case IDs att processa
            case_processor_func: Funktion för case processing
            checkpoint_manager: Checkpoint manager instance
            output_dir: Output directory för TFRecord filer
            **kwargs: Additional arguments för case processor
            
        Returns:
            Processing statistics
        """
        logger.info(f"🚀 Starting memory-efficient processing av {len(case_ids)} cases")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Expected batches: {len(case_ids) // self.batch_size + 1}")
        
        # Create case batches
        case_batches = self._create_case_batches(case_ids)
        
        # Initialize stats
        total_stats = {'train': 0, 'validation': 0, 'test': 0}
        start_time = time.time()
        
        try:
            for batch_idx, case_batch in enumerate(case_batches):
                logger.info(f"📦 Processing batch {batch_idx + 1}/{len(case_batches)}")
                
                # Process batch
                batch_results = self._process_batch(
                    case_batch, case_processor_func, **kwargs
                )
                
                if batch_results:
                    # Stream till TFRecord
                    batch_stats = self._stream_to_tfrecord(
                        batch_results, batch_idx, output_dir
                    )
                    
                    # Update total stats
                    for split in total_stats:
                        total_stats[split] += batch_stats.get(split, 0)
                
                # Checkpoint
                if self._should_checkpoint(batch_idx):
                    checkpoint_stats = {
                        'processed_batches': batch_idx + 1,
                        'total_batches': len(case_batches),
                        'successful_cases': len(self.successful_cases),
                        'failed_cases': len(self.failed_cases),
                        'total_windows': self.total_windows,
                        'memory_usage_mb': get_memory_usage_mb()
                    }
                    
                    checkpoint_manager.save_checkpoint(
                        processed_cases=len(self.successful_cases),
                        successful_cases=self.successful_cases,
                        failed_cases=self.failed_cases,
                        processing_stats=checkpoint_stats,
                        tfrecord_stats=total_stats,
                        memory_usage_mb=get_memory_usage_mb()
                    )
                    
                    logger.info(f"💾 Checkpoint saved efter batch {batch_idx + 1}")
                
                # Memory cleanup
                self._cleanup_memory()
                
                self.processed_batches += 1
        
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            # Save emergency checkpoint
            checkpoint_manager.save_checkpoint(
                processed_cases=len(self.successful_cases),
                successful_cases=self.successful_cases,
                failed_cases=self.failed_cases,
                processing_stats={'error': str(e)},
                tfrecord_stats=total_stats,
                memory_usage_mb=get_memory_usage_mb()
            )
            raise
        
        processing_time = time.time() - start_time
        
        # Final stats
        final_stats = {
            'total_cases': len(case_ids),
            'successful_cases': len(self.successful_cases),
            'failed_cases': len(self.failed_cases),
            'success_rate': len(self.successful_cases) / len(case_ids) * 100,
            'total_windows': self.total_windows,
            'processing_time_seconds': processing_time,
            'processed_batches': self.processed_batches,
            'samples_per_split': total_stats
        }
        
        logger.info(f"🎉 Memory-efficient processing completed!")
        logger.info(f"   Total time: {processing_time:.1f} seconds")
        logger.info(f"   Success rate: {final_stats['success_rate']:.1f}%")
        logger.info(f"   Total windows: {self.total_windows}")
        logger.info(f"   Train/Val/Test: {total_stats['train']}/{total_stats['validation']}/{total_stats['test']}")
        
        return final_stats 