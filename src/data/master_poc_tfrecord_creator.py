#!/usr/bin/env python3
"""
Master POC TFRecord Creation Module
Implementerar TFRecord creation enligt Master POC specifikation
"""

import numpy as np
import pandas as pd
import logging
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import tensorflow as tf

logger = logging.getLogger(__name__)

@dataclass
class TFRecordConfig:
    """Konfiguration för TFRecord creation."""
    timeseries_shape: Tuple[int, int] = (300, 16)  # [time_steps, features]
    static_shape: Tuple[int] = (6,)                # [features]
    target_shape: Tuple[int] = (8,)                # [predictions]
    compression_type: str = "GZIP"                # Compression för TFRecord
    buffer_size: int = 1000                       # Buffer size för streaming

class MasterPOCTFRecordCreator:
    """
    Master POC TFRecord Creator enligt specifikation.
    
    Schema:
    - timeseries: [300, 16] - sliding windows med 16 timeseries features
    - static: [6] - age, sex, height, weight, bmi, asa
    - targets: [8] - 3 drugs + 5 ventilator predictions
    """
    
    def __init__(self, config: Optional[TFRecordConfig] = None):
        self.config = config or TFRecordConfig()
        logger.info(f"MasterPOCTFRecordCreator initialiserad:")
        logger.info(f"   Timeseries Shape: {self.config.timeseries_shape}")
        logger.info(f"   Static Shape: {self.config.static_shape}")
        logger.info(f"   Target Shape: {self.config.target_shape}")
        logger.info(f"   Compression: {self.config.compression_type}")
    
    def create_tfrecord_schema(self) -> Dict[str, tf.io.FixedLenSequenceFeature]:
        """
        Skapa TFRecord schema enligt Master POC specifikation.
        
        Returns:
            Dict med feature definitions för TensorFlow
        """
        schema = {
            'timeseries': tf.io.FixedLenSequenceFeature(
                shape=[],
                dtype=tf.float32,
                allow_missing=True
            ),
            'static': tf.io.FixedLenSequenceFeature(
                shape=[],
                dtype=tf.float32,
                allow_missing=True
            ),
            'targets': tf.io.FixedLenSequenceFeature(
                shape=[],
                dtype=tf.float32,
                allow_missing=True
            )
        }
        
        logger.info(f"TFRecord schema skapat: {list(schema.keys())}")
        return schema
    
    def create_tfrecord_example(self, 
                               timeseries_data: np.ndarray,
                               static_data: np.ndarray,
                               targets_data: np.ndarray) -> tf.train.Example:
        """
        Skapa en TFRecord Example från data.
        
        Args:
            timeseries_data: Shape [300, 16] - sliding window data
            static_data: Shape [6] - static patient features
            targets_data: Shape [8] - target predictions
            
        Returns:
            tf.train.Example för TFRecord
        """
        # Validera input shapes
        if timeseries_data.shape != self.config.timeseries_shape:
            raise ValueError(f"Timeseries data måste ha shape {self.config.timeseries_shape}, fick {timeseries_data.shape}")
        
        if static_data.shape != self.config.static_shape:
            raise ValueError(f"Static data måste ha shape {self.config.static_shape}, fick {static_data.shape}")
        
        if targets_data.shape != self.config.target_shape:
            raise ValueError(f"Targets data måste ha shape {self.config.target_shape}, fick {targets_data.shape}")
        
        # Konvertera till float32 och flatten för TensorFlow
        timeseries_flat = timeseries_data.astype(np.float32).flatten()
        static_flat = static_data.astype(np.float32).flatten()
        targets_flat = targets_data.astype(np.float32).flatten()
        
        # Skapa feature dict med float_list istället för bytes_list
        feature = {
            'timeseries': tf.train.Feature(float_list=tf.train.FloatList(value=timeseries_flat)),
            'static': tf.train.Feature(float_list=tf.train.FloatList(value=static_flat)),
            'targets': tf.train.Feature(float_list=tf.train.FloatList(value=targets_flat))
        }
        
        # Skapa Example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        return example
    
    def create_tfrecord_file(self, 
                           data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                           output_path: str,
                           split_name: str = "train") -> str:
        """
        Skapa TFRecord fil från lista med data.
        
        Args:
            data_list: Lista med (timeseries, static, targets) tuples
            output_path: Sökväg för output fil
            split_name: Namn på split (train/validation/test)
            
        Returns:
            Sökväg till skapad TFRecord fil
        """
        if not data_list:
            logger.warning(f"Ingen data för {split_name} split")
            return None
        
        # Skapa output directory om den inte finns
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skapa TFRecord fil
        tfrecord_path = f"{output_path}_{split_name}.tfrecord"
        
        options = tf.io.TFRecordOptions(compression_type=self.config.compression_type)
        
        with tf.io.TFRecordWriter(tfrecord_path, options=options) as writer:
            for i, (timeseries_data, static_data, targets_data) in enumerate(data_list):
                try:
                    example = self.create_tfrecord_example(timeseries_data, static_data, targets_data)
                    writer.write(example.SerializeToString())
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Skrivit {i + 1} examples till {split_name} TFRecord")
                
                except Exception as e:
                    logger.error(f"Fel vid skrivning av example {i}: {e}")
                    continue
        
        logger.info(f"TFRecord fil skapad: {tfrecord_path} med {len(data_list)} examples")
        return tfrecord_path
    
    def create_memory_efficient_tfrecord(self,
                                       windows_data: np.ndarray,
                                       static_data: np.ndarray,
                                       targets_data: np.ndarray,
                                       output_path: str,
                                       split_name: str = "train") -> str:
        """
        Skapa TFRecord med memory-efficient streaming.
        
        Args:
            windows_data: Shape [n_windows, 300, 16] - sliding windows
            static_data: Shape [n_windows, 6] - static features per window
            targets_data: Shape [n_windows, 8] - targets per window
            output_path: Sökväg för output fil
            split_name: Namn på split
            
        Returns:
            Sökväg till skapad TFRecord fil
        """
        if len(windows_data) == 0:
            logger.warning(f"Ingen data för {split_name} split")
            return None
        
        # Validera shapes
        if windows_data.shape[1:] != self.config.timeseries_shape:
            raise ValueError(f"Windows data måste ha shape [n_windows, {self.config.timeseries_shape[0]}, {self.config.timeseries_shape[1]}], fick {windows_data.shape}")
        
        if static_data.shape[1:] != self.config.static_shape:
            raise ValueError(f"Static data måste ha shape [n_windows, {self.config.static_shape[0]}], fick {static_data.shape}")
        
        if targets_data.shape[1:] != self.config.target_shape:
            raise ValueError(f"Targets data måste ha shape [n_windows, {self.config.target_shape[0]}], fick {targets_data.shape}")
        
        # Skapa output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skapa TFRecord fil
        tfrecord_path = f"{output_path}_{split_name}.tfrecord"
        
        options = tf.io.TFRecordOptions(compression_type=self.config.compression_type)
        
        with tf.io.TFRecordWriter(tfrecord_path, options=options) as writer:
            for i in range(len(windows_data)):
                try:
                    example = self.create_tfrecord_example(
                        windows_data[i],
                        static_data[i],
                        targets_data[i]
                    )
                    writer.write(example.SerializeToString())
                    
                    # Logga progress var 1000:e example
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Skrivit {i + 1}/{len(windows_data)} examples till {split_name} TFRecord")
                
                except Exception as e:
                    logger.error(f"Fel vid skrivning av example {i}: {e}")
                    continue
        
        logger.info(f"Memory-efficient TFRecord skapad: {tfrecord_path} med {len(windows_data)} examples")
        return tfrecord_path
    
    def read_tfrecord_file(self, tfrecord_path: str) -> List[Dict[str, np.ndarray]]:
        """
        Läsa TFRecord fil och returnera data.
        
        Args:
            tfrecord_path: Sökväg till TFRecord fil
            
        Returns:
            Lista med dictionaries innehållande parsed data
        """
        if not os.path.exists(tfrecord_path):
            raise FileNotFoundError(f"TFRecord fil finns inte: {tfrecord_path}")
        
        schema = self.create_tfrecord_schema()
        
        def parse_example(example_proto):
            parsed = tf.io.parse_single_example(example_proto, schema)
            return parsed
        
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=self.config.compression_type)
        dataset = dataset.map(parse_example)
        
        parsed_data = []
        for example in dataset:
            # Konvertera till numpy arrays och reshape
            timeseries_data = example['timeseries'].numpy().reshape(self.config.timeseries_shape)
            static_data = example['static'].numpy().reshape(self.config.static_shape)
            targets_data = example['targets'].numpy().reshape(self.config.target_shape)
            
            parsed_data.append({
                'timeseries': timeseries_data,
                'static': static_data,
                'targets': targets_data
            })
        
        logger.info(f"Läst {len(parsed_data)} examples från {tfrecord_path}")
        return parsed_data
    
    def validate_tfrecord_schema(self, tfrecord_path: str) -> bool:
        """
        Validera att TFRecord fil har korrekt schema.
        
        Args:
            tfrecord_path: Sökväg till TFRecord fil
            
        Returns:
            True om schema är korrekt
        """
        try:
            parsed_data = self.read_tfrecord_file(tfrecord_path)
            
            if not parsed_data:
                logger.error("TFRecord fil är tom")
                return False
            
            # Validera första example
            first_example = parsed_data[0]
            
            # Validera timeseries shape
            if first_example['timeseries'].shape != self.config.timeseries_shape:
                logger.error(f"Timeseries shape är {first_example['timeseries'].shape}, förväntat {self.config.timeseries_shape}")
                return False
            
            # Validera static shape
            if first_example['static'].shape != self.config.static_shape:
                logger.error(f"Static shape är {first_example['static'].shape}, förväntat {self.config.static_shape}")
                return False
            
            # Validera targets shape
            if first_example['targets'].shape != self.config.target_shape:
                logger.error(f"Targets shape är {first_example['targets'].shape}, förväntat {self.config.target_shape}")
                return False
            
            logger.info("TFRecord schema validering lyckades")
            return True
            
        except Exception as e:
            logger.error(f"TFRecord schema validering misslyckades: {e}")
            return False
    
    def get_tfrecord_info(self, tfrecord_path: str) -> Dict[str, Any]:
        """
        Hämta information om TFRecord fil.
        
        Args:
            tfrecord_path: Sökväg till TFRecord fil
            
        Returns:
            Dictionary med fil information
        """
        if not os.path.exists(tfrecord_path):
            return {"error": "Fil finns inte"}
        
        try:
            parsed_data = self.read_tfrecord_file(tfrecord_path)
            
            info = {
                "file_path": tfrecord_path,
                "file_size_mb": os.path.getsize(tfrecord_path) / (1024 * 1024),
                "num_examples": len(parsed_data),
                "timeseries_shape": self.config.timeseries_shape,
                "static_shape": self.config.static_shape,
                "targets_shape": self.config.target_shape,
                "compression": self.config.compression_type
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}

def create_master_poc_tfrecord_creator(config: Optional[TFRecordConfig] = None) -> MasterPOCTFRecordCreator:
    """Factory function för att skapa MasterPOCTFRecordCreator."""
    return MasterPOCTFRecordCreator(config)

def split_data_70_15_15(windows_data: np.ndarray, 
                        static_data: np.ndarray, 
                        targets_data: np.ndarray,
                        random_seed: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Dela data i 70/15/15 split för train/validation/test.
    
    Args:
        windows_data: Shape [n_windows, 300, 16] - sliding windows
        static_data: Shape [n_windows, 6] - static features per window
        targets_data: Shape [n_windows, 8] - targets per window
        random_seed: Seed för deterministisk randomization
        
    Returns:
        Tuple med (train_data, validation_data, test_data) dictionaries
    """
    n_windows = len(windows_data)
    
    if n_windows < 3:
        logger.warning(f"Dataset för litet för split: {n_windows} windows. Returnerar allt som train.")
        return (
            {'windows': windows_data, 'static': static_data, 'targets': targets_data},
            {'windows': np.array([]).reshape(0, 300, 16), 'static': np.array([]).reshape(0, 6), 'targets': np.array([]).reshape(0, 8)},
            {'windows': np.array([]).reshape(0, 300, 16), 'static': np.array([]).reshape(0, 6), 'targets': np.array([]).reshape(0, 8)}
        )
    
    # Skapa random indices för deterministisk split
    np.random.seed(random_seed)
    indices = np.random.permutation(n_windows)
    
    # Beräkna split boundaries
    train_end = int(0.7 * n_windows)
    val_end = int(0.85 * n_windows)  # 70% + 15% = 85%
    
    # Dela indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Skapa splits
    train_data = {
        'windows': windows_data[train_indices],
        'static': static_data[train_indices],
        'targets': targets_data[train_indices]
    }
    
    validation_data = {
        'windows': windows_data[val_indices],
        'static': static_data[val_indices],
        'targets': targets_data[val_indices]
    }
    
    test_data = {
        'windows': windows_data[test_indices],
        'static': static_data[test_indices],
        'targets': targets_data[test_indices]
    }
    
    logger.info(f"Data split: Train={len(train_indices)} ({len(train_indices)/n_windows*100:.1f}%), "
                f"Val={len(val_indices)} ({len(val_indices)/n_windows*100:.1f}%), "
                f"Test={len(test_indices)} ({len(test_indices)/n_windows*100:.1f}%)")
    
    return train_data, validation_data, test_data

def split_data_by_case_70_15_15(case_windows: Dict[str, List[np.ndarray]], 
                                case_static: Dict[str, np.ndarray], 
                                case_targets: Dict[str, List[np.ndarray]],
                                random_seed: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Dela data i 70/15/15 split baserat på cases (inte windows).
    Samma case hamnar alltid i samma split.
    
    Args:
        case_windows: Dict med case_id -> List[np.ndarray] (windows per case)
        case_static: Dict med case_id -> np.ndarray (static features per case)
        case_targets: Dict med case_id -> List[np.ndarray] (targets per case)
        random_seed: Seed för deterministisk randomization
        
    Returns:
        Tuple med (train_data, validation_data, test_data) dictionaries
    """
    case_ids = list(case_windows.keys())
    n_cases = len(case_ids)
    
    if n_cases < 3:
        logger.warning(f"För få cases för split: {n_cases} cases. Returnerar allt som train.")
        # Kombinera alla cases till train
        all_windows = []
        all_static = []
        all_targets = []
        
        for case_id in case_ids:
            all_windows.extend(case_windows[case_id])
            all_static.append(case_static[case_id])
            all_targets.extend(case_targets[case_id])
        
        return (
            {'windows': np.array(all_windows), 'static': np.array(all_static), 'targets': np.array(all_targets)},
            {'windows': np.array([]).reshape(0, 300, 16), 'static': np.array([]).reshape(0, 6), 'targets': np.array([]).reshape(0, 8)},
            {'windows': np.array([]).reshape(0, 300, 16), 'static': np.array([]).reshape(0, 6), 'targets': np.array([]).reshape(0, 8)}
        )
    
    # Skapa random case order för deterministisk split
    np.random.seed(random_seed)
    case_indices = np.random.permutation(n_cases)
    
    # Beräkna split boundaries baserat på cases
    train_end = int(0.7 * n_cases)
    val_end = int(0.85 * n_cases)  # 70% + 15% = 85%
    
    # Dela case indices
    train_case_indices = case_indices[:train_end]
    val_case_indices = case_indices[train_end:val_end]
    test_case_indices = case_indices[val_end:]
    
    # Skapa splits baserat på cases
    train_data = {'windows': [], 'static': [], 'targets': []}
    validation_data = {'windows': [], 'static': [], 'targets': []}
    test_data = {'windows': [], 'static': [], 'targets': []}
    
    # Train split
    for idx in train_case_indices:
        case_id = case_ids[idx]
        train_data['windows'].extend(case_windows[case_id])
        train_data['static'].append(case_static[case_id])
        train_data['targets'].extend(case_targets[case_id])
    
    # Validation split
    for idx in val_case_indices:
        case_id = case_ids[idx]
        validation_data['windows'].extend(case_windows[case_id])
        validation_data['static'].append(case_static[case_id])
        validation_data['targets'].extend(case_targets[case_id])
    
    # Test split
    for idx in test_case_indices:
        case_id = case_ids[idx]
        test_data['windows'].extend(case_windows[case_id])
        test_data['static'].append(case_static[case_id])
        test_data['targets'].extend(case_targets[case_id])
    
    # Konvertera till numpy arrays
    train_data = {
        'windows': np.array(train_data['windows']) if train_data['windows'] else np.array([]).reshape(0, 300, 16),
        'static': np.array(train_data['static']) if train_data['static'] else np.array([]).reshape(0, 6),
        'targets': np.array(train_data['targets']) if train_data['targets'] else np.array([]).reshape(0, 8)
    }
    
    validation_data = {
        'windows': np.array(validation_data['windows']) if validation_data['windows'] else np.array([]).reshape(0, 300, 16),
        'static': np.array(validation_data['static']) if validation_data['static'] else np.array([]).reshape(0, 6),
        'targets': np.array(validation_data['targets']) if validation_data['targets'] else np.array([]).reshape(0, 8)
    }
    
    test_data = {
        'windows': np.array(test_data['windows']) if test_data['windows'] else np.array([]).reshape(0, 300, 16),
        'static': np.array(test_data['static']) if test_data['static'] else np.array([]).reshape(0, 6),
        'targets': np.array(test_data['targets']) if test_data['targets'] else np.array([]).reshape(0, 8)
    }
    
    logger.info(f"Case-based split: Train={len(train_case_indices)} cases, "
                f"Val={len(val_case_indices)} cases, "
                f"Test={len(test_case_indices)} cases")
    
    return train_data, validation_data, test_data

def create_three_tfrecord_files(windows_data: np.ndarray,
                               static_data: np.ndarray,
                               targets_data: np.ndarray,
                               output_path: str,
                               random_seed: int = 42) -> Dict[str, str]:
    """
    Skapa tre separata TFRecord-filer: train.tfrecord, validation.tfrecord, test.tfrecord
    
    Args:
        windows_data: Shape [n_windows, 300, 16] - sliding windows
        static_data: Shape [n_windows, 6] - static features per window
        targets_data: Shape [n_windows, 8] - targets per window
        output_path: Base path för output filer
        random_seed: Seed för deterministisk split
        
    Returns:
        Dict med paths till skapade TFRecord-filer
    """
    # Skapa split
    train_data, validation_data, test_data = split_data_70_15_15(
        windows_data, static_data, targets_data, random_seed
    )
    
    # Skapa TFRecord creator
    tfrecord_creator = MasterPOCTFRecordCreator()
    
    # Skapa output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    created_files = {}
    
    # Skapa train.tfrecord
    if len(train_data['windows']) > 0:
        train_path = tfrecord_creator.create_memory_efficient_tfrecord(
            train_data['windows'], train_data['static'], train_data['targets'],
            output_path, "train"
        )
        created_files['train'] = train_path
    else:
        created_files['train'] = None
    
    # Skapa validation.tfrecord
    if len(validation_data['windows']) > 0:
        val_path = tfrecord_creator.create_memory_efficient_tfrecord(
            validation_data['windows'], validation_data['static'], validation_data['targets'],
            output_path, "validation"
        )
        created_files['validation'] = val_path
    else:
        created_files['validation'] = None
    
    # Skapa test.tfrecord
    if len(test_data['windows']) > 0:
        test_path = tfrecord_creator.create_memory_efficient_tfrecord(
            test_data['windows'], test_data['static'], test_data['targets'],
            output_path, "test"
        )
        created_files['test'] = test_path
    else:
        created_files['test'] = None
    
    logger.info(f"Skapade tre TFRecord-filer: {list(created_files.keys())}")
    return created_files

def create_split_metadata(train_data: Dict[str, np.ndarray],
                         validation_data: Dict[str, np.ndarray],
                         test_data: Dict[str, np.ndarray],
                         total_windows: int,
                         random_seed: int = 42) -> Dict[str, Any]:
    """
    Skapa metadata för split-statistik.
    
    Args:
        train_data: Train split data
        validation_data: Validation split data
        test_data: Test split data
        total_windows: Total antal windows
        random_seed: Seed som användes för split
        
    Returns:
        Dict med split metadata
    """
    train_windows = len(train_data['windows'])
    val_windows = len(validation_data['windows'])
    test_windows = len(test_data['windows'])
    
    metadata = {
        'split_info': {
            'total_windows': total_windows,
            'train_windows': train_windows,
            'validation_windows': val_windows,
            'test_windows': test_windows,
            'train_percentage': (train_windows / total_windows * 100) if total_windows > 0 else 0,
            'validation_percentage': (val_windows / total_windows * 100) if total_windows > 0 else 0,
            'test_percentage': (test_windows / total_windows * 100) if total_windows > 0 else 0,
        },
        'split_config': {
            'random_seed': random_seed,
            'split_ratio': '70/15/15',
            'split_method': 'window_based'
        },
        'data_shapes': {
            'timeseries_shape': [300, 16],
            'static_shape': [6],
            'targets_shape': [8]
        },
        'created_at': pd.Timestamp.now().isoformat(),
        'version': '1.0'
    }
    
    logger.info(f"Split metadata skapad: Train={train_windows} ({metadata['split_info']['train_percentage']:.1f}%), "
                f"Val={val_windows} ({metadata['split_info']['validation_percentage']:.1f}%), "
                f"Test={test_windows} ({metadata['split_info']['test_percentage']:.1f}%)")
    
    return metadata

def save_split_metadata(metadata: Dict[str, Any], output_path: str) -> str:
    """
    Spara split metadata till JSON-fil.
    
    Args:
        metadata: Metadata dictionary
        output_path: Base path för metadata fil
        
    Returns:
        Path till sparad metadata fil
    """
    import json
    
    # Skapa output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Skapa metadata fil path
    metadata_path = f"{output_path}_split_metadata.json"
    
    # Spara metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Split metadata sparad: {metadata_path}")
    return metadata_path