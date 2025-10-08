"""
TFRecord Standard Format v2.0 för Master POC CNN-LSTM-LSTM Pipeline

Detta är det officiella, standardiserade TFRecord-formatet som ska användas
för all preprocessing och training i Master POC CNN-LSTM-LSTM projektet.

Version: 2.0 (Master POC)
Datum: 2025-01-23
Giltig för: Master POC preprocessing (utan targets), TensorFlow training, och alla framtida implementationer

VIKTIGT: Preprocessing skapar ENDAST input features - targets skapas av modellen!
"""

import os
import sys

# Lägg till src path för att undvika import-problem
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# STANDARDISERAT TFRECORD FORMAT v1.0
# =============================================================================

class TFRecordStandardFormat:
    """
    Standardiserat TFRecord-format för Master POC CNN-LSTM-LSTM pipeline.
    
    Detta format ska användas av:
    - Master POC preprocessing (src/data/master_poc_preprocessing_orchestrator.py)
    - TensorFlow training (src/master_poc_cnn_lstm_lstm_training.py)
    - Memory efficient processing (src/memory_efficient_batch_processor.py)
    - Alla framtida implementationer
    
    VIKTIGT: Preprocessing TFRecord innehåller ENDAST input features - inga targets!
    """
    
    # Konstanter för Master POC standardformatet v2.0
    TIMESERIES_WINDOW_SHAPE = (300, 16)  # 300 timestamps, 16 Master POC features
    STATIC_SHAPE = (6,)                  # age, sex, height, weight, bmi, asa
    # TARGET_SHAPE tas bort - targets skapas av modellen, inte preprocessing!
    
    @staticmethod
    def get_feature_description() -> Dict[str, tf.io.FixedLenFeature]:
        """
        Returnerar den standardiserade feature description för Master POC preprocessing parsing.
        
        VIKTIGT: Detta är det ENDA giltiga formatet för Master POC TFRecord parsing.
        Innehåller ENDAST input features - inga targets!
        
        Returns:
            Dict med feature descriptions för tf.io.parse_single_example
        """
        return {
            # Timeseries data som serialized tensor för flexibilitet
            'timeseries_windows': tf.io.FixedLenFeature([], tf.string),
            
            # Static features som float array (RIKTIG PATIENTDATA, EJ MOCK)
            'static_features': tf.io.FixedLenFeature([6], tf.float32),
            
            # Metadata för validering (optional men rekommenderat)
            'timeseries_window_shape': tf.io.FixedLenFeature([2], tf.int64),
            'static_shape': tf.io.FixedLenFeature([1], tf.int64),
            
            # Case ID för debugging och tracking
            'case_id': tf.io.FixedLenFeature([], tf.string)
        }
    
    @staticmethod
    def serialize_example(
        timeseries_windows: np.ndarray,
        static_features: np.ndarray,
        case_id: str = ""
    ) -> tf.train.Example:
        """
        Serialiserar ett exempel enligt Master POC standardformatet v2.0.
        
        Args:
            timeseries_windows: Timeseries data shape (n_windows, 300, 16) enligt Master POC
            static_features: Static features shape (6,) - age, sex, height, weight, bmi, asa
            case_id: Case identifier för tracking
            
        Returns:
            tf.train.Example i Master POC standardformat (utan targets)
            
        Raises:
            ValueError: Om shapes inte matchar standardformatet
        """
        # Validera shapes
        if timeseries_windows.shape[1:] != TFRecordStandardFormat.TIMESERIES_WINDOW_SHAPE:
            raise ValueError(f"Timeseries window shape {timeseries_windows.shape[1:]} != expected {TFRecordStandardFormat.TIMESERIES_WINDOW_SHAPE}")
        
        if static_features.shape != TFRecordStandardFormat.STATIC_SHAPE:
            raise ValueError(f"Static features shape {static_features.shape} != expected {TFRecordStandardFormat.STATIC_SHAPE}")
        
        # Serialisera tensors
        timeseries_serialized = tf.io.serialize_tensor(tf.constant(timeseries_windows, dtype=tf.float32))
        
        # Skapa feature dictionary (utan targets)
        feature = {
            'timeseries_windows': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[timeseries_serialized.numpy()])
            ),
            'static_features': tf.train.Feature(
                float_list=tf.train.FloatList(value=static_features.astype(np.float32))
            ),
            'timeseries_window_shape': tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(timeseries_windows.shape[1:]))
            ),
            'static_shape': tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(static_features.shape))
            ),
            'case_id': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[case_id.encode('utf-8')])
            )
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    @staticmethod
    def parse_example(example_proto) -> Dict[str, tf.Tensor]:
        """
        Parsear TFRecord example enligt Master POC standardformatet v2.0.
        
        Args:
            example_proto: Serialized tf.train.Example
            
        Returns:
            Dict med inputs för Master POC CNN-LSTM-LSTM model:
            - inputs: {'timeseries_input': (300, 16), 'static_input': (6,)}
            
        VIKTIGT: Returnerar ENDAST inputs - targets skapas av modellen!
        """
        # Parse med standardiserad feature description
        feature_description = TFRecordStandardFormat.get_feature_description()
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Deserialize tensors
        timeseries_tensor = tf.io.parse_tensor(parsed_features['timeseries_windows'], out_type=tf.float32)
        
        # Reshape till expected shapes
        timeseries = tf.reshape(timeseries_tensor, TFRecordStandardFormat.TIMESERIES_WINDOW_SHAPE)
        
        # Static features (RIKTIG PATIENTDATA från klinisk data)
        static_features = parsed_features['static_features']
        
        # Returnera i Master POC CNN-LSTM-LSTM model format (endast inputs)
        inputs = {
            'timeseries_input': timeseries,
            'static_input': static_features
        }
        
        return inputs
    
    @staticmethod
    def validate_tfrecord_compatibility(tfrecord_path: str) -> bool:
        """
        Validerar att en TFRecord-fil är kompatibel med standardformatet.
        
        Args:
            tfrecord_path: Sökväg till TFRecord-fil
            
        Returns:
            True om kompatibel, False annars
        """
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            
            for example in dataset.take(1):
                inputs, targets = TFRecordStandardFormat.parse_example(example)
                
                # Validera shapes
                timeseries_shape = inputs['timeseries_input'].shape
                static_shape = inputs['patient_input'].shape
                drug_shape = targets['drug_output'].shape
                vent_shape = targets['vent_output'].shape
                
                expected_shapes = {
                    'timeseries': TFRecordStandardFormat.WINDOW_SHAPE,
                    'static': TFRecordStandardFormat.STATIC_SHAPE,
                    'drug': (3,),
                    'vent': (4,)
                }
                
                if (timeseries_shape == expected_shapes['timeseries'] and
                    static_shape == expected_shapes['static'] and
                    drug_shape == expected_shapes['drug'] and
                    vent_shape == expected_shapes['vent']):
                    
                    logger.info(f"✅ TFRecord {tfrecord_path} är kompatibel med standardformat v1.0")
                    return True
                else:
                    logger.error(f"❌ TFRecord {tfrecord_path} har inkompatibla shapes:")
                    logger.error(f"   Timeseries: {timeseries_shape} != {expected_shapes['timeseries']}")
                    logger.error(f"   Static: {static_shape} != {expected_shapes['static']}")
                    logger.error(f"   Drug: {drug_shape} != {expected_shapes['drug']}")
                    logger.error(f"   Vent: {vent_shape} != {expected_shapes['vent']}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Fel vid validering av TFRecord {tfrecord_path}: {e}")
            return False
    
    @staticmethod
    def convert_legacy_format_to_standard(
        legacy_tfrecord_path: str,
        output_path: str,
        legacy_format: str = "auto"
    ) -> bool:
        """
        Konverterar äldre TFRecord-format till standardformat v1.0.
        
        Args:
            legacy_tfrecord_path: Sökväg till befintlig TFRecord
            output_path: Sökväg för konverterad fil
            legacy_format: "pytorch", "sagemaker", "memory_efficient", eller "auto"
            
        Returns:
            True om konvertering lyckades
        """
        logger.info(f"🔄 Konverterar {legacy_tfrecord_path} till standardformat v1.0")
        
        try:
            # Auto-detect format om inte specificerat
            if legacy_format == "auto":
                legacy_format = TFRecordStandardFormat._detect_legacy_format(legacy_tfrecord_path)
            
            # Läs legacy format och konvertera
            dataset = tf.data.TFRecordDataset(legacy_tfrecord_path)
            
            with tf.io.TFRecordWriter(output_path) as writer:
                for example_proto in dataset:
                    try:
                        # Parse enligt legacy format
                        window, target, static, case_id = TFRecordStandardFormat._parse_legacy_example(
                            example_proto, legacy_format
                        )
                        
                        # Konvertera till standardformat
                        standard_example = TFRecordStandardFormat.serialize_example(
                            window, target, static, case_id
                        )
                        
                        writer.write(standard_example.SerializeToString())
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Hoppade över korrupt exempel: {e}")
                        continue
            
            # Validera konverterad fil
            if TFRecordStandardFormat.validate_tfrecord_compatibility(output_path):
                logger.info(f"✅ Konvertering lyckades: {output_path}")
                return True
            else:
                logger.error(f"❌ Konvertering misslyckades för {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fel vid konvertering: {e}")
            return False
    
    @staticmethod
    def _detect_legacy_format(tfrecord_path: str) -> str:
        """Detekterar vilket legacy format som används."""
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            
            for raw_example in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_example.numpy())
                
                features = example.features.feature
                
                if 'window' in features and 'target' in features and 'static' in features:
                    if features['window'].HasField('bytes_list'):
                        return "pytorch"
                    else:
                        return "memory_efficient"
                elif 'timeseries' in features:
                    return "sagemaker"
                else:
                    return "unknown"
                    
        except Exception:
            return "unknown"
    
    @staticmethod
    def _parse_legacy_example(example_proto, format_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Parsear legacy format och returnera standardiserade värden."""
        
        if format_type == "pytorch":
            # PyTorch format (senaste 500 cases)
            feature_desc = {
                'window': tf.io.FixedLenFeature([], tf.string),
                'target': tf.io.FixedLenFeature([], tf.string),
                'static': tf.io.FixedLenFeature([6], tf.float32),
                'case_id': tf.io.FixedLenFeature([], tf.string, default_value="")
            }
            
            parsed = tf.io.parse_single_example(example_proto, feature_desc)
            
            window = tf.io.parse_tensor(parsed['window'], out_type=tf.float32).numpy()
            target = tf.io.parse_tensor(parsed['target'], out_type=tf.float32).numpy()
            static = parsed['static'].numpy()
            case_id = parsed['case_id'].numpy().decode('utf-8') if parsed['case_id'].numpy() else ""
            
        elif format_type == "sagemaker":
            # SageMaker format
            feature_desc = {
                'timeseries': tf.io.FixedLenFeature([300 * 14], tf.float32),
                'static': tf.io.FixedLenFeature([6], tf.float32),
                'target': tf.io.FixedLenFeature([7], tf.float32),
                'case_id': tf.io.FixedLenFeature([], tf.string, default_value="")
            }
            
            parsed = tf.io.parse_single_example(example_proto, feature_desc)
            
            window = tf.reshape(parsed['timeseries'], [300, 16]).numpy()
            target = parsed['target'].numpy()
            static = parsed['static'].numpy()
            case_id = parsed['case_id'].numpy().decode('utf-8') if parsed['case_id'].numpy() else ""
            
        elif format_type == "memory_efficient":
            # Memory efficient format
            feature_desc = {
                'window': tf.io.FixedLenFeature([], tf.string),
                'target': tf.io.FixedLenFeature([], tf.string),
                'static': tf.io.FixedLenFeature([6], tf.float32),
                'case_id': tf.io.FixedLenFeature([], tf.string, default_value="")
            }
            
            parsed = tf.io.parse_single_example(example_proto, feature_desc)
            
            # Memory efficient använder tensorflow serialisering
            window = tf.io.parse_tensor(parsed['window'], out_type=tf.float32).numpy()
            target = tf.io.parse_tensor(parsed['target'], out_type=tf.float32).numpy()
            static = parsed['static'].numpy()
            case_id = parsed['case_id'].numpy().decode('utf-8') if parsed['case_id'].numpy() else ""
            
        else:
            raise ValueError(f"Okänt legacy format: {format_type}")
        
        return window, target, static, case_id


# =============================================================================
# HJÄLPFUNKTIONER FÖR STANDARDFORMAT
# =============================================================================

def create_standard_tfrecord_writer(output_path: str) -> tf.io.TFRecordWriter:
    """
    Skapar en TFRecordWriter för standardformat.
    
    Args:
        output_path: Sökväg för output-fil
        
    Returns:
        TFRecordWriter redo för standardformat
    """
    return tf.io.TFRecordWriter(output_path)


def create_standard_dataset(
    tfrecord_path: str,
    batch_size: int = 32,
    validation_split: float = 0.15,
    shuffle_buffer: int = 1000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Skapar training och validation datasets från standardformat TFRecord.
    
    Args:
        tfrecord_path: Sökväg till TFRecord-fil
        batch_size: Batch size för training
        validation_split: Andel data för validering (0.0-1.0)
        shuffle_buffer: Buffer size för shuffling
        
    Returns:
        Tuple av (train_dataset, val_dataset)
    """
    # Validera format först
    if not TFRecordStandardFormat.validate_tfrecord_compatibility(tfrecord_path):
        raise ValueError(f"TFRecord {tfrecord_path} är inte kompatibel med standardformat v1.0")
    
    # Skapa dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(TFRecordStandardFormat.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Räkna totalt antal exempel
    dataset_size = sum(1 for _ in dataset)
    
    # Skapa train/val split
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Shuffle och split
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Batch och prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"📊 Dataset skapad: {train_size} training, {val_size} validation samples")
    
    return train_dataset, val_dataset


# =============================================================================
# MIGRATION HJÄLPFUNKTIONER
# =============================================================================

def migrate_all_legacy_tfrecords(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.tfrecord"
) -> List[str]:
    """
    Migrerar alla legacy TFRecord-filer till standardformat v1.0.
    
    Args:
        input_dir: Directory med legacy TFRecord-filer
        output_dir: Directory för konverterade filer
        pattern: Fil-pattern för att hitta TFRecord-filer
        
    Returns:
        Lista med framgångsrikt konverterade filer
    """
    import glob
    import os
    
    # Hitta alla TFRecord-filer
    search_pattern = os.path.join(input_dir, pattern)
    tfrecord_files = glob.glob(search_pattern)
    
    converted_files = []
    
    for tfrecord_path in tfrecord_files:
        filename = os.path.basename(tfrecord_path)
        output_path = os.path.join(output_dir, f"standard_{filename}")
        
        # Hoppa över om redan konverterad
        if os.path.exists(output_path):
            logger.info(f"⏭️ Hoppade över {filename} (redan konverterad)")
            continue
        
        # Konvertera till standardformat
        if TFRecordStandardFormat.convert_legacy_format_to_standard(tfrecord_path, output_path):
            converted_files.append(output_path)
            logger.info(f"✅ Konverterade {filename} → standard_{filename}")
        else:
            logger.error(f"❌ Misslyckades konvertera {filename}")
    
    logger.info(f"📊 Migration slutförd: {len(converted_files)}/{len(tfrecord_files)} filer konverterade")
    
    return converted_files 