"""
TFRecord Standard Format v1.0 - Standalone

Fristående version för robust AWS SageMaker-integration.
Endast beroende: tensorflow, numpy, logging.
"""
import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class TFRecordStandardFormat:
    WINDOW_SHAPE = (300, 14)
    TARGET_SHAPE = (7,)
    STATIC_SHAPE = (6,)

    @staticmethod
    def get_feature_description() -> Dict[str, tf.io.FixedLenFeature]:
        return {
            'window': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string),
            'static': tf.io.FixedLenFeature([6], tf.float32),
            'window_shape': tf.io.FixedLenFeature([2], tf.int64),
            'target_shape': tf.io.FixedLenFeature([1], tf.int64),
            'case_id': tf.io.FixedLenFeature([], tf.string)
        }

    @staticmethod
    def serialize_example(window: np.ndarray, target: np.ndarray, static: np.ndarray, case_id: str = "") -> tf.train.Example:
        if window.shape != TFRecordStandardFormat.WINDOW_SHAPE:
            raise ValueError(f"Window shape {window.shape} != {TFRecordStandardFormat.WINDOW_SHAPE}")
        if target.shape != TFRecordStandardFormat.TARGET_SHAPE:
            raise ValueError(f"Target shape {target.shape} != {TFRecordStandardFormat.TARGET_SHAPE}")
        if static.shape != TFRecordStandardFormat.STATIC_SHAPE:
            raise ValueError(f"Static shape {static.shape} != {TFRecordStandardFormat.STATIC_SHAPE}")
        window_serialized = tf.io.serialize_tensor(tf.constant(window, dtype=tf.float32))
        target_serialized = tf.io.serialize_tensor(tf.constant(target, dtype=tf.float32))
        feature = {
            'window': tf.train.Feature(bytes_list=tf.train.BytesList(value=[window_serialized.numpy()])),
            'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_serialized.numpy()])),
            'static': tf.train.Feature(float_list=tf.train.FloatList(value=static.astype(np.float32))),
            'window_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(window.shape))),
            'target_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(target.shape))),
            'case_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[case_id.encode('utf-8')]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def parse_example(example_proto) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        feature_description = TFRecordStandardFormat.get_feature_description()
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        window_tensor = tf.io.parse_tensor(parsed_features['window'], out_type=tf.float32)
        target_tensor = tf.io.parse_tensor(parsed_features['target'], out_type=tf.float32)
        timeseries = tf.reshape(window_tensor, TFRecordStandardFormat.WINDOW_SHAPE)
        target = tf.reshape(target_tensor, TFRecordStandardFormat.TARGET_SHAPE)
        static = parsed_features['static']
        target = tf.where(tf.math.is_nan(target), tf.zeros_like(target), target)
        drug_target = target[:3]
        vent_target = target[3:]
        inputs = {'timeseries_input': timeseries, 'patient_input': static}
        targets = {'drug_output': drug_target, 'vent_output': vent_target}
        return inputs, targets

    @staticmethod
    def validate_tfrecord_compatibility(tfrecord_path: str) -> bool:
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            for example in dataset.take(1):
                inputs, targets = TFRecordStandardFormat.parse_example(example)
                if (inputs['timeseries_input'].shape == TFRecordStandardFormat.WINDOW_SHAPE and
                    inputs['patient_input'].shape == TFRecordStandardFormat.STATIC_SHAPE and
                    targets['drug_output'].shape == (3,) and
                    targets['vent_output'].shape == (4,)):
                    logger.info(f"✅ TFRecord {tfrecord_path} är kompatibel med standardformat v1.0")
                    return True
                else:
                    logger.error(f"❌ TFRecord {tfrecord_path} har felaktiga shapes")
                    return False
        except Exception as e:
            logger.error(f"❌ Fel vid validering av TFRecord {tfrecord_path}: {e}")
            return False 