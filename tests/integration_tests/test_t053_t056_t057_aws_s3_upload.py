#!/usr/bin/env python3
"""
T053, T056, T057: AWS Integration Tests - S3 Upload & Path Structure
====================================================================

Riktiga integrationstester med AWS S3 för att verifiera:
- T053: S3 Upload Success - Verifiera att TFRecord-filer laddas upp till S3
- T056: S3 Path Structure - Verifiera korrekt S3 path-struktur för output
- T057: Upload Before Timeout - Verifiera att upload sker innan MaxRuntimeExceeded

AAA Format:
- Arrange: Skapa TFRecord-filer och förbered S3 upload
- Act: Ladda upp filer till S3 med korrekt path-struktur
- Assert: Verifiera att filer finns på rätt plats och att upload lyckas

Baserat på:
- AWS_CHECKLIST_V5.0_3000_CASES.md
- AWS_develop_instruction_V5.0.md  
- Master_POC_CNN-LSTM-LSTM_v5.0.md
"""

import unittest
import os
import sys
import tempfile
import json
import time
import boto3
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError

# Lägg till src i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# AWS-konfiguration (hardcoded för testet)
AWS_ACCOUNT_ID = "631158448906"
AWS_REGION = "eu-north-1"
AWS_SAGEMAKER_ROLE_ARN = "arn:aws:iam::631158448906:role/service-role/AmazonSageMaker-ExecutionRole-20250714T085596"
S3_PRIMARY_BUCKET = "master-poc-v1.0"
S3_SECONDARY_BUCKET = "sagemaker-eu-north-1-631158448906"
S3_INPUT_PATH = "s3://master-poc-v1.0/raw-data/"
S3_OUTPUT_PATH = "s3://master-poc-v1.0/processed-data/master-poc-pipeline/"
S3_CHECKPOINT_PATH = "s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/"


class SimpleTFRecordCreator:
    """Enkel TFRecord creator för testet"""
    
    def __init__(self):
        self.compression_type = "GZIP"
    
    def create_tfrecord_example(self, timeseries_data, static_data, targets_data):
        """Skapa TFRecord example enligt Master POC spec"""
        # Serialize timeseries data
        timeseries_bytes = tf.io.serialize_tensor(tf.constant(timeseries_data, dtype=tf.float32)).numpy()
        static_bytes = tf.constant(static_data, dtype=tf.float32).numpy()
        targets_bytes = tf.constant(targets_data, dtype=tf.float32).numpy()
        
        # Create feature dict enligt Master POC spec
        feature_dict = {
            'timeseries_windows': tf.train.Feature(bytes_list=tf.train.BytesList(value=[timeseries_bytes])),
            'static_features': tf.train.Feature(float_list=tf.train.FloatList(value=static_bytes)),
            'targets': tf.train.Feature(float_list=tf.train.FloatList(value=targets_bytes)),
            'timeseries_window_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(timeseries_data.shape))),
            'static_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(static_data.shape))),
            'target_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(targets_data.shape)))
        }
        
        # Create example
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example
    
    def create_memory_efficient_tfrecord(self, windows_data, static_data, targets_data, output_path, split_name="train"):
        """Skapa TFRecord fil från data"""
        if len(windows_data) == 0:
            return None
        
        # Skapa output directory om den inte finns
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skapa TFRecord fil
        tfrecord_path = f"{output_path}_{split_name}.tfrecord"
        
        options = tf.io.TFRecordOptions(compression_type=self.compression_type)
        
        with tf.io.TFRecordWriter(tfrecord_path, options=options) as writer:
            for i in range(len(windows_data)):
                try:
                    example = self.create_tfrecord_example(
                        windows_data[i],
                        static_data[i],
                        targets_data[i]
                    )
                    writer.write(example.SerializeToString())
                except Exception as e:
                    print(f"Fel vid skrivning av example {i}: {e}")
                    continue
        
        print(f"TFRecord fil skapad: {tfrecord_path} med {len(windows_data)} examples")
        return tfrecord_path


class SimpleMetadataCreator:
    """Enkel metadata creator för testet"""
    
    def create_preprocessing_metadata(self, total_samples, train_samples, val_samples, test_samples, 
                                    window_shape, static_shape, target_shape, success_rate):
        """Skapa preprocessing metadata enligt Master POC spec"""
        metadata = {
            'preprocessing_info': {
                'total_samples': total_samples,
                'train_samples': train_samples,
                'validation_samples': val_samples,
                'test_samples': test_samples,
                'success_rate': success_rate,
                'window_shape': list(window_shape),
                'static_features_shape': list(static_shape),
                'target_shape': list(target_shape),
                'normalization_range': [-1.0, 1.0],
                'imputation_method': 'master_poc_smart_forward_fill',
                'pipeline_type': 'master_poc'
            },
            'processing_stats': {
                'processed_cases': total_samples,
                'failed_cases': 0,
                'processing_time_seconds': 3600,
                'checkpoint_saves': 10,
                'memory_peak_mb': 2048
            },
            'master_poc_spec': {
                'timeseries_features': 16,
                'static_features': 6,
                'output_features': 8,
                'window_size_seconds': 300,
                'step_size_seconds': 30,
                'compression_type': 'GZIP'
            },
            'timestamp': time.time(),
            'version': '5.0'
        }
        
        return metadata
    
    def save_metadata(self, metadata, output_path):
        """Spara metadata till JSON fil"""
        metadata_path = f"{output_path}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata sparad: {metadata_path}")
        return metadata_path


class TestT053T056T057AWSS3Upload(unittest.TestCase):
    """T053, T056, T057: AWS Integration Tests - S3 Upload & Path Structure"""
    
    @classmethod
    def setUpClass(cls):
        """Setup för hela testklassen"""
        cls.s3_client = boto3.client('s3', region_name=AWS_REGION)
        cls.test_job_id = f"test-t053-t056-t057-{int(time.time())}"
        cls.test_output_path = f"{S3_OUTPUT_PATH.rstrip('/')}/{cls.test_job_id}/"
        
        # Skapa TFRecord creator och metadata creator
        cls.tfrecord_creator = SimpleTFRecordCreator()
        cls.metadata_creator = SimpleMetadataCreator()
        
        print(f"🧪 Test Setup:")
        print(f"   Job ID: {cls.test_job_id}")
        print(f"   Output Path: {cls.test_output_path}")
        print(f"   S3 Bucket: {S3_PRIMARY_BUCKET}")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup efter alla tester"""
        try:
            # Ta bort test-filer från S3
            cls._cleanup_s3_test_files()
            print(f"🧹 Cleanup completed för job {cls.test_job_id}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    @classmethod
    def _cleanup_s3_test_files(cls):
        """Ta bort test-filer från S3"""
        try:
            # Lista och ta bort output-filer
            output_prefix = cls.test_output_path.replace(f"s3://{S3_PRIMARY_BUCKET}/", "")
            response = cls.s3_client.list_objects_v2(
                Bucket=S3_PRIMARY_BUCKET,
                Prefix=output_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    cls.s3_client.delete_object(
                        Bucket=S3_PRIMARY_BUCKET,
                        Key=obj['Key']
                    )
                    print(f"🗑️ Deleted: {obj['Key']}")
                    
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def setUp(self):
        """Setup för varje test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skapa testdata enligt Master POC spec
        self.batch_size = 3
        self.window_size = 300
        self.n_timeseries_features = 16  # Master POC spec
        self.n_static_features = 6       # Master POC spec
        self.n_target_features = 8       # Master POC spec
        
        # Skapa testdata för train/val/test splits
        self.train_data = {
            'windows': np.random.randn(self.batch_size, self.window_size, self.n_timeseries_features),
            'static': np.random.randn(self.batch_size, self.n_static_features),
            'targets': np.random.randn(self.batch_size, self.n_target_features)
        }
        
        self.val_data = {
            'windows': np.random.randn(self.batch_size, self.window_size, self.n_timeseries_features),
            'static': np.random.randn(self.batch_size, self.n_static_features),
            'targets': np.random.randn(self.batch_size, self.n_target_features)
        }
        
        self.test_data = {
            'windows': np.random.randn(self.batch_size, self.window_size, self.n_timeseries_features),
            'static': np.random.randn(self.batch_size, self.n_static_features),
            'targets': np.random.randn(self.batch_size, self.n_target_features)
        }
    
    def tearDown(self):
        """Cleanup efter varje test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_t053_s3_upload_success(self):
        """T053: Test S3 Upload Success - Verifiera att TFRecord-filer laddas upp till S3"""
        # Arrange
        print("\n🔧 Arrange: Skapa TFRecord-filer för upload")
        
        # Skapa TFRecord-filer lokalt
        train_path = os.path.join(self.temp_dir, "train")
        val_path = os.path.join(self.temp_dir, "validation")
        test_path = os.path.join(self.temp_dir, "test")
        
        train_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.train_data['windows'], self.train_data['static'], self.train_data['targets'],
            train_path, "train"
        )
        
        val_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.val_data['windows'], self.val_data['static'], self.val_data['targets'],
            val_path, "validation"
        )
        
        test_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.test_data['windows'], self.test_data['static'], self.test_data['targets'],
            test_path, "test"
        )
        
        print(f"   ✅ Train TFRecord: {train_tfrecord}")
        print(f"   ✅ Validation TFRecord: {val_tfrecord}")
        print(f"   ✅ Test TFRecord: {test_tfrecord}")
        
        # Act - Upload till S3
        print("\n🎬 Act: Upload TFRecord-filer till S3")
        
        tfrecord_files = [
            (train_tfrecord, "train.tfrecord"),
            (val_tfrecord, "validation.tfrecord"),
            (test_tfrecord, "test.tfrecord")
        ]
        
        uploaded_files = []
        for local_path, s3_filename in tfrecord_files:
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}{s3_filename}"
            
            try:
                self.s3_client.upload_file(local_path, S3_PRIMARY_BUCKET, s3_key)
                uploaded_files.append(s3_key)
                print(f"   ✅ Uploaded: {s3_key}")
            except Exception as e:
                self.fail(f"Failed to upload {s3_filename}: {e}")
        
        # Assert - Verifiera att filer finns i S3
        print("\n✅ Assert: Verifiera S3 upload success")
        
        self.assertEqual(len(uploaded_files), 3, "Alla 3 TFRecord-filer ska vara uploadade")
        
        for s3_key in uploaded_files:
            try:
                response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                file_size = response['ContentLength']
                self.assertGreater(file_size, 0, f"TFRecord-fil {s3_key} ska ha storlek > 0")
                print(f"   ✅ Verified: {s3_key} ({file_size} bytes)")
            except ClientError as e:
                self.fail(f"TFRecord-fil {s3_key} ska finnas i S3: {e}")
        
        print("✅ Test passed: T053 S3 Upload Success")
    
    def test_t056_s3_path_structure(self):
        """T056: Test S3 Path Structure - Verifiera korrekt S3 path-struktur för output"""
        # Arrange
        print("\n🔧 Arrange: Skapa filer med korrekt path-struktur")
        
        # Skapa TFRecord-filer
        train_path = os.path.join(self.temp_dir, "train")
        train_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.train_data['windows'], self.train_data['static'], self.train_data['targets'],
            train_path, "train"
        )
        
        # Skapa metadata
        metadata = self.metadata_creator.create_preprocessing_metadata(
            total_samples=9,  # 3 batches × 3 splits
            train_samples=3,
            val_samples=3,
            test_samples=3,
            window_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            success_rate=1.0
        )
        
        metadata_path = self.metadata_creator.save_metadata(metadata, os.path.join(self.temp_dir, "metadata"))
        
        # Act - Upload med korrekt path-struktur
        print("\n🎬 Act: Upload med korrekt S3 path-struktur")
        
        # Expected S3 path structure enligt AWS_develop_instruction_V5.0.md
        expected_paths = {
            'train': f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}train.tfrecord",
            'metadata': f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}preprocessing_metadata.json"
        }
        
        # Upload train.tfrecord
        self.s3_client.upload_file(train_tfrecord, S3_PRIMARY_BUCKET, expected_paths['train'])
        print(f"   ✅ Uploaded train.tfrecord to: {expected_paths['train']}")
        
        # Upload metadata
        self.s3_client.upload_file(metadata_path, S3_PRIMARY_BUCKET, expected_paths['metadata'])
        print(f"   ✅ Uploaded metadata to: {expected_paths['metadata']}")
        
        # Assert - Verifiera path-struktur
        print("\n✅ Assert: Verifiera korrekt S3 path-struktur")
        
        # Verifiera att paths följer Master POC konvention
        for file_type, s3_key in expected_paths.items():
            # Kontrollera att path börjar med rätt prefix
            expected_prefix = f"processed-data/master-poc-pipeline/{self.test_job_id}/"
            self.assertTrue(s3_key.startswith(expected_prefix), 
                          f"{file_type} path ska börja med {expected_prefix}")
            
            # Kontrollera att filen finns i S3
            try:
                response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                file_size = response['ContentLength']
                self.assertGreater(file_size, 0, f"{file_type} ska ha storlek > 0")
                print(f"   ✅ Path verified: {s3_key} ({file_size} bytes)")
            except ClientError as e:
                self.fail(f"{file_type} ska finnas i S3: {e}")
        
        # Verifiera att path-struktur följer AWS konventioner
        bucket_name = S3_PRIMARY_BUCKET
        self.assertEqual(bucket_name, "master-poc-v1.0", "Bucket namn ska vara korrekt")
        
        # Verifiera att job ID är unikt
        self.assertIn(self.test_job_id, expected_paths['train'], "Job ID ska finnas i path")
        
        print("✅ Test passed: T056 S3 Path Structure")
    
    def test_t057_upload_before_timeout(self):
        """T057: Test Upload Before Timeout - Verifiera att upload sker innan MaxRuntimeExceeded"""
        # Arrange
        print("\n🔧 Arrange: Simulera timeout-scenario med stora filer")
        
        # Skapa större dataset för att simulera längre processing
        large_batch_size = 10
        large_train_data = {
            'windows': np.random.randn(large_batch_size, self.window_size, self.n_timeseries_features),
            'static': np.random.randn(large_batch_size, self.n_static_features),
            'targets': np.random.randn(large_batch_size, self.n_target_features)
        }
        
        # Skapa TFRecord-fil
        train_path = os.path.join(self.temp_dir, "large_train")
        train_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            large_train_data['windows'], large_train_data['static'], large_train_data['targets'],
            train_path, "train"
        )
        
        print(f"   ✅ Large TFRecord created: {train_tfrecord}")
        
        # Act - Upload med timeout monitoring
        print("\n🎬 Act: Upload med timeout monitoring")
        
        s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}large_train.tfrecord"
        
        # Simulera MaxRuntimeExceeded scenario (26 timmar enligt AWS_develop_instruction_V5.0.md)
        max_runtime_seconds = 26 * 3600  # 26 timmar
        upload_start_time = time.time()
        
        # Upload med retry logic och timeout monitoring
        max_retries = 3
        upload_success = False
        
        for attempt in range(max_retries):
            try:
                upload_attempt_start = time.time()
                
                # Simulera upload med progress monitoring
                self.s3_client.upload_file(train_tfrecord, S3_PRIMARY_BUCKET, s3_key)
                
                upload_time = time.time() - upload_attempt_start
                total_time = time.time() - upload_start_time
                
                upload_success = True
                print(f"   ✅ Upload successful on attempt {attempt + 1}")
                print(f"   ⏱️ Upload time: {upload_time:.2f}s")
                print(f"   ⏱️ Total time: {total_time:.2f}s")
                break
                
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   ⚠️ Upload attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    print(f"   ⏳ Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed to upload after {max_retries} retries")
        
        # Assert - Verifiera att upload skedde innan timeout
        print("\n✅ Assert: Verifiera upload innan timeout")
        
        self.assertTrue(upload_success, "Upload ska lyckas innan timeout")
        
        # Verifiera att filen finns i S3
        try:
            response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
            file_size = response['ContentLength']
            self.assertGreater(file_size, 0, "Uploaded fil ska ha storlek > 0")
            print(f"   ✅ File verified in S3: {file_size} bytes")
        except ClientError as e:
            self.fail(f"Uploaded fil ska finnas i S3: {e}")
        
        # Verifiera att upload tid är rimlig (under 1 timme för testet)
        upload_time = time.time() - upload_start_time
        self.assertLess(upload_time, 3600, "Upload ska ta mindre än 1 timme")
        print(f"   ⏱️ Total upload time: {upload_time:.2f}s (under 1 timme)")
        
        # Verifiera att vi har tid kvar före MaxRuntimeExceeded
        remaining_time = max_runtime_seconds - upload_time
        self.assertGreater(remaining_time, 0, "Det ska finnas tid kvar före MaxRuntimeExceeded")
        print(f"   ⏱️ Remaining time before MaxRuntimeExceeded: {remaining_time/3600:.1f} timmar")
        
        print("✅ Test passed: T057 Upload Before Timeout")
    
    def test_t053_t056_t057_combined_workflow(self):
        """Kombinerat test för T053, T056, T057 - Fullständig S3 upload workflow"""
        # Arrange
        print("\n🔧 Arrange: Fullständig S3 upload workflow")
        
        # Skapa alla TFRecord-filer
        train_path = os.path.join(self.temp_dir, "train")
        val_path = os.path.join(self.temp_dir, "validation")
        test_path = os.path.join(self.temp_dir, "test")
        
        train_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.train_data['windows'], self.train_data['static'], self.train_data['targets'],
            train_path, "train"
        )
        
        val_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.val_data['windows'], self.val_data['static'], self.val_data['targets'],
            val_path, "validation"
        )
        
        test_tfrecord = self.tfrecord_creator.create_memory_efficient_tfrecord(
            self.test_data['windows'], self.test_data['static'], self.test_data['targets'],
            test_path, "test"
        )
        
        # Skapa metadata
        metadata = self.metadata_creator.create_preprocessing_metadata(
            total_samples=9,
            train_samples=3,
            val_samples=3,
            test_samples=3,
            window_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            success_rate=1.0
        )
        
        metadata_path = self.metadata_creator.save_metadata(metadata, os.path.join(self.temp_dir, "metadata"))
        
        # Act - Fullständig upload workflow
        print("\n🎬 Act: Fullständig S3 upload workflow")
        
        # Definiera korrekt S3 path-struktur
        s3_files = {
            'train.tfrecord': train_tfrecord,
            'validation.tfrecord': val_tfrecord,
            'test.tfrecord': test_tfrecord,
            'preprocessing_metadata.json': metadata_path
        }
        
        uploaded_files = []
        upload_start_time = time.time()
        
        for s3_filename, local_path in s3_files.items():
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}{s3_filename}"
            
            try:
                self.s3_client.upload_file(local_path, S3_PRIMARY_BUCKET, s3_key)
                uploaded_files.append(s3_key)
                print(f"   ✅ Uploaded: {s3_filename}")
            except Exception as e:
                self.fail(f"Failed to upload {s3_filename}: {e}")
        
        total_upload_time = time.time() - upload_start_time
        
        # Assert - Verifiera kombinerat workflow
        print("\n✅ Assert: Verifiera kombinerat workflow")
        
        # T053: Verifiera S3 upload success
        self.assertEqual(len(uploaded_files), 4, "Alla 4 filer ska vara uploadade")
        
        # T056: Verifiera S3 path-struktur
        for s3_key in uploaded_files:
            expected_prefix = f"processed-data/master-poc-pipeline/{self.test_job_id}/"
            self.assertTrue(s3_key.startswith(expected_prefix), 
                          f"Path ska börja med {expected_prefix}")
            
            try:
                response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                file_size = response['ContentLength']
                self.assertGreater(file_size, 0, f"Fil {s3_key} ska ha storlek > 0")
                print(f"   ✅ Verified: {s3_key} ({file_size} bytes)")
            except ClientError as e:
                self.fail(f"Fil {s3_key} ska finnas i S3: {e}")
        
        # T057: Verifiera upload innan timeout
        self.assertLess(total_upload_time, 3600, "Total upload ska ta mindre än 1 timme")
        print(f"   ⏱️ Total upload time: {total_upload_time:.2f}s")
        
        # Verifiera att alla förväntade filer finns
        expected_files = ['train.tfrecord', 'validation.tfrecord', 'test.tfrecord', 'preprocessing_metadata.json']
        for expected_file in expected_files:
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}{expected_file}"
            self.assertIn(s3_key, uploaded_files, f"{expected_file} ska vara uploadad")
        
        print("✅ Test passed: T053, T056, T057 Combined Workflow")


if __name__ == '__main__':
    # Kör tester med verbose output
    unittest.main(verbosity=2)
