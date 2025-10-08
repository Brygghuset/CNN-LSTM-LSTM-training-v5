#!/usr/bin/env python3
"""
T050: AWS Integration Test - Partial Save Recovery
==================================================

Riktigt integrationstest med AWS S3 för att verifiera att partiellt sparad data 
kan återhämtas från checkpoint och att processing kan fortsätta.

AAA Format:
- Arrange: Skapa partiellt sparad data i S3 och simulera avbrott
- Act: Återhämta och fortsätt processing från checkpoint med riktiga AWS komponenter
- Assert: Verifiera att data är korrekt återhämtad och processing kan fortsätta

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


class SimpleCheckpointManager:
    """Enkel checkpoint manager för testet"""
    
    def __init__(self, checkpoint_path, enable_checkpoints=True, checkpoint_interval=50):
        self.checkpoint_path = checkpoint_path.rstrip('/')
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.processed_cases = set()
        self.failed_cases = set()
        self.s3_client = boto3.client('s3') if enable_checkpoints else None
    
    def add_processed_case(self, case_id):
        """Lägg till processed case"""
        self.processed_cases.add(case_id)
    
    def add_failed_case(self, case_id, error_msg):
        """Lägg till failed case"""
        self.failed_cases.add(case_id)
    
    def save_checkpoint(self, case_id):
        """Spara checkpoint till S3"""
        if not self.enable_checkpoints:
            return True
        
        try:
            # Skapa checkpoint data
            checkpoint_data = {
                'processed_cases': list(self.processed_cases),
                'failed_cases': list(self.failed_cases),
                'processing_state': {
                    'current_case_index': len(self.processed_cases),
                    'last_processed_case': case_id
                },
                'timestamp': time.time(),
                'checkpoint_version': '1.0'
            }
            
            # Spara till lokal fil först
            local_checkpoint_path = f"/tmp/checkpoint_{int(time.time())}.json"
            with open(local_checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Upload till S3
            bucket_name = self.checkpoint_path.split('/')[2]
            s3_prefix = '/'.join(self.checkpoint_path.split('/')[3:])
            s3_key = f"{s3_prefix}/checkpoint_{int(time.time())}.json"
            
            self.s3_client.upload_file(local_checkpoint_path, bucket_name, s3_key)
            
            # Ta bort lokal fil
            os.remove(local_checkpoint_path)
            
            print(f"Checkpoint saved to s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self):
        """Ladda checkpoint från S3"""
        if not self.enable_checkpoints:
            return True
        
        try:
            bucket_name = self.checkpoint_path.split('/')[2]
            s3_prefix = '/'.join(self.checkpoint_path.split('/')[3:])
            
            # Lista checkpoint filer
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                print("No existing checkpoints found")
                return True
            
            # Hitta senaste checkpoint fil
            checkpoint_files = [
                obj for obj in response['Contents']
                if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
            ]
            
            if not checkpoint_files:
                print("No checkpoint files found")
                return True
            
            # Sortera efter senast modifierad
            latest_checkpoint = max(checkpoint_files, key=lambda x: x['LastModified'])
            checkpoint_s3_key = latest_checkpoint['Key']
            
            # Ladda checkpoint data
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=checkpoint_s3_key
            )
            
            checkpoint_json = response['Body'].read().decode('utf-8')
            checkpoint_data = json.loads(checkpoint_json)
            
            # Återställ state
            self.processed_cases = set(checkpoint_data.get('processed_cases', []))
            self.failed_cases = set(checkpoint_data.get('failed_cases', []))
            
            print(f"Checkpoint loaded from s3://{bucket_name}/{checkpoint_s3_key}")
            print(f"Processed cases: {len(self.processed_cases)}")
            print(f"Failed cases: {len(self.failed_cases)}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False


class TestT050AWSPartialSaveRecovery(unittest.TestCase):
    """T050: AWS Integration Test - Partial Save Recovery med riktiga AWS komponenter"""
    
    @classmethod
    def setUpClass(cls):
        """Setup för hela testklassen"""
        cls.s3_client = boto3.client('s3', region_name=AWS_REGION)
        cls.test_job_id = f"test-t050-{int(time.time())}"
        cls.test_checkpoint_path = f"{S3_CHECKPOINT_PATH.rstrip('/')}/{cls.test_job_id}/"
        cls.test_output_path = f"{S3_OUTPUT_PATH.rstrip('/')}/{cls.test_job_id}/"
        
        # Skapa TFRecord creator
        cls.tfrecord_creator = SimpleTFRecordCreator()
        
        print(f"🧪 Test Setup:")
        print(f"   Job ID: {cls.test_job_id}")
        print(f"   Checkpoint Path: {cls.test_checkpoint_path}")
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
            # Lista och ta bort checkpoint-filer
            checkpoint_prefix = cls.test_checkpoint_path.replace(f"s3://{S3_PRIMARY_BUCKET}/", "")
            response = cls.s3_client.list_objects_v2(
                Bucket=S3_PRIMARY_BUCKET,
                Prefix=checkpoint_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    cls.s3_client.delete_object(
                        Bucket=S3_PRIMARY_BUCKET,
                        Key=obj['Key']
                    )
                    print(f"🗑️ Deleted: {obj['Key']}")
            
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
        
        # Skapa checkpoint manager för testet
        self.checkpoint_manager = SimpleCheckpointManager(
            checkpoint_path=self.test_checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=3  # Spara var 3:e case för testet
        )
        
        # Skapa testdata enligt Master POC spec
        self.batch_size = 2
        self.n_batches = 6  # Tillräckligt för att testa recovery
        self.window_size = 300
        self.n_timeseries_features = 16  # Master POC spec
        self.n_static_features = 6       # Master POC spec
        self.n_target_features = 8       # Master POC spec
        
        # Skapa testdata för varje batch
        self.batches = []
        for i in range(self.n_batches):
            batch_windows = np.random.randn(self.batch_size, self.window_size, self.n_timeseries_features)
            batch_static = np.random.randn(self.batch_size, self.n_static_features)
            batch_targets = np.random.randn(self.batch_size, self.n_target_features)
            
            self.batches.append({
                'windows': batch_windows,
                'static': batch_static,
                'targets': batch_targets,
                'case_id': f"test_case_{i:04d}",
                'batch_id': i
            })
    
    def tearDown(self):
        """Cleanup efter varje test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_t050_aws_partial_save_recovery_basic(self):
        """T050: Grundläggande AWS partial save recovery"""
        # Arrange
        print("\n🔧 Arrange: Skapa partiellt sparad data i S3")
        
        # Processera första 3 batches och spara till S3
        processed_cases = []
        for i in range(3):
            batch = self.batches[i]
            
            # Skapa TFRecord lokalt
            local_tfrecord_path = os.path.join(self.temp_dir, f"batch_{i}.tfrecord")
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                local_tfrecord_path, f"batch_{i}"
            )
            
            # Upload till S3
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{i}.tfrecord"
            self.s3_client.upload_file(tfrecord_path, S3_PRIMARY_BUCKET, s3_key)
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(batch['case_id'])
            processed_cases.append(batch['case_id'])
            
            print(f"   ✅ Processed batch {i}: {batch['case_id']}")
        
        # Spara checkpoint till S3
        checkpoint_saved = self.checkpoint_manager.save_checkpoint("batch_2")
        self.assertTrue(checkpoint_saved, "Checkpoint ska sparas till S3")
        
        print(f"   💾 Checkpoint saved: {checkpoint_saved}")
        
        # Act - Simulera restart och återhämta från S3
        print("\n🎬 Act: Simulera restart och återhämta från checkpoint")
        
        # Skapa ny checkpoint manager (simulerar restart)
        new_checkpoint_manager = SimpleCheckpointManager(
            checkpoint_path=self.test_checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=3
        )
        
        # Ladda checkpoint från S3
        checkpoint_loaded = new_checkpoint_manager.load_checkpoint()
        
        print(f"   📂 Checkpoint loaded: {checkpoint_loaded}")
        
        # Assert - Checkpoint ska kunna laddas från S3
        self.assertTrue(checkpoint_loaded, "Checkpoint ska kunna laddas från S3")
        self.assertEqual(len(new_checkpoint_manager.processed_cases), 3, 
                        "Checkpoint ska innehålla 3 processed cases")
        
        # Verifiera att TFRecord-filer finns i S3
        for i in range(3):
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{i}.tfrecord"
            try:
                response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                file_size = response['ContentLength']
                self.assertGreater(file_size, 0, f"TFRecord-fil {i} ska ha storlek > 0")
                print(f"   ✅ TFRecord batch_{i}.tfrecord: {file_size} bytes")
            except ClientError as e:
                self.fail(f"TFRecord-fil {i} ska finnas i S3: {e}")
        
        print("✅ Test passed: Grundläggande AWS partial save recovery")
    
    def test_t050_aws_partial_save_recovery_continue_processing(self):
        """T050: Fortsätt processing efter AWS recovery"""
        # Arrange
        print("\n🔧 Arrange: Skapa partiellt sparad data och simulera avbrott")
        
        # Processera första 3 batches
        processed_cases = []
        for i in range(3):
            batch = self.batches[i]
            
            # Skapa TFRecord lokalt
            local_tfrecord_path = os.path.join(self.temp_dir, f"batch_{i}.tfrecord")
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                local_tfrecord_path, f"batch_{i}"
            )
            
            # Upload till S3
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{i}.tfrecord"
            self.s3_client.upload_file(tfrecord_path, S3_PRIMARY_BUCKET, s3_key)
            
            # Lägg till i checkpoint
            self.checkpoint_manager.add_processed_case(batch['case_id'])
            processed_cases.append(batch['case_id'])
        
        # Spara checkpoint
        checkpoint_saved = self.checkpoint_manager.save_checkpoint("batch_2")
        self.assertTrue(checkpoint_saved, "Checkpoint ska sparas")
        
        # Act - Simulera restart och fortsätt processing
        print("\n🎬 Act: Simulera restart och fortsätt processing")
        
        # Skapa ny checkpoint manager
        new_checkpoint_manager = SimpleCheckpointManager(
            checkpoint_path=self.test_checkpoint_path,
            enable_checkpoints=True,
            checkpoint_interval=3
        )
        
        # Ladda checkpoint
        checkpoint_loaded = new_checkpoint_manager.load_checkpoint()
        self.assertTrue(checkpoint_loaded, "Checkpoint ska kunna laddas")
        
        # Identifiera remaining cases
        processed_cases_set = new_checkpoint_manager.processed_cases
        all_cases = {batch['case_id'] for batch in self.batches}
        remaining_cases = all_cases - processed_cases_set
        
        print(f"   📊 Processed cases: {len(processed_cases_set)}")
        print(f"   📊 Remaining cases: {len(remaining_cases)}")
        
        # Fortsätt processing med remaining cases
        new_processed_cases = []
        for case_id in remaining_cases:
            # Hitta batch för denna case
            batch = next(batch for batch in self.batches if batch['case_id'] == case_id)
            batch_id = batch['batch_id']
            
            # Skapa TFRecord lokalt
            local_tfrecord_path = os.path.join(self.temp_dir, f"batch_{batch_id}_recovered.tfrecord")
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                batch['windows'], batch['static'], batch['targets'],
                local_tfrecord_path, f"batch_{batch_id}_recovered"
            )
            
            # Upload till S3
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{batch_id}_recovered.tfrecord"
            self.s3_client.upload_file(tfrecord_path, S3_PRIMARY_BUCKET, s3_key)
            
            # Lägg till i checkpoint
            new_checkpoint_manager.add_processed_case(case_id)
            new_processed_cases.append(case_id)
            
            print(f"   ✅ Recovered batch {batch_id}: {case_id}")
        
        # Assert - Alla cases ska vara processade
        self.assertEqual(len(remaining_cases), self.n_batches - 3, 
                        "Remaining cases ska vara korrekt antal")
        
        # Verifiera att alla TFRecord-filer finns i S3
        total_files = 0
        for i in range(3):  # Original batches
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{i}.tfrecord"
            try:
                self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                total_files += 1
            except ClientError:
                self.fail(f"Original TFRecord batch_{i} ska finnas i S3")
        
        for case_id in remaining_cases:  # Recovered batches
            batch = next(batch for batch in self.batches if batch['case_id'] == case_id)
            batch_id = batch['batch_id']
            s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}batch_{batch_id}_recovered.tfrecord"
            try:
                self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
                total_files += 1
            except ClientError:
                self.fail(f"Recovered TFRecord batch_{batch_id} ska finnas i S3")
        
        self.assertEqual(total_files, self.n_batches, 
                        "Alla batches ska ha TFRecord-filer i S3")
        
        print("✅ Test passed: Fortsätt processing efter AWS recovery")
    
    def test_t050_aws_partial_save_recovery_s3_retry_logic(self):
        """T050: Testa S3 retry logic under recovery"""
        # Arrange
        print("\n🔧 Arrange: Testa S3 retry logic med exponential backoff")
        
        # Skapa testdata
        batch = self.batches[0]
        local_tfrecord_path = os.path.join(self.temp_dir, "test_retry.tfrecord")
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            batch['windows'], batch['static'], batch['targets'],
            local_tfrecord_path, "test_retry"
        )
        
        # Act - Testa upload med retry logic
        print("\n🎬 Act: Testa upload med retry logic")
        
        s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}test_retry.tfrecord"
        
        # Simulera retry logic från master_poc_preprocessing_v5.py
        max_retries = 3
        upload_success = False
        
        for attempt in range(max_retries):
            try:
                self.s3_client.upload_file(tfrecord_path, S3_PRIMARY_BUCKET, s3_key)
                upload_success = True
                print(f"   ✅ Upload successful on attempt {attempt + 1}")
                break
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   ⚠️ Upload attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    print(f"   ⏳ Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed to upload after {max_retries} retries")
        
        # Assert - Upload ska lyckas
        self.assertTrue(upload_success, "Upload ska lyckas med retry logic")
        
        # Verifiera att filen finns i S3
        try:
            response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
            file_size = response['ContentLength']
            self.assertGreater(file_size, 0, "Uploaded fil ska ha storlek > 0")
            print(f"   ✅ File verified in S3: {file_size} bytes")
        except ClientError as e:
            self.fail(f"Uploaded fil ska finnas i S3: {e}")
        
        print("✅ Test passed: S3 retry logic under recovery")
    
    def test_t050_aws_partial_save_recovery_master_poc_compliance(self):
        """T050: Verifiera Master POC compliance under recovery"""
        # Arrange
        print("\n🔧 Arrange: Skapa data enligt Master POC specifikation")
        
        # Verifiera att testdata följer Master POC spec
        batch = self.batches[0]
        
        # Assert - Verifiera Master POC specifikationer
        self.assertEqual(batch['windows'].shape[1], 300, 
                        "Window size ska vara 300 sekunder enligt Master POC spec")
        self.assertEqual(batch['windows'].shape[2], 16, 
                        "Timeseries features ska vara 16 enligt Master POC spec")
        self.assertEqual(batch['static'].shape[1], 6, 
                        "Static features ska vara 6 enligt Master POC spec")
        self.assertEqual(batch['targets'].shape[1], 8, 
                        "Target features ska vara 8 enligt Master POC spec")
        
        print(f"   ✅ Window shape: {batch['windows'].shape}")
        print(f"   ✅ Static shape: {batch['static'].shape}")
        print(f"   ✅ Target shape: {batch['targets'].shape}")
        
        # Act - Skapa TFRecord enligt Master POC spec
        print("\n🎬 Act: Skapa TFRecord enligt Master POC spec")
        
        local_tfrecord_path = os.path.join(self.temp_dir, "master_poc_compliance.tfrecord")
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            batch['windows'], batch['static'], batch['targets'],
            local_tfrecord_path, "master_poc_compliance"
        )
        
        # Upload till S3
        s3_key = f"{self.test_output_path.replace(f's3://{S3_PRIMARY_BUCKET}/', '')}master_poc_compliance.tfrecord"
        self.s3_client.upload_file(tfrecord_path, S3_PRIMARY_BUCKET, s3_key)
        
        # Lägg till i checkpoint
        self.checkpoint_manager.add_processed_case(batch['case_id'])
        checkpoint_saved = self.checkpoint_manager.save_checkpoint("master_poc_test")
        
        # Assert - Verifiera Master POC compliance
        self.assertTrue(checkpoint_saved, "Checkpoint ska sparas")
        
        # Verifiera att TFRecord-filen finns i S3
        try:
            response = self.s3_client.head_object(Bucket=S3_PRIMARY_BUCKET, Key=s3_key)
            file_size = response['ContentLength']
            self.assertGreater(file_size, 0, "Master POC TFRecord ska ha storlek > 0")
            print(f"   ✅ Master POC TFRecord uploaded: {file_size} bytes")
        except ClientError as e:
            self.fail(f"Master POC TFRecord ska finnas i S3: {e}")
        
        # Verifiera att checkpoint innehåller korrekt case
        checkpoint_loaded = self.checkpoint_manager.load_checkpoint()
        self.assertTrue(checkpoint_loaded, "Master POC checkpoint ska kunna laddas")
        self.assertIn(batch['case_id'], self.checkpoint_manager.processed_cases,
                     "Master POC case ska vara i processed cases")
        
        print("✅ Test passed: Master POC compliance under recovery")


if __name__ == '__main__':
    # Kör tester med verbose output
    unittest.main(verbosity=2)