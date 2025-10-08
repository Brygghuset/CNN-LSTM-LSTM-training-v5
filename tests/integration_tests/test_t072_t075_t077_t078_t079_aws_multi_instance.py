#!/usr/bin/env python3
"""
T072, T075, T077, T078, T079: AWS Integration Tests - Multi-Instance & Distributed Checkpoints
=============================================================================================

Riktiga integrationstester med AWS S3 för att verifiera:
- T072: No Case Overlap - Verifiera att instanser inte processar samma cases
- T075: Distribution with 6 Instances - Verifiera distribution med 6 instanser
- T077: Distributed Checkpoint Creation - Verifiera att varje instans skapar sina checkpoints
- T078: Checkpoint Path Conflicts - Verifiera att instanser inte skriver till samma checkpoint
- T079: Instance-Specific Recovery - Verifiera att instanser kan återhämta sina egna checkpoints

AAA Format:
- Arrange: Skapa multi-instance scenario och förbered distributed checkpoint system
- Act: Simulera flera instanser med case distribution och checkpoint creation
- Assert: Verifiera att ingen case overlap finns och att checkpoints är instance-specific

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


class MultiInstanceCheckpointManager:
    """Multi-instance checkpoint manager för testet"""
    
    def __init__(self, checkpoint_path, host_info, enable_checkpoints=True, checkpoint_interval=50):
        self.checkpoint_path = checkpoint_path.rstrip('/')
        self.host_info = host_info
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.processed_cases = set()
        self.failed_cases = set()
        self.current_case_index = 0
        self.s3_client = boto3.client('s3') if enable_checkpoints else None
        
        # Skapa instance-specific checkpoint path
        self.instance_checkpoint_path = self._create_instance_checkpoint_path()
    
    def _create_instance_checkpoint_path(self):
        """Skapa instance-specific checkpoint path enligt Master POC spec"""
        current_host = self.host_info['current_host']
        base_path = self.checkpoint_path.rstrip('/')
        instance_path = f"{base_path}/{current_host}"
        print(f"Instance checkpoint path: {instance_path}")
        return instance_path
    
    def add_processed_case(self, case_id):
        """Lägg till processed case"""
        self.processed_cases.add(case_id)
        self.current_case_index += 1
    
    def add_failed_case(self, case_id, error_msg):
        """Lägg till failed case"""
        self.failed_cases.add(case_id)
        self.current_case_index += 1
    
    def save_checkpoint(self, case_id):
        """Spara checkpoint till S3 med instance-specific path"""
        if not self.enable_checkpoints:
            return True
        
        try:
            # Skapa checkpoint data med instance info
            checkpoint_data = {
                'processed_cases': list(self.processed_cases),
                'failed_cases': list(self.failed_cases),
                'processing_state': {
                    'current_case_index': self.current_case_index,
                    'last_processed_case': case_id
                },
                'instance_info': {
                    'current_host': self.host_info['current_host'],
                    'host_index': self.host_info['host_index'],
                    'total_hosts': self.host_info['total_hosts'],
                    'all_hosts': self.host_info['all_hosts']
                },
                'timestamp': time.time(),
                'checkpoint_version': '1.0'
            }
            
            # Spara till lokal fil först
            local_checkpoint_path = f"/tmp/checkpoint_{self.host_info['current_host']}_{int(time.time())}.json"
            with open(local_checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Upload till S3 med instance-specific path
            bucket_name = self.instance_checkpoint_path.split('/')[2]
            s3_prefix = '/'.join(self.instance_checkpoint_path.split('/')[3:])
            s3_key = f"{s3_prefix}/checkpoint_{int(time.time())}.json"
            
            self.s3_client.upload_file(local_checkpoint_path, bucket_name, s3_key)
            
            # Ta bort lokal fil
            os.remove(local_checkpoint_path)
            
            print(f"Instance checkpoint saved to s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"Failed to save instance checkpoint: {e}")
            return False
    
    def load_checkpoint(self):
        """Ladda checkpoint från S3 med instance-specific path"""
        if not self.enable_checkpoints:
            return True
        
        try:
            bucket_name = self.instance_checkpoint_path.split('/')[2]
            s3_prefix = '/'.join(self.instance_checkpoint_path.split('/')[3:])
            
            # Lista checkpoint filer för denna instans
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                print(f"No existing checkpoints found for instance {self.host_info['current_host']}")
                return True
            
            # Hitta senaste checkpoint fil för denna instans
            checkpoint_files = [
                obj for obj in response['Contents']
                if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
            ]
            
            if not checkpoint_files:
                print(f"No checkpoint files found for instance {self.host_info['current_host']}")
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
            
            # Verifiera att checkpoint tillhör rätt instans
            instance_info = checkpoint_data.get('instance_info', {})
            if instance_info.get('current_host') != self.host_info['current_host']:
                print(f"Checkpoint belongs to different instance: {instance_info.get('current_host')}")
                return False
            
            # Återställ state
            self.processed_cases = set(checkpoint_data.get('processed_cases', []))
            self.failed_cases = set(checkpoint_data.get('failed_cases', []))
            self.current_case_index = checkpoint_data.get('processing_state', {}).get('current_case_index', 0)
            
            print(f"Instance checkpoint loaded from s3://{bucket_name}/{checkpoint_s3_key}")
            print(f"Instance {self.host_info['current_host']}: Processed cases: {len(self.processed_cases)}")
            print(f"Instance {self.host_info['current_host']}: Failed cases: {len(self.failed_cases)}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load instance checkpoint: {e}")
            return False
    
    def get_remaining_cases(self, all_cases):
        """Hämta återstående cases efter checkpoint resume"""
        processed_and_failed = self.processed_cases | self.failed_cases
        return [case for case in all_cases if case not in processed_and_failed]
    
    def get_processing_stats(self):
        """Hämta processing statistik för denna instans"""
        return {
            'instance_host': self.host_info['current_host'],
            'instance_index': self.host_info['host_index'],
            'total_cases': len(self.processed_cases) + len(self.failed_cases),
            'processed_cases': len(self.processed_cases),
            'failed_cases': len(self.failed_cases),
            'current_case_index': self.current_case_index,
            'checkpoint_interval': self.checkpoint_interval,
            'enable_checkpoints': self.enable_checkpoints
        }


class CaseDistributor:
    """Case distributor för multi-instance testing"""
    
    @staticmethod
    def distribute_cases_across_instances(case_ids, host_info):
        """
        Dela upp cases mellan instanser för att undvika dubbel processing.
        Använder modulo för jämn fördelning enligt Master POC spec.
        """
        host_index = host_info['host_index']
        total_hosts = host_info['total_hosts']
        
        instance_cases = [
            case_id for i, case_id in enumerate(case_ids)
            if i % total_hosts == host_index
        ]
        
        print(f"Instance {host_info['current_host']} (index {host_index}): {len(instance_cases)} cases")
        return instance_cases
    
    @staticmethod
    def verify_no_case_overlap(all_instances_cases):
        """Verifiera att ingen case overlap finns mellan instanser"""
        all_cases = set()
        overlaps = []
        
        for instance_host, cases in all_instances_cases.items():
            instance_cases = set(cases)
            overlap = all_cases & instance_cases
            if overlap:
                overlaps.append((instance_host, overlap))
            all_cases.update(instance_cases)
        
        return overlaps, all_cases
    
    @staticmethod
    def verify_even_distribution(all_instances_cases):
        """Verifiera jämn fördelning av cases mellan instanser"""
        case_counts = [len(cases) for cases in all_instances_cases.values()]
        
        if not case_counts:
            return True, 0
        
        min_cases = min(case_counts)
        max_cases = max(case_counts)
        
        # Tillåt max 1 case skillnad för jämn fördelning
        is_even = (max_cases - min_cases) <= 1
        
        return is_even, max_cases - min_cases


class TestT072T075T077T078T079AWSMultiInstance(unittest.TestCase):
    """T072, T075, T077, T078, T079: AWS Integration Tests - Multi-Instance & Distributed Checkpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Setup för hela testklassen"""
        cls.s3_client = boto3.client('s3', region_name=AWS_REGION)
        cls.test_job_id = f"test-t072-t075-t077-t078-t079-{int(time.time())}"
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
            # Lista och ta bort checkpoint-filer för alla instanser
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
                    
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def setUp(self):
        """Setup för varje test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skapa testdata enligt Master POC spec
        self.batch_size = 2
        self.n_cases = 24  # Delbart med 6 instanser
        self.window_size = 300
        self.n_timeseries_features = 16  # Master POC spec
        self.n_static_features = 6       # Master POC spec
        self.n_target_features = 8        # Master POC spec
        
        # Skapa testdata för alla cases
        self.all_cases = [f"case_{i:04d}" for i in range(self.n_cases)]
        
        # Skapa testdata för varje case
        self.cases_data = {}
        for case_id in self.all_cases:
            self.cases_data[case_id] = {
                'windows': np.random.randn(self.batch_size, self.window_size, self.n_timeseries_features),
                'static': np.random.randn(self.batch_size, self.n_static_features),
                'targets': np.random.randn(self.batch_size, self.n_target_features)
            }
    
    def tearDown(self):
        """Cleanup efter varje test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_t072_no_case_overlap(self):
        """T072: Test No Case Overlap - Verifiera att instanser inte processar samma cases"""
        # Arrange
        print("\n🔧 Arrange: Skapa multi-instance scenario")
        
        # Skapa 3 instanser för testet
        instances = [
            {'current_host': 'algo-1', 'host_index': 0, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-2', 'host_index': 1, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-3', 'host_index': 2, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']}
        ]
        
        # Act - Distribuera cases mellan instanser
        print("\n🎬 Act: Distribuera cases mellan instanser")
        
        all_instances_cases = {}
        for instance in instances:
            instance_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            all_instances_cases[instance['current_host']] = instance_cases
            
            print(f"   📋 Instance {instance['current_host']}: {len(instance_cases)} cases")
            print(f"      Cases: {instance_cases[:5]}{'...' if len(instance_cases) > 5 else ''}")
        
        # Assert - Verifiera att ingen case overlap finns
        print("\n✅ Assert: Verifiera att ingen case overlap finns")
        
        overlaps, all_distributed_cases = CaseDistributor.verify_no_case_overlap(all_instances_cases)
        
        self.assertEqual(len(overlaps), 0, f"Case overlap hittades: {overlaps}")
        
        # Verifiera att alla cases är distribuerade
        self.assertEqual(len(all_distributed_cases), self.n_cases, 
                        "Alla cases ska vara distribuerade")
        
        # Verifiera att alla ursprungliga cases finns i distribuerade cases
        self.assertEqual(set(all_distributed_cases), set(self.all_cases),
                        "Distribuerade cases ska matcha ursprungliga cases")
        
        # Verifiera jämn fördelning
        is_even, max_diff = CaseDistributor.verify_even_distribution(all_instances_cases)
        self.assertTrue(is_even, f"Case distribution är inte jämn, max skillnad: {max_diff}")
        
        print(f"   ✅ No case overlap verified")
        print(f"   ✅ All {self.n_cases} cases distributed")
        print(f"   ✅ Even distribution verified (max diff: {max_diff})")
        
        print("✅ Test passed: T072 No Case Overlap")
    
    def test_t075_distribution_with_6_instances(self):
        """T075: Test Distribution with 6 Instances - Verifiera distribution med 6 instanser"""
        # Arrange
        print("\n🔧 Arrange: Skapa 6-instance scenario")
        
        # Skapa 6 instanser för testet
        instances = [
            {'current_host': f'algo-{i}', 'host_index': i, 'total_hosts': 6, 
             'all_hosts': [f'algo-{j}' for j in range(1, 7)]}
            for i in range(6)
        ]
        
        # Act - Distribuera cases mellan 6 instanser
        print("\n🎬 Act: Distribuera cases mellan 6 instanser")
        
        all_instances_cases = {}
        for instance in instances:
            instance_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            all_instances_cases[instance['current_host']] = instance_cases
            
            print(f"   📋 Instance {instance['current_host']}: {len(instance_cases)} cases")
        
        # Assert - Verifiera distribution med 6 instanser
        print("\n✅ Assert: Verifiera distribution med 6 instanser")
        
        # Verifiera att alla 6 instanser fick cases
        self.assertEqual(len(all_instances_cases), 6, "Alla 6 instanser ska ha cases")
        
        # Verifiera att ingen case overlap finns
        overlaps, all_distributed_cases = CaseDistributor.verify_no_case_overlap(all_instances_cases)
        self.assertEqual(len(overlaps), 0, f"Case overlap hittades: {overlaps}")
        
        # Verifiera att alla cases är distribuerade
        self.assertEqual(len(all_distributed_cases), self.n_cases, 
                        "Alla cases ska vara distribuerade")
        
        # Verifiera jämn fördelning (24 cases / 6 instanser = 4 cases per instans)
        is_even, max_diff = CaseDistributor.verify_even_distribution(all_instances_cases)
        self.assertTrue(is_even, f"Case distribution är inte jämn, max skillnad: {max_diff}")
        
        # Verifiera att varje instans fick exakt 4 cases
        for instance_host, cases in all_instances_cases.items():
            self.assertEqual(len(cases), 4, f"Instance {instance_host} ska ha exakt 4 cases")
        
        print(f"   ✅ All 6 instances have cases")
        print(f"   ✅ No case overlap verified")
        print(f"   ✅ Even distribution verified (4 cases per instance)")
        
        print("✅ Test passed: T075 Distribution with 6 Instances")
    
    def test_t077_distributed_checkpoint_creation(self):
        """T077: Test Distributed Checkpoint Creation - Verifiera att varje instans skapar sina checkpoints"""
        # Arrange
        print("\n🔧 Arrange: Skapa distributed checkpoint scenario")
        
        # Skapa 3 instanser med checkpoint managers
        instances = [
            {'current_host': 'algo-1', 'host_index': 0, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-2', 'host_index': 1, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-3', 'host_index': 2, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']}
        ]
        
        checkpoint_managers = {}
        for instance in instances:
            checkpoint_managers[instance['current_host']] = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=2
            )
        
        # Act - Skapa checkpoints för varje instans
        print("\n🎬 Act: Skapa checkpoints för varje instans")
        
        # Distribuera cases och processera
        for instance in instances:
            instance_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            
            checkpoint_manager = checkpoint_managers[instance['current_host']]
            
            # Processera första 2 cases för varje instans
            for i, case_id in enumerate(instance_cases[:2]):
                checkpoint_manager.add_processed_case(case_id)
                
                # Spara checkpoint var 2:e case
                if (i + 1) % 2 == 0:
                    checkpoint_saved = checkpoint_manager.save_checkpoint(case_id)
                    self.assertTrue(checkpoint_saved, f"Checkpoint ska sparas för {instance['current_host']}")
                    print(f"   💾 Checkpoint saved for {instance['current_host']}")
        
        # Assert - Verifiera distributed checkpoint creation
        print("\n✅ Assert: Verifiera distributed checkpoint creation")
        
        # Lista alla checkpoint-filer i S3
        bucket_name = self.test_checkpoint_path.split('/')[2]
        checkpoint_prefix = '/'.join(self.test_checkpoint_path.split('/')[3:])
        
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=checkpoint_prefix
        )
        
        self.assertIn('Contents', response, "Checkpoint-filer ska finnas i S3")
        
        checkpoint_files = [
            obj for obj in response['Contents']
            if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
        ]
        
        self.assertGreater(len(checkpoint_files), 0, "Minst en checkpoint-fil ska finnas")
        
        # Verifiera att varje instans har sina egna checkpoint-filer
        instance_checkpoints = {}
        for checkpoint_file in checkpoint_files:
            s3_key = checkpoint_file['Key']
            
            # Extrahera instans från path (s3://bucket/path/algo-X/checkpoint_xxx.json)
            path_parts = s3_key.split('/')
            instance_name = path_parts[-2]  # algo-X
            
            if instance_name not in instance_checkpoints:
                instance_checkpoints[instance_name] = []
            instance_checkpoints[instance_name].append(s3_key)
        
        # Verifiera att varje instans har checkpoint-filer
        for instance in instances:
            instance_host = instance['current_host']
            self.assertIn(instance_host, instance_checkpoints, 
                         f"Instance {instance_host} ska ha checkpoint-filer")
            
            instance_files = instance_checkpoints[instance_host]
            self.assertGreater(len(instance_files), 0, 
                             f"Instance {instance_host} ska ha minst en checkpoint-fil")
            
            print(f"   ✅ Instance {instance_host}: {len(instance_files)} checkpoint files")
        
        print("✅ Test passed: T077 Distributed Checkpoint Creation")
    
    def test_t078_checkpoint_path_conflicts(self):
        """T078: Test Checkpoint Path Conflicts - Verifiera att instanser inte skriver till samma checkpoint"""
        # Arrange
        print("\n🔧 Arrange: Skapa checkpoint path conflict scenario")
        
        # Skapa 3 instanser med checkpoint managers
        instances = [
            {'current_host': 'algo-1', 'host_index': 0, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-2', 'host_index': 1, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-3', 'host_index': 2, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']}
        ]
        
        checkpoint_managers = {}
        checkpoint_paths = {}
        
        for instance in instances:
            checkpoint_manager = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=1
            )
            checkpoint_managers[instance['current_host']] = checkpoint_manager
            checkpoint_paths[instance['current_host']] = checkpoint_manager.instance_checkpoint_path
        
        # Act - Skapa checkpoints samtidigt för alla instanser
        print("\n🎬 Act: Skapa checkpoints samtidigt för alla instanser")
        
        # Verifiera att alla instanser har unika checkpoint paths
        unique_paths = set(checkpoint_paths.values())
        self.assertEqual(len(unique_paths), len(instances), 
                        "Alla instanser ska ha unika checkpoint paths")
        
        # Skapa checkpoints för alla instanser
        for instance in instances:
            checkpoint_manager = checkpoint_managers[instance['current_host']]
            checkpoint_manager.add_processed_case(f"test_case_{instance['current_host']}")
            
            checkpoint_saved = checkpoint_manager.save_checkpoint(f"test_case_{instance['current_host']}")
            self.assertTrue(checkpoint_saved, f"Checkpoint ska sparas för {instance['current_host']}")
            print(f"   💾 Checkpoint saved for {instance['current_host']}")
        
        # Assert - Verifiera att inga path conflicts finns
        print("\n✅ Assert: Verifiera att inga path conflicts finns")
        
        # Lista alla checkpoint-filer i S3
        bucket_name = self.test_checkpoint_path.split('/')[2]
        checkpoint_prefix = '/'.join(self.test_checkpoint_path.split('/')[3:])
        
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=checkpoint_prefix
        )
        
        self.assertIn('Contents', response, "Checkpoint-filer ska finnas i S3")
        
        checkpoint_files = [
            obj for obj in response['Contents']
            if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
        ]
        
        # Gruppera checkpoint-filer per instans
        instance_files = {}
        for checkpoint_file in checkpoint_files:
            s3_key = checkpoint_file['Key']
            path_parts = s3_key.split('/')
            instance_name = path_parts[-2]  # algo-X
            
            if instance_name not in instance_files:
                instance_files[instance_name] = []
            instance_files[instance_name].append(s3_key)
        
        # Verifiera att varje instans har sina egna filer
        for instance in instances:
            instance_host = instance['current_host']
            self.assertIn(instance_host, instance_files, 
                         f"Instance {instance_host} ska ha checkpoint-filer")
            
            instance_checkpoint_files = instance_files[instance_host]
            self.assertGreater(len(instance_checkpoint_files), 0, 
                             f"Instance {instance_host} ska ha minst en checkpoint-fil")
            
            # Verifiera att filerna tillhör rätt instans
            for file_path in instance_checkpoint_files:
                self.assertIn(f"/{instance_host}/", file_path, 
                             f"Checkpoint-fil {file_path} ska tillhöra {instance_host}")
        
        # Verifiera att inga filer finns i fel instans-mapp
        for instance in instances:
            instance_host = instance['current_host']
            for other_instance in instances:
                if other_instance['current_host'] != instance_host:
                    other_host = other_instance['current_host']
                    for file_path in instance_files.get(instance_host, []):
                        self.assertNotIn(f"/{other_host}/", file_path, 
                                        f"Checkpoint-fil {file_path} ska inte tillhöra {other_host}")
        
        print(f"   ✅ All instances have unique checkpoint paths")
        print(f"   ✅ No path conflicts detected")
        print(f"   ✅ Instance-specific files verified")
        
        print("✅ Test passed: T078 Checkpoint Path Conflicts")
    
    def test_t079_instance_specific_recovery(self):
        """T079: Test Instance-Specific Recovery - Verifiera att instanser kan återhämta sina egna checkpoints"""
        # Arrange
        print("\n🔧 Arrange: Skapa instance-specific recovery scenario")
        
        # Skapa 3 instanser med checkpoint managers
        instances = [
            {'current_host': 'algo-1', 'host_index': 0, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-2', 'host_index': 1, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']},
            {'current_host': 'algo-3', 'host_index': 2, 'total_hosts': 3, 'all_hosts': ['algo-1', 'algo-2', 'algo-3']}
        ]
        
        # Skapa initial checkpoint managers och spara checkpoints
        initial_checkpoint_managers = {}
        for instance in instances:
            checkpoint_manager = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=1
            )
            initial_checkpoint_managers[instance['current_host']] = checkpoint_manager
            
            # Lägg till processed cases för varje instans
            instance_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            
            # Processera första 2 cases för varje instans
            for case_id in instance_cases[:2]:
                checkpoint_manager.add_processed_case(case_id)
            
            # Spara checkpoint
            checkpoint_saved = checkpoint_manager.save_checkpoint(instance_cases[1])
            self.assertTrue(checkpoint_saved, f"Checkpoint ska sparas för {instance['current_host']}")
            print(f"   💾 Initial checkpoint saved for {instance['current_host']}")
        
        # Act - Simulera restart och återhämta instance-specific checkpoints
        print("\n🎬 Act: Simulera restart och återhämta instance-specific checkpoints")
        
        # Skapa nya checkpoint managers (simulerar restart)
        recovery_checkpoint_managers = {}
        for instance in instances:
            checkpoint_manager = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=1
            )
            recovery_checkpoint_managers[instance['current_host']] = checkpoint_manager
        
        # Ladda checkpoints för varje instans
        for instance in instances:
            checkpoint_manager = recovery_checkpoint_managers[instance['current_host']]
            checkpoint_loaded = checkpoint_manager.load_checkpoint()
            
            self.assertTrue(checkpoint_loaded, f"Checkpoint ska kunna laddas för {instance['current_host']}")
            print(f"   📂 Checkpoint loaded for {instance['current_host']}")
        
        # Assert - Verifiera instance-specific recovery
        print("\n✅ Assert: Verifiera instance-specific recovery")
        
        # Verifiera att varje instans återhämtade sina egna cases
        for instance in instances:
            instance_host = instance['current_host']
            
            # Hämta ursprungliga cases för denna instans
            original_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            
            # Hämta återhämtade cases
            recovery_checkpoint_manager = recovery_checkpoint_managers[instance_host]
            recovered_cases = list(recovery_checkpoint_manager.processed_cases)
            
            # Verifiera att återhämtade cases matchar ursprungliga cases
            expected_cases = set(original_cases[:2])  # Första 2 cases
            recovered_cases_set = set(recovered_cases)
            
            self.assertEqual(recovered_cases_set, expected_cases,
                           f"Instance {instance_host} ska återhämta sina egna cases")
            
            # Verifiera att instansen inte återhämtade andra instansers cases
            for other_instance in instances:
                if other_instance['current_host'] != instance_host:
                    other_cases = CaseDistributor.distribute_cases_across_instances(
                        self.all_cases, other_instance
                    )
                    other_cases_set = set(other_cases[:2])
                    
                    overlap = recovered_cases_set & other_cases_set
                    self.assertEqual(len(overlap), 0,
                                   f"Instance {instance_host} ska inte återhämta cases från {other_instance['current_host']}")
            
            print(f"   ✅ Instance {instance_host}: {len(recovered_cases)} cases recovered correctly")
        
        # Verifiera att alla instanser kan identifiera sina remaining cases
        for instance in instances:
            instance_host = instance['current_host']
            checkpoint_manager = recovery_checkpoint_managers[instance_host]
            
            original_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            remaining_cases = checkpoint_manager.get_remaining_cases(original_cases)
            
            # Första 2 cases ska vara processade, resten ska vara remaining
            expected_remaining = original_cases[2:]
            self.assertEqual(set(remaining_cases), set(expected_remaining),
                           f"Instance {instance_host} ska ha korrekta remaining cases")
            
            print(f"   ✅ Instance {instance_host}: {len(remaining_cases)} remaining cases")
        
        print("✅ Test passed: T079 Instance-Specific Recovery")
    
    def test_t072_t075_t077_t078_t079_combined_workflow(self):
        """Kombinerat test för T072, T075, T077, T078, T079 - Fullständig multi-instance workflow"""
        # Arrange
        print("\n🔧 Arrange: Fullständig multi-instance workflow")
        
        # Skapa 6 instanser för kombinerat test
        instances = [
            {'current_host': f'algo-{i}', 'host_index': i, 'total_hosts': 6, 
             'all_hosts': [f'algo-{j}' for j in range(1, 7)]}
            for i in range(6)
        ]
        
        # Act - Fullständig multi-instance workflow
        print("\n🎬 Act: Fullständig multi-instance workflow")
        
        # T072 & T075: Distribuera cases mellan 6 instanser
        all_instances_cases = {}
        checkpoint_managers = {}
        
        for instance in instances:
            instance_cases = CaseDistributor.distribute_cases_across_instances(
                self.all_cases, instance
            )
            all_instances_cases[instance['current_host']] = instance_cases
            
            checkpoint_manager = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=2
            )
            checkpoint_managers[instance['current_host']] = checkpoint_manager
            
            print(f"   📋 Instance {instance['current_host']}: {len(instance_cases)} cases")
        
        # T077 & T078: Skapa distributed checkpoints
        for instance in instances:
            instance_host = instance['current_host']
            instance_cases = all_instances_cases[instance_host]
            checkpoint_manager = checkpoint_managers[instance_host]
            
            # Processera första 2 cases för varje instans
            for i, case_id in enumerate(instance_cases[:2]):
                checkpoint_manager.add_processed_case(case_id)
                
                # Spara checkpoint var 2:e case
                if (i + 1) % 2 == 0:
                    checkpoint_saved = checkpoint_manager.save_checkpoint(case_id)
                    self.assertTrue(checkpoint_saved, f"Checkpoint ska sparas för {instance_host}")
                    print(f"   💾 Checkpoint saved for {instance_host}")
        
        # T079: Simulera restart och instance-specific recovery
        print("\n🔄 Simulera restart och instance-specific recovery")
        
        recovery_checkpoint_managers = {}
        for instance in instances:
            checkpoint_manager = MultiInstanceCheckpointManager(
                checkpoint_path=self.test_checkpoint_path,
                host_info=instance,
                enable_checkpoints=True,
                checkpoint_interval=2
            )
            recovery_checkpoint_managers[instance['current_host']] = checkpoint_manager
        
        # Ladda checkpoints för varje instans
        for instance in instances:
            checkpoint_manager = recovery_checkpoint_managers[instance['current_host']]
            checkpoint_loaded = checkpoint_manager.load_checkpoint()
            self.assertTrue(checkpoint_loaded, f"Checkpoint ska kunna laddas för {instance['current_host']}")
        
        # Assert - Verifiera kombinerat workflow
        print("\n✅ Assert: Verifiera kombinerat workflow")
        
        # T072: Verifiera att ingen case overlap finns
        overlaps, all_distributed_cases = CaseDistributor.verify_no_case_overlap(all_instances_cases)
        self.assertEqual(len(overlaps), 0, f"Case overlap hittades: {overlaps}")
        
        # T075: Verifiera distribution med 6 instanser
        self.assertEqual(len(all_instances_cases), 6, "Alla 6 instanser ska ha cases")
        is_even, max_diff = CaseDistributor.verify_even_distribution(all_instances_cases)
        self.assertTrue(is_even, f"Case distribution är inte jämn, max skillnad: {max_diff}")
        
        # T077 & T078: Verifiera distributed checkpoint creation och path conflicts
        bucket_name = self.test_checkpoint_path.split('/')[2]
        checkpoint_prefix = '/'.join(self.test_checkpoint_path.split('/')[3:])
        
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=checkpoint_prefix
        )
        
        self.assertIn('Contents', response, "Checkpoint-filer ska finnas i S3")
        
        checkpoint_files = [
            obj for obj in response['Contents']
            if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
        ]
        
        # Gruppera checkpoint-filer per instans
        instance_files = {}
        for checkpoint_file in checkpoint_files:
            s3_key = checkpoint_file['Key']
            path_parts = s3_key.split('/')
            instance_name = path_parts[-2]  # algo-X
            
            if instance_name not in instance_files:
                instance_files[instance_name] = []
            instance_files[instance_name].append(s3_key)
        
        # Verifiera att varje instans har sina egna checkpoint-filer
        for instance in instances:
            instance_host = instance['current_host']
            self.assertIn(instance_host, instance_files, 
                         f"Instance {instance_host} ska ha checkpoint-filer")
            self.assertGreater(len(instance_files[instance_host]), 0, 
                             f"Instance {instance_host} ska ha minst en checkpoint-fil")
        
        # T079: Verifiera instance-specific recovery
        for instance in instances:
            instance_host = instance['current_host']
            checkpoint_manager = recovery_checkpoint_managers[instance_host]
            
            original_cases = all_instances_cases[instance_host]
            remaining_cases = checkpoint_manager.get_remaining_cases(original_cases)
            
            # Första 2 cases ska vara processade, resten ska vara remaining
            expected_remaining = original_cases[2:]
            self.assertEqual(set(remaining_cases), set(expected_remaining),
                           f"Instance {instance_host} ska ha korrekta remaining cases")
        
        print(f"   ✅ No case overlap verified")
        print(f"   ✅ 6-instance distribution verified")
        print(f"   ✅ Distributed checkpoint creation verified")
        print(f"   ✅ No path conflicts verified")
        print(f"   ✅ Instance-specific recovery verified")
        
        print("✅ Test passed: T072, T075, T077, T078, T079 Combined Workflow")


if __name__ == '__main__':
    # Kör tester med verbose output
    unittest.main(verbosity=2)
