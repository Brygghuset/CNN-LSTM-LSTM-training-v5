#!/usr/bin/env python3
"""
Master POC Checkpoint Manager v5.0
==================================

Implementerar checkpoint system för Master POC preprocessing enligt AWS_CHECKLIST_V5.0_3000_CASES.md.
Baserat på pytorch_preprocessing_entry_point.py checkpoint manager.

Author: Medical AI Development Team
Version: 5.0.0
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
import boto3
from botocore.exceptions import ClientError

# Setup logging
logger = logging.getLogger(__name__)

class MasterPOCCheckpointManager:
    """Checkpoint manager för Master POC preprocessing"""
    
    def __init__(self, checkpoint_path: str, enable_checkpoints: bool = True, checkpoint_interval: int = 50):
        """
        Initiera checkpoint manager.
        
        Args:
            checkpoint_path: S3 path för checkpoint storage
            enable_checkpoints: Om checkpoints ska aktiveras
            checkpoint_interval: Antal cases mellan checkpoints
        """
        self.checkpoint_path = checkpoint_path.rstrip('/')
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        
        # State tracking
        self.processed_cases: Set[str] = set()
        self.failed_cases: Set[str] = set()
        self.current_case_index = 0
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        
        # S3 client
        self.s3_client = boto3.client('s3') if enable_checkpoints else None
        
        logger.info(f"🔧 Checkpoint Manager Initialized:")
        logger.info(f"   Path: {self.checkpoint_path}")
        logger.info(f"   Enabled: {self.enable_checkpoints}")
        logger.info(f"   Interval: {self.checkpoint_interval}")
    
    def should_save_checkpoint(self, case_id: str) -> bool:
        """
        Kontrollera om checkpoint ska sparas för denna case.
        
        Args:
            case_id: Case ID som just processades
            
        Returns:
            bool: True om checkpoint ska sparas
        """
        if not self.enable_checkpoints:
            return False
        
        # Spara var checkpoint_interval:e case
        return (self.current_case_index + 1) % self.checkpoint_interval == 0
    
    def add_processed_case(self, case_id: str) -> None:
        """
        Lägg till en processad case.
        
        Args:
            case_id: Case ID som processades
        """
        self.processed_cases.add(case_id)
        self.current_case_index += 1
        
        logger.debug(f"✅ Case {case_id} processed (total: {len(self.processed_cases)})")
    
    def add_failed_case(self, case_id: str, error: str) -> None:
        """
        Lägg till en misslyckad case.
        
        Args:
            case_id: Case ID som misslyckades
            error: Felmeddelande
        """
        self.failed_cases.add(case_id)
        self.current_case_index += 1
        
        logger.warning(f"❌ Case {case_id} failed: {error}")
    
    def create_checkpoint_data(self) -> Dict[str, Any]:
        """
        Skapa checkpoint data struktur.
        
        Returns:
            dict: Checkpoint data
        """
        checkpoint_data = {
            'checkpoint_info': {
                'timestamp': datetime.now().isoformat(),
                'checkpoint_path': self.checkpoint_path,
                'checkpoint_interval': self.checkpoint_interval,
                'enable_checkpoints': self.enable_checkpoints,
                'processing_time_seconds': time.time() - self.start_time
            },
            'processing_state': {
                'current_case_index': self.current_case_index,
                'total_processed': len(self.processed_cases),
                'total_failed': len(self.failed_cases),
                'last_checkpoint_time': self.last_checkpoint_time
            },
            'processed_cases': sorted(list(self.processed_cases)),
            'failed_cases': sorted(list(self.failed_cases)),
            'failed_case_details': {}  # Kan utökas med felmeddelanden
        }
        
        return checkpoint_data
    
    def save_checkpoint(self, case_id: str) -> bool:
        """
        Spara checkpoint till S3.
        
        Args:
            case_id: Case ID som triggade checkpoint
            
        Returns:
            bool: True om checkpoint sparades framgångsrikt
        """
        if not self.enable_checkpoints:
            logger.debug("Checkpoints disabled, skipping save")
            return True
        
        try:
            checkpoint_data = self.create_checkpoint_data()
            
            # Skapa checkpoint filnamn
            checkpoint_filename = f"checkpoint_{int(time.time())}.json"
            checkpoint_s3_path = f"{self.checkpoint_path}/{checkpoint_filename}"
            
            # Konvertera till JSON
            checkpoint_json = json.dumps(checkpoint_data, indent=2)
            
            # Ladda upp till S3
            bucket_name, s3_key = self._parse_s3_path(checkpoint_s3_path)
            
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=checkpoint_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            self.last_checkpoint_time = time.time()
            
            logger.info(f"💾 Checkpoint saved:")
            logger.info(f"   Triggered by: {case_id}")
            logger.info(f"   S3 Path: {checkpoint_s3_path}")
            logger.info(f"   Processed: {len(self.processed_cases)}")
            logger.info(f"   Failed: {len(self.failed_cases)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> bool:
        """
        Ladda senaste checkpoint från S3.
        
        Returns:
            bool: True om checkpoint laddades framgångsrikt
        """
        if not self.enable_checkpoints:
            logger.debug("Checkpoints disabled, skipping load")
            return True
        
        try:
            # Hitta senaste checkpoint fil
            bucket_name, s3_prefix = self._parse_s3_path(self.checkpoint_path)
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                logger.info("No existing checkpoints found")
                return True
            
            # Hitta senaste checkpoint fil
            checkpoint_files = [
                obj for obj in response['Contents']
                if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
            ]
            
            if not checkpoint_files:
                logger.info("No checkpoint files found")
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
            self.current_case_index = checkpoint_data.get('processing_state', {}).get('current_case_index', 0)
            
            logger.info(f"📂 Checkpoint loaded:")
            logger.info(f"   S3 Path: s3://{bucket_name}/{checkpoint_s3_key}")
            logger.info(f"   Processed: {len(self.processed_cases)}")
            logger.info(f"   Failed: {len(self.failed_cases)}")
            logger.info(f"   Case Index: {self.current_case_index}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def get_remaining_cases(self, all_cases: List[str]) -> List[str]:
        """
        Hämta cases som inte har processats än.
        
        Args:
            all_cases: Lista med alla cases som ska processas
            
        Returns:
            List[str]: Cases som inte har processats än
        """
        remaining_cases = [
            case_id for case_id in all_cases
            if case_id not in self.processed_cases and case_id not in self.failed_cases
        ]
        
        logger.info(f"📋 Remaining cases: {len(remaining_cases)}")
        return remaining_cases
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Hämta processing statistik.
        
        Returns:
            dict: Processing statistik
        """
        total_cases = len(self.processed_cases) + len(self.failed_cases)
        processing_time = time.time() - self.start_time
        
        stats = {
            'total_cases': total_cases,
            'processed_cases': len(self.processed_cases),
            'failed_cases': len(self.failed_cases),
            'processing_time_seconds': processing_time,
            'cases_per_second': total_cases / processing_time if processing_time > 0 else 0,
            'checkpoint_interval': self.checkpoint_interval,
            'enable_checkpoints': self.enable_checkpoints
        }
        
        return stats
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parse S3 path till bucket och key.
        
        Args:
            s3_path: S3 path (s3://bucket/key)
            
        Returns:
            tuple: (bucket_name, key)
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_without_protocol = s3_path[5:]  # Ta bort 's3://'
        parts = path_without_protocol.split('/', 1)
        
        bucket_name = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        return bucket_name, key
    
    def cleanup_old_checkpoints(self, keep_last: int = 3) -> bool:
        """
        Rensa gamla checkpoint filer, behåll senaste N.
        
        Args:
            keep_last: Antal senaste checkpoints att behålla
            
        Returns:
            bool: True om cleanup lyckades
        """
        if not self.enable_checkpoints:
            return True
        
        try:
            bucket_name, s3_prefix = self._parse_s3_path(self.checkpoint_path)
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                return True
            
            # Hitta checkpoint filer
            checkpoint_files = [
                obj for obj in response['Contents']
                if obj['Key'].endswith('.json') and 'checkpoint_' in obj['Key']
            ]
            
            if len(checkpoint_files) <= keep_last:
                return True
            
            # Sortera efter senast modifierad
            checkpoint_files.sort(key=lambda x: x['LastModified'], reverse=True)
            
            # Ta bort gamla checkpoints
            files_to_delete = checkpoint_files[keep_last:]
            
            for file_obj in files_to_delete:
                self.s3_client.delete_object(
                    Bucket=bucket_name,
                    Key=file_obj['Key']
                )
                logger.debug(f"🗑️ Deleted old checkpoint: {file_obj['Key']}")
            
            logger.info(f"🧹 Cleaned up {len(files_to_delete)} old checkpoints")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup checkpoints: {e}")
            return False

def get_memory_usage_mb() -> float:
    """Hämta aktuell minnesanvändning i MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def create_checkpoint_manager(checkpoint_path: str, enable_checkpoints: bool = True, checkpoint_interval: int = 50) -> MasterPOCCheckpointManager:
    """Skapa en ny MasterPOCCheckpointManager instans."""
    return MasterPOCCheckpointManager(checkpoint_path, enable_checkpoints, checkpoint_interval)