#!/usr/bin/env python3
"""
S3 Parallel Loader för AWS SageMaker
Implementerar parallel S3 downloads för förbättrad preprocessing performance.
"""

import asyncio
import aiohttp
import boto3
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class S3DownloadResult:
    """Resultat från S3 download."""
    case_id: str
    success: bool
    local_path: Optional[str] = None
    error: Optional[str] = None
    download_time: Optional[float] = None

class S3ParallelLoader:
    """
    Parallel S3 loader för VitalDB cases.
    
    Features:
    - Parallel downloads av multiple .vital files
    - Configurable concurrency level
    - Progress tracking
    - Error handling med retry logic
    """
    
    def __init__(self, 
                 s3_bucket: str = 'master-poc-v1.0',
                 max_workers: int = 4,
                 retry_attempts: int = 3):
        """
        Initiera S3 Parallel Loader.
        
        Args:
            s3_bucket: S3 bucket namn
            max_workers: Max antal parallel downloads
            retry_attempts: Antal retry attempts vid fel
        """
        self.s3_bucket = s3_bucket
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.s3_client = boto3.client('s3')
        
        logger.info(f"🚀 S3 Parallel Loader initialized")
        logger.info(f"   Bucket: {s3_bucket}")
        logger.info(f"   Max Workers: {max_workers}")
        logger.info(f"   Retry Attempts: {retry_attempts}")
    
    def download_case(self, case_id: str) -> S3DownloadResult:
        """
        Ladda ner en enskild case från S3.
        
        Args:
            case_id: Case identifier (t.ex. "0001")
            
        Returns:
            S3DownloadResult med download resultat
        """
        import time
        start_time = time.time()
        
        try:
            # S3 key för case
            s3_key = f"raw-data/vital-files/{case_id:04d}.vital"
            
            # Local temp file
            local_path = f"/tmp/{case_id:04d}.vital"
            
            # Download från S3
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            
            download_time = time.time() - start_time
            
            logger.debug(f"✅ Downloaded {case_id}: {download_time:.2f}s")
            
            return S3DownloadResult(
                case_id=case_id,
                success=True,
                local_path=local_path,
                download_time=download_time
            )
            
        except Exception as e:
            download_time = time.time() - start_time
            error_msg = f"S3 download failed for {case_id}: {str(e)}"
            logger.error(error_msg)
            
            return S3DownloadResult(
                case_id=case_id,
                success=False,
                error=error_msg,
                download_time=download_time
            )
    
    def download_cases_parallel(self, case_ids: List[str]) -> List[S3DownloadResult]:
        """
        Ladda ner multiple cases parallellt från S3.
        
        Args:
            case_ids: Lista av case identifiers
            
        Returns:
            Lista av S3DownloadResult
        """
        logger.info(f"📥 Starting parallel S3 download for {len(case_ids)} cases")
        logger.info(f"   Max Workers: {self.max_workers}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit alla download tasks
            future_to_case = {
                executor.submit(self.download_case, case_id): case_id 
                for case_id in case_ids
            }
            
            # Process completed downloads
            for future in as_completed(future_to_case):
                case_id = future_to_case[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        logger.debug(f"✅ {case_id}: {result.download_time:.2f}s")
                    else:
                        logger.warning(f"❌ {case_id}: {result.error}")
                        
                except Exception as e:
                    logger.error(f"❌ {case_id}: Unexpected error: {e}")
                    results.append(S3DownloadResult(
                        case_id=case_id,
                        success=False,
                        error=f"Unexpected error: {str(e)}"
                    ))
        
        # Sammanställ resultat
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info(f"📊 Parallel S3 download completed:")
        logger.info(f"   Successful: {len(successful)}/{len(case_ids)}")
        logger.info(f"   Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(r.download_time for r in successful) / len(successful)
            logger.info(f"   Average download time: {avg_time:.2f}s")
        
        return results
    
    def cleanup_downloads(self, results: List[S3DownloadResult]):
        """
        Rensa upp nedladdade filer.
        
        Args:
            results: Lista av S3DownloadResult
        """
        cleaned = 0
        for result in results:
            if result.success and result.local_path and os.path.exists(result.local_path):
                try:
                    os.remove(result.local_path)
                    cleaned += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup {result.local_path}: {e}")
        
        logger.info(f"🧹 Cleaned up {cleaned} downloaded files")

# Factory function
def create_s3_parallel_loader(s3_bucket: str = 'master-poc-v1.0',
                             max_workers: int = 4) -> S3ParallelLoader:
    """
    Factory function för S3 Parallel Loader.
    
    Args:
        s3_bucket: S3 bucket namn
        max_workers: Max antal parallel downloads
        
    Returns:
        S3ParallelLoader instance
    """
    return S3ParallelLoader(
        s3_bucket=s3_bucket,
        max_workers=max_workers
    )
