#!/usr/bin/env python3
"""
Master POC Preprocessing Starter Script v5.0
==============================================

Starter script för Master POC preprocessing med SageMaker PyTorch Estimator.
Implementerar alla kritiska funktioner från AWS_CHECKLIST_V5.0_3000_CASES.md.

KRITISKA FUNKTIONER:
- Multi-instance case distribution
- Spot instances med max_wait
- Enable checkpoints default=True
- Robust case format parsing
- Output verification

Author: Medical AI Development Team
Version: 5.0.0
"""

import argparse
import boto3
import time
from datetime import datetime
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

def create_sagemaker_estimator(args):
    """Skapa SageMaker PyTorch Estimator med Master POC v5.0 konfiguration."""
    
    # Get execution role - använd specifik role från aws_config.env
    role = "arn:aws:iam::631158448906:role/service-role/AmazonSageMaker-ExecutionRole-20250714T085596"
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='master_poc_preprocessing_v5.py',
        source_dir='src/',
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        framework_version='1.12.1',
        py_version='py38',
        hyperparameters={
            'cases': args.cases,
            'window-size': 300,
            'step-size': 30,
            'batch-size': args.batch_size,
            'checkpoint-interval': args.checkpoint_interval,
            'enable-checkpoints': 'true',  # KRITISKT: Default True
            'enable-memory-efficient': 'auto',
            'memory-efficient-threshold': 50,
            'timeseries-features': 16,  # Master POC spec
            'static-features': 6,       # Master POC spec
            'output-features': 8,       # Master POC spec
            'normalization-range': '-1.0 1.0',
            'imputation-method': 'master_poc_smart_forward_fill',
            'pipeline-type': 'master_poc',
            's3-bucket': args.s3_bucket,
            'checkpoint-base-path': args.checkpoint_path,
            # Orchestrator-specifika hyperparameters
            'enable-s3': 'true',
            'enable-streaming': 'true',
            'memory-efficient': 'true'
        },
        use_spot_instances=args.use_spot,
        max_run=args.max_run,
        max_wait=args.max_wait if args.use_spot else None,
        output_path=args.output_path,
        job_name=f'master-poc-preprocessing-v5-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        tags=[
            {'Key': 'Project', 'Value': 'Master-POC'},
            {'Key': 'Version', 'Value': '5.0'},
            {'Key': 'Environment', 'Value': 'Production'}
        ]
    )
    
    return estimator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Master POC Preprocessing v5.0 Starter')
    
    # Required arguments
    parser.add_argument('--cases', type=str, required=True,
                       help='Case range to process (e.g., "1-3000", "1-10,17,0022")')
    
    # AWS Configuration
    parser.add_argument('--instance-type', type=str, default='ml.m5.2xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=6,
                       help='Number of SageMaker instances')
    parser.add_argument('--use-spot', action='store_true', default=True,
                       help='Use spot instances for cost savings')
    parser.add_argument('--max-run', type=int, default=93600,
                       help='Maximum runtime in seconds (26 hours)')
    parser.add_argument('--max-wait', type=int, default=187200,
                       help='Maximum wait time for spot instances (52 hours)')
    
    # S3 Configuration
    parser.add_argument('--s3-bucket', type=str, default='master-poc-v1.0',
                       help='S3 bucket name')
    parser.add_argument('--output-path', type=str, 
                       default='s3://master-poc-v1.0/processed-data/master-poc-pipeline/',
                       help='S3 output path')
    parser.add_argument('--checkpoint-path', type=str,
                       default='s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/',
                       help='S3 checkpoint path')
    
    # Processing Configuration
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Cases per processing batch')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Save checkpoint every N cases')
    
    # Test mode
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with smaller dataset')
    
    return parser.parse_args()

def validate_configuration(args):
    """Validera konfiguration innan körning."""
    print("🔍 Validating configuration...")
    
    # Validate case format
    if not args.cases:
        raise ValueError("Cases parameter is required")
    
    # Validate instance count för case distribution
    if args.instance_count < 1:
        raise ValueError("Instance count must be >= 1")
    
    # Validate spot instance configuration
    if args.use_spot and args.max_wait <= args.max_run:
        raise ValueError("Max wait must be > max run for spot instances")
    
    # Validate S3 paths
    if not args.s3_bucket.startswith('master-poc'):
        print("⚠️ Warning: S3 bucket doesn't follow Master POC naming convention")
    
    print("✅ Configuration validation passed")

def monitor_job(estimator):
    """Övervaka SageMaker job progress."""
    
    # Start job first
    print("🚀 Starting SageMaker training job...")
    estimator.fit(wait=False)
    
    # Now we can access the job name
    print(f"📊 Monitoring job: {estimator.latest_training_job.name}")
    
    # Monitor progress
    while True:
        try:
            job_status = estimator.latest_training_job.describe()['TrainingJobStatus']
            
            if job_status in ['Completed', 'Failed', 'Stopped']:
                print(f"🏁 Job finished with status: {job_status}")
                break
            else:
                print(f"⏳ Job status: {job_status}")
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            time.sleep(60)

def verify_output(args):
    """Verifiera att output skapades korrekt."""
    print("🔍 Verifying output...")
    
    try:
        s3_client = boto3.client('s3')
        bucket = args.s3_bucket
        prefix = 'processed-data/master-poc-pipeline/'
        
        # List files in output path
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"✅ Found {len(files)} files in output:")
            for file in files:
                print(f"   - {file}")
        else:
            print("❌ No files found in output path")
            
    except Exception as e:
        print(f"❌ Error verifying output: {e}")

def main():
    """Main entry point."""
    print("🚀 Starting Master POC Preprocessing v5.0")
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate configuration
        validate_configuration(args)
        
        # Print configuration
        print("\n📋 Configuration:")
        print(f"   Cases: {args.cases}")
        print(f"   Instance Type: {args.instance_type}")
        print(f"   Instance Count: {args.instance_count}")
        print(f"   Use Spot Instances: {args.use_spot}")
        print(f"   Max Run: {args.max_run}s ({args.max_run/3600:.1f}h)")
        if args.use_spot:
            print(f"   Max Wait: {args.max_wait}s ({args.max_wait/3600:.1f}h)")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Checkpoint Interval: {args.checkpoint_interval}")
        print(f"   S3 Bucket: {args.s3_bucket}")
        print(f"   Output Path: {args.output_path}")
        
        # Create estimator
        print("\n🔧 Creating SageMaker estimator...")
        estimator = create_sagemaker_estimator(args)
        
        # Start job
        print("\n🚀 Starting preprocessing job...")
        monitor_job(estimator)
        
        # Verify output
        print("\n🔍 Verifying output...")
        verify_output(args)
        
        print("\n🎉 Master POC preprocessing v5.0 completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

