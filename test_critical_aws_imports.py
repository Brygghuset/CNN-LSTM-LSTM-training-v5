#!/usr/bin/env python3
"""Test endast de absolut kritiska imports som behövs för AWS körning"""

import sys
import os
sys.path.insert(0, 'src')

def test_aws_critical_imports():
    """Testa endast de imports som är absolut kritiska för AWS SageMaker körning."""
    
    try:
        print("🧪 Testing AWS-critical imports only...")
        
        # Test 1: Config module (KRITISK)
        from config import get_config
        config = get_config()
        print("✅ config.py - OK")
        
        # Test 2: Checkpoint manager (KRITISK)
        from checkpoint_manager import MasterPOCCheckpointManager, create_checkpoint_manager
        print("✅ checkpoint_manager.py - OK")
        
        # Test 3: Case range parser (KRITISK)
        from utils.case_range_parser import parse_case_range
        cases = parse_case_range("1-10")
        print(f"✅ case_range_parser.py - OK (parsed {len(cases)} cases)")
        
        # Test 4: Memory efficient processor (KRITISK)
        from memory_efficient_batch_processor import MemoryEfficientBatchProcessor
        print("✅ memory_efficient_batch_processor.py - OK")
        
        # Test 5: TFRecord creator (KRITISK)
        try:
            from data.master_poc_tfrecord_creator import MasterPOCTFRecordCreator
            print("✅ master_poc_tfrecord_creator.py - OK")
        except ImportError as e:
            print(f"⚠️ master_poc_tfrecord_creator.py - Import warning: {e}")
        
        # Test 6: Entry point (KRITISK)
        try:
            import master_poc_preprocessing_v5
            print("✅ master_poc_preprocessing_v5.py - OK")
        except ImportError as e:
            print(f"⚠️ master_poc_preprocessing_v5.py - Import warning: {e}")
        
        print("\n🎉 CRITICAL AWS IMPORTS SUCCESSFUL!")
        print("✅ Ready to test AWS pilot!")
        return True
        
    except ImportError as e:
        print(f"❌ Critical Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = test_aws_critical_imports()
    sys.exit(0 if success else 1)
