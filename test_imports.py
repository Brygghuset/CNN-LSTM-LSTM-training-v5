#!/usr/bin/env python3
"""Test alla kritiska imports för AWS preprocessing"""

import sys
import os
sys.path.insert(0, 'src')

def test_critical_imports():
    """Testa alla kritiska imports som behövs för AWS körning."""
    
    try:
        print("🧪 Testing critical imports...")
        
        # Test 1: Config module
        from config import get_config
        config = get_config()
        print("✅ config.py - OK")
        
        # Test 2: Checkpoint manager
        from checkpoint_manager import MasterPOCCheckpointManager, create_checkpoint_manager
        print("✅ checkpoint_manager.py - OK")
        
        # Test 3: Case range parser
        from utils.case_range_parser import parse_case_range
        cases = parse_case_range("1-10")
        print(f"✅ case_range_parser.py - OK (parsed {len(cases)} cases)")
        
        # Test 4: Orchestrator (kritisk test)
        from data.master_poc_preprocessing_orchestrator import MasterPOCPreprocessingOrchestrator, create_master_poc_orchestrator
        print("✅ master_poc_preprocessing_orchestrator.py - OK")
        
        # Test 5: Pipeline
        from data.master_poc_preprocessing_pipeline import MasterPOCPreprocessingPipeline, MasterPOCPreprocessingConfig
        print("✅ master_poc_preprocessing_pipeline.py - OK")
        
        # Test 6: TFRecord creator
        from data.master_poc_tfrecord_creator import MasterPOCTFRecordCreator
        print("✅ master_poc_tfrecord_creator.py - OK")
        
        # Test 7: Entry point
        import master_poc_preprocessing_v5
        print("✅ master_poc_preprocessing_v5.py - OK")
        
        print("\n🎉 ALL CRITICAL IMPORTS SUCCESSFUL!")
        print("✅ Ready for AWS pilot test!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = test_critical_imports()
    sys.exit(0 if success else 1)
