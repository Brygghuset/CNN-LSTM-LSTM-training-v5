#!/usr/bin/env python3
"""
Utility functions för Master POC CNN-LSTM-LSTM v5.0
==================================================

Kopierat från befintlig fungerande kod för att säkerställa kompatibilitet.

Author: Medical AI Development Team
Version: 5.0.0
"""

import logging
from typing import List

# Setup logging
logger = logging.getLogger(__name__)

def parse_case_range(case_range: str) -> List[str]:
    """
    Parse case range från olika format:
    - "1-100" -> ["0001", "0002", ..., "0100"]
    - "0001-0100" -> ["0001", "0002", ..., "0100"] 
    - "1,2,3" -> ["0001", "0002", "0003"]
    - "0001,0002,0003" -> ["0001", "0002", "0003"]
    - "1-10,17,0022" -> ["0001", "0002", ..., "0010", "0017", "0022"] (mixed format)
    """
    # Hantera tom sträng och whitespace
    case_range = case_range.strip()
    if not case_range:
        return []
    
    # Handle mixed format first (contains both '-' and ',')
    if '-' in case_range and ',' in case_range:
        # Mixed format: "1-10,17,0022"
        parts = case_range.split(',')
        all_case_ids = []
        
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue
            if '-' in part:
                # Range part: "1-10"
                start, end = part.split('-')
                start_int = int(start)
                end_int = int(end)
                
                for i in range(start_int, end_int + 1):
                    all_case_ids.append(f"{i:04d}")
            else:
                # Single case part: "17" or "0022"
                if part.isdigit():
                    all_case_ids.append(f"{int(part):04d}")
                else:
                    all_case_ids.append(part)
        
        logger.info(f"📦 Mixed format {case_range} -> {len(all_case_ids)} cases")
        return all_case_ids
    
    elif '-' in case_range:
        # Handle range format: "1-100" or "0001-0100"
        start, end = case_range.split('-')
        
        # Normalize to integers
        start_int = int(start)
        end_int = int(end)
        
        # Generate zero-padded case IDs
        case_ids = []
        for i in range(start_int, end_int + 1):
            case_ids.append(f"{i:04d}")
        
        logger.info(f"📦 Batch range {case_range} -> {len(case_ids)} cases: {case_ids[0]}-{case_ids[-1]}")
        return case_ids
    
    elif ',' in case_range:
        # Handle comma-separated format: "1,2,3" or "0001,0002,0003"
        cases = case_range.split(',')
        case_ids = []
        
        for case in cases:
            case = case.strip()
            if not case:  # Skip empty cases
                continue
            if case.isdigit():
                # Convert "1" to "0001"
                case_ids.append(f"{int(case):04d}")
            else:
                # Assume already formatted "0001"
                case_ids.append(case)
        
        logger.info(f"📦 Comma-separated {case_range} -> {len(case_ids)} cases")
        return case_ids
    
    else:
        # Single case
        case = case_range.strip()
        if case.isdigit():
            return [f"{int(case):04d}"]
        else:
            return [case]
