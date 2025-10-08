#!/usr/bin/env python3
"""
T047: Test Split Randomization - Verifiera att split är randomiserad men deterministisk

AAA Format:
- Arrange: Skapa testdata med olika random seeds
- Act: Kör split_data_70_15_15 med olika seeds
- Assert: Verifiera att olika seeds ger olika splits, samma seed ger samma split
"""

import unittest
import numpy as np
import sys
import os
import importlib.util

# Lägg till src i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Ladda modulen direkt för att undvika import-problem
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
tfrecord_creator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfrecord_creator_module)

# Importera funktioner
split_data_70_15_15 = tfrecord_creator_module.split_data_70_15_15
split_data_by_case_70_15_15 = tfrecord_creator_module.split_data_by_case_70_15_15


class TestT047SplitRandomization(unittest.TestCase):
    """T047: Test Split Randomization - Verifiera att split är randomiserad men deterministisk"""
    
    def setUp(self):
        """Setup testdata"""
        # Skapa testdata med 100 windows
        self.n_windows = 100
        self.windows_data = np.random.randn(self.n_windows, 300, 16)
        self.static_data = np.random.randn(self.n_windows, 6)
        self.targets_data = np.random.randn(self.n_windows, 8)
        
        # Skapa case-based testdata
        self.case_windows = {}
        self.case_static = {}
        self.case_targets = {}
        
        for i in range(10):  # 10 cases
            case_id = f"case_{i:03d}"
            n_windows_per_case = np.random.randint(5, 15)  # 5-14 windows per case
            
            self.case_windows[case_id] = [np.random.randn(300, 16) for _ in range(n_windows_per_case)]
            self.case_static[case_id] = np.random.randn(6)
            self.case_targets[case_id] = [np.random.randn(8) for _ in range(n_windows_per_case)]
    
    def test_t047_split_randomization_different_seeds(self):
        """T047: Olika seeds ska ge olika splits"""
        # Arrange
        seed1 = 42
        seed2 = 123
        seed3 = 999
        
        # Act
        train1, val1, test1 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed1
        )
        train2, val2, test2 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed2
        )
        train3, val3, test3 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed3
        )
        
        # Assert
        # Olika seeds ska ge olika splits (inte nödvändigtvis olika storlekar)
        # Men de ska ha olika data-innehåll
        train1_flat = train1['windows'].flatten()
        train2_flat = train2['windows'].flatten()
        train3_flat = train3['windows'].flatten()
        
        # Kontrollera att olika seeds ger olika data (inte bara olika storlekar)
        self.assertFalse(np.array_equal(train1_flat, train2_flat), 
                        "Olika seeds ska ge olika train data")
        self.assertFalse(np.array_equal(train1_flat, train3_flat), 
                        "Olika seeds ska ge olika train data")
        self.assertFalse(np.array_equal(train2_flat, train3_flat), 
                        "Olika seeds ska ge olika train data")
        
        # Kontrollera att val data också är olika
        val1_flat = val1['windows'].flatten()
        val2_flat = val2['windows'].flatten()
        val3_flat = val3['windows'].flatten()
        
        self.assertFalse(np.array_equal(val1_flat, val2_flat), 
                        "Olika seeds ska ge olika val data")
        self.assertFalse(np.array_equal(val1_flat, val3_flat), 
                        "Olika seeds ska ge olika val data")
        self.assertFalse(np.array_equal(val2_flat, val3_flat), 
                        "Olika seeds ska ge olika val data")
        
        # Kontrollera att test data också är olika
        test1_flat = test1['windows'].flatten()
        test2_flat = test2['windows'].flatten()
        test3_flat = test3['windows'].flatten()
        
        self.assertFalse(np.array_equal(test1_flat, test2_flat), 
                        "Olika seeds ska ge olika test data")
        self.assertFalse(np.array_equal(test1_flat, test3_flat), 
                        "Olika seeds ska ge olika test data")
        self.assertFalse(np.array_equal(test2_flat, test3_flat), 
                        "Olika seeds ska ge olika test data")
    
    def test_t047_split_randomization_same_seed_deterministic(self):
        """T047: Samma seed ska ge samma split"""
        # Arrange
        seed = 42
        
        # Act - Kör samma split två gånger med samma seed
        train1, val1, test1 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        train2, val2, test2 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        
        # Assert
        # Samma seed ska ge exakt samma split storlekar
        self.assertEqual(len(train1['windows']), len(train2['windows']), 
                        "Samma seed ska ge samma train split storlek")
        self.assertEqual(len(val1['windows']), len(val2['windows']), 
                        "Samma seed ska ge samma val split storlek")
        self.assertEqual(len(test1['windows']), len(test2['windows']), 
                        "Samma seed ska ge samma test split storlek")
        
        # Samma seed ska ge exakt samma data (samma windows i samma splits)
        np.testing.assert_array_equal(train1['windows'], train2['windows'], 
                                    "Samma seed ska ge samma train data")
        np.testing.assert_array_equal(val1['windows'], val2['windows'], 
                                     "Samma seed ska ge samma val data")
        np.testing.assert_array_equal(test1['windows'], test2['windows'], 
                                     "Samma seed ska ge samma test data")
        
        np.testing.assert_array_equal(train1['static'], train2['static'], 
                                    "Samma seed ska ge samma train static data")
        np.testing.assert_array_equal(val1['static'], val2['static'], 
                                     "Samma seed ska ge samma val static data")
        np.testing.assert_array_equal(test1['static'], test2['static'], 
                                     "Samma seed ska ge samma test static data")
        
        np.testing.assert_array_equal(train1['targets'], train2['targets'], 
                                    "Samma seed ska ge samma train targets")
        np.testing.assert_array_equal(val1['targets'], val2['targets'], 
                                     "Samma seed ska ge samma val targets")
        np.testing.assert_array_equal(test1['targets'], test2['targets'], 
                                     "Samma seed ska ge samma test targets")
    
    def test_t047_split_randomization_case_based_different_seeds(self):
        """T047: Case-based split med olika seeds ska ge olika splits"""
        # Arrange
        seed1 = 42
        seed2 = 123
        seed3 = 999
        
        # Act
        train1, val1, test1 = split_data_by_case_70_15_15(
            self.case_windows, self.case_static, self.case_targets, seed1
        )
        train2, val2, test2 = split_data_by_case_70_15_15(
            self.case_windows, self.case_static, self.case_targets, seed2
        )
        train3, val3, test3 = split_data_by_case_70_15_15(
            self.case_windows, self.case_static, self.case_targets, seed3
        )
        
        # Assert
        # Olika seeds ska ge olika case splits (inte nödvändigtvis olika storlekar)
        # Men de ska ha olika data-innehåll
        train1_flat = train1['windows'].flatten()
        train2_flat = train2['windows'].flatten()
        train3_flat = train3['windows'].flatten()
        
        # Kontrollera att olika seeds ger olika data (inte bara olika storlekar)
        self.assertFalse(np.array_equal(train1_flat, train2_flat), 
                        "Olika seeds ska ge olika case-based train data")
        self.assertFalse(np.array_equal(train1_flat, train3_flat), 
                        "Olika seeds ska ge olika case-based train data")
        self.assertFalse(np.array_equal(train2_flat, train3_flat), 
                        "Olika seeds ska ge olika case-based train data")
        
        # Kontrollera att val data också är olika
        val1_flat = val1['windows'].flatten()
        val2_flat = val2['windows'].flatten()
        val3_flat = val3['windows'].flatten()
        
        self.assertFalse(np.array_equal(val1_flat, val2_flat), 
                        "Olika seeds ska ge olika case-based val data")
        self.assertFalse(np.array_equal(val1_flat, val3_flat), 
                        "Olika seeds ska ge olika case-based val data")
        self.assertFalse(np.array_equal(val2_flat, val3_flat), 
                        "Olika seeds ska ge olika case-based val data")
        
        # Kontrollera att test data också är olika
        test1_flat = test1['windows'].flatten()
        test2_flat = test2['windows'].flatten()
        test3_flat = test3['windows'].flatten()
        
        self.assertFalse(np.array_equal(test1_flat, test2_flat), 
                        "Olika seeds ska ge olika case-based test data")
        self.assertFalse(np.array_equal(test1_flat, test3_flat), 
                        "Olika seeds ska ge olika case-based test data")
        self.assertFalse(np.array_equal(test2_flat, test3_flat), 
                        "Olika seeds ska ge olika case-based test data")
    
    def test_t047_split_randomization_case_based_same_seed_deterministic(self):
        """T047: Case-based split med samma seed ska ge samma split"""
        # Arrange
        seed = 42
        
        # Act - Kör samma split två gånger med samma seed
        train1, val1, test1 = split_data_by_case_70_15_15(
            self.case_windows, self.case_static, self.case_targets, seed
        )
        train2, val2, test2 = split_data_by_case_70_15_15(
            self.case_windows, self.case_static, self.case_targets, seed
        )
        
        # Assert
        # Samma seed ska ge exakt samma split storlekar
        self.assertEqual(len(train1['windows']), len(train2['windows']), 
                        "Samma seed ska ge samma case-based train split storlek")
        self.assertEqual(len(val1['windows']), len(val2['windows']), 
                        "Samma seed ska ge samma case-based val split storlek")
        self.assertEqual(len(test1['windows']), len(test2['windows']), 
                        "Samma seed ska ge samma case-based test split storlek")
        
        # Samma seed ska ge exakt samma data
        np.testing.assert_array_equal(train1['windows'], train2['windows'], 
                                    "Samma seed ska ge samma case-based train data")
        np.testing.assert_array_equal(val1['windows'], val2['windows'], 
                                     "Samma seed ska ge samma case-based val data")
        np.testing.assert_array_equal(test1['windows'], test2['windows'], 
                                     "Samma seed ska ge samma case-based test data")
        
        np.testing.assert_array_equal(train1['static'], train2['static'], 
                                    "Samma seed ska ge samma case-based train static data")
        np.testing.assert_array_equal(val1['static'], val2['static'], 
                                     "Samma seed ska ge samma case-based val static data")
        np.testing.assert_array_equal(test1['static'], test2['static'], 
                                     "Samma seed ska ge samma case-based test static data")
        
        np.testing.assert_array_equal(train1['targets'], train2['targets'], 
                                    "Samma seed ska ge samma case-based train targets")
        np.testing.assert_array_equal(val1['targets'], val2['targets'], 
                                     "Samma seed ska ge samma case-based val targets")
        np.testing.assert_array_equal(test1['targets'], test2['targets'], 
                                     "Samma seed ska ge samma case-based test targets")
    
    def test_t047_split_randomization_edge_cases(self):
        """T047: Edge cases för randomization"""
        # Arrange
        # Mycket liten dataset
        small_windows = np.random.randn(3, 300, 16)
        small_static = np.random.randn(3, 6)
        small_targets = np.random.randn(3, 8)
        
        # Act
        train1, val1, test1 = split_data_70_15_15(
            small_windows, small_static, small_targets, 42
        )
        train2, val2, test2 = split_data_70_15_15(
            small_windows, small_static, small_targets, 123
        )
        
        # Assert
        # För små datasets ska olika seeds fortfarande ge olika splits
        # (även om båda kan ha allt i train)
        self.assertIsInstance(train1['windows'], np.ndarray, 
                             "Train split ska vara numpy array")
        self.assertIsInstance(train2['windows'], np.ndarray, 
                             "Train split ska vara numpy array")
        
        # För små datasets (3 windows) ska val och test vara tomma eller mycket små
        # Eftersom 3 * 0.7 = 2.1 -> 2 windows till train
        # 3 * 0.15 = 0.45 -> 0 windows till val
        # 3 * 0.15 = 0.45 -> 0 windows till test
        # Men det kan variera beroende på seed
        self.assertLessEqual(len(val1['windows']), 1, 
                            "Val split ska vara tom eller mycket liten för små dataset")
        self.assertLessEqual(len(test1['windows']), 1, 
                            "Test split ska vara tom eller mycket liten för små dataset")
        self.assertLessEqual(len(val2['windows']), 1, 
                            "Val split ska vara tom eller mycket liten för små dataset")
        self.assertLessEqual(len(test2['windows']), 1, 
                            "Test split ska vara tom eller mycket liten för små dataset")
        
        # Train ska ha majoriteten av data
        self.assertGreaterEqual(len(train1['windows']), 2, 
                               "Train split ska ha majoriteten av data för små dataset")
        self.assertGreaterEqual(len(train2['windows']), 2, 
                               "Train split ska ha majoriteten av data för små dataset")
    
    def test_t047_split_randomization_seed_zero(self):
        """T047: Seed 0 ska fungera korrekt"""
        # Arrange
        seed = 0
        
        # Act
        train1, val1, test1 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        train2, val2, test2 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        
        # Assert
        # Seed 0 ska ge deterministisk split
        self.assertEqual(len(train1['windows']), len(train2['windows']), 
                        "Seed 0 ska ge deterministisk split")
        self.assertEqual(len(val1['windows']), len(val2['windows']), 
                        "Seed 0 ska ge deterministisk split")
        self.assertEqual(len(test1['windows']), len(test2['windows']), 
                        "Seed 0 ska ge deterministisk split")
        
        # Seed 0 ska ge samma data
        np.testing.assert_array_equal(train1['windows'], train2['windows'], 
                                    "Seed 0 ska ge samma train data")
        np.testing.assert_array_equal(val1['windows'], val2['windows'], 
                                     "Seed 0 ska ge samma val data")
        np.testing.assert_array_equal(test1['windows'], test2['windows'], 
                                     "Seed 0 ska ge samma test data")
    
    def test_t047_split_randomization_large_seed(self):
        """T047: Stora seeds ska fungera korrekt"""
        # Arrange
        seed = 2**31 - 1  # Maximalt tillåtna seed för NumPy
        
        # Act
        train1, val1, test1 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        train2, val2, test2 = split_data_70_15_15(
            self.windows_data, self.static_data, self.targets_data, seed
        )
        
        # Assert
        # Stora seeds ska ge deterministisk split
        self.assertEqual(len(train1['windows']), len(train2['windows']), 
                        "Stora seeds ska ge deterministisk split")
        self.assertEqual(len(val1['windows']), len(val2['windows']), 
                        "Stora seeds ska ge deterministisk split")
        self.assertEqual(len(test1['windows']), len(test2['windows']), 
                        "Stora seeds ska ge deterministisk split")
        
        # Stora seeds ska ge samma data
        np.testing.assert_array_equal(train1['windows'], train2['windows'], 
                                    "Stora seeds ska ge samma train data")
        np.testing.assert_array_equal(val1['windows'], val2['windows'], 
                                     "Stora seeds ska ge samma val data")
        np.testing.assert_array_equal(test1['windows'], test2['windows'], 
                                     "Stora seeds ska ge samma test data")


if __name__ == '__main__':
    unittest.main()
