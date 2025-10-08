#!/usr/bin/env python3
"""
T040: Test TFRecord File Creation
Verifiera att TFRecord-filer faktiskt skapas på disk
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf
import shutil

# Lägg till src-mappen i Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Importera direkt från vår master POC TFRecord creator modul
import importlib.util
spec = importlib.util.spec_from_file_location(
    "master_poc_tfrecord_creator", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'master_poc_tfrecord_creator.py')
)
master_poc_tfrecord_creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_poc_tfrecord_creator)

# Använd modulen
MasterPOCTFRecordCreator = master_poc_tfrecord_creator.MasterPOCTFRecordCreator
TFRecordConfig = master_poc_tfrecord_creator.TFRecordConfig
create_master_poc_tfrecord_creator = master_poc_tfrecord_creator.create_master_poc_tfrecord_creator

class TestT040TFRecordFileCreation(unittest.TestCase):
    """T040: Test TFRecord File Creation"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup efter varje test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_t040_tfrecord_file_creation_basic(self):
        """
        T040: Test TFRecord File Creation
        Verifiera att TFRecord-filer faktiskt skapas på disk
        """
        # Arrange
        # Skapa test data
        n_windows = 10
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_tfrecord_creation")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att fil har korrekt namn
        expected_filename = "test_tfrecord_creation_test.tfrecord"
        self.assertEqual(os.path.basename(tfrecord_path), expected_filename, 
                       f"TFRecord fil ska ha namn {expected_filename}")
        
        # Verifiera att fil har korrekt storlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "TFRecord fil ska ha storlek > 0")
        
        # Verifiera att fil är läsbar
        self.assertTrue(os.access(tfrecord_path, os.R_OK), "TFRecord fil ska vara läsbar")
        
        print(f"✅ T040 PASSED: Basic TFRecord file creation fungerar korrekt")
        print(f"   Fil skapad: {tfrecord_path}")
        print(f"   Fil storlek: {file_size / (1024 * 1024):.2f}MB")
    
    def test_t040_tfrecord_file_creation_multiple_files(self):
        """
        Verifiera att multiple TFRecord-filer kan skapas på disk
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Skapa flera filer med olika namn
        file_names = ["train", "validation", "test"]
        created_files = []
        
        # Act
        for file_name in file_names:
            output_path = os.path.join(self.temp_dir, f"test_multiple_{file_name}")
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                windows_data, static_data, targets_data, output_path, file_name
            )
            created_files.append(tfrecord_path)
        
        # Assert
        # Verifiera att alla filer skapas
        self.assertEqual(len(created_files), len(file_names), "Ska skapa alla filer")
        
        for i, tfrecord_path in enumerate(created_files):
            # Verifiera att fil existerar
            self.assertIsNotNone(tfrecord_path, f"TFRecord fil {i} ska skapas")
            self.assertTrue(os.path.exists(tfrecord_path), f"TFRecord fil {i} ska existera på disk")
            
            # Verifiera att fil har korrekt namn
            expected_filename = f"test_multiple_{file_names[i]}_{file_names[i]}.tfrecord"
            self.assertEqual(os.path.basename(tfrecord_path), expected_filename, 
                           f"TFRecord fil {i} ska ha namn {expected_filename}")
            
            # Verifiera att fil har storlek > 0
            file_size = os.path.getsize(tfrecord_path)
            self.assertGreater(file_size, 0, f"TFRecord fil {i} ska ha storlek > 0")
        
        print(f"✅ T040 PASSED: Multiple TFRecord file creation fungerar korrekt")
        print(f"   Skapade {len(created_files)} filer")
    
    def test_t040_tfrecord_file_creation_directory_creation(self):
        """
        Verifiera att directories skapas automatiskt för TFRecord-filer
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Skapa output path med nested directories som inte finns
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        output_path = os.path.join(nested_dir, "test_nested")
        
        # Verifiera att directories inte finns från början
        self.assertFalse(os.path.exists(nested_dir), "Nested directory ska inte finnas från början")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas i nested directory")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att nested directories skapas
        self.assertTrue(os.path.exists(nested_dir), "Nested directories ska skapas automatiskt")
        
        # Verifiera att fil har korrekt sökväg
        expected_path = os.path.join(nested_dir, "test_nested_test.tfrecord")
        self.assertEqual(tfrecord_path, expected_path, "TFRecord fil ska ha korrekt sökväg")
        
        print(f"✅ T040 PASSED: Directory creation för TFRecord files fungerar korrekt")
        print(f"   Nested directory skapad: {nested_dir}")
    
    def test_t040_tfrecord_file_creation_file_permissions(self):
        """
        Verifiera att TFRecord-filer har korrekta permissions
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_permissions")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att fil är läsbar
        self.assertTrue(os.access(tfrecord_path, os.R_OK), "TFRecord fil ska vara läsbar")
        
        # Verifiera att fil är skrivbar (för framtida uppdateringar)
        self.assertTrue(os.access(tfrecord_path, os.W_OK), "TFRecord fil ska vara skrivbar")
        
        # Verifiera att fil inte är executable (säkerhet)
        self.assertFalse(os.access(tfrecord_path, os.X_OK), "TFRecord fil ska inte vara executable")
        
        print(f"✅ T040 PASSED: File permissions för TFRecord files fungerar korrekt")
    
    def test_t040_tfrecord_file_creation_file_overwrite(self):
        """
        Verifiera att TFRecord-filer kan skrivas över
        """
        # Arrange
        n_windows = 5
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_overwrite")
        
        # Act - Skapa första filen
        tfrecord_path1 = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test1"
        )
        
        # Verifiera att första filen skapas
        self.assertIsNotNone(tfrecord_path1, "Första TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path1), "Första TFRecord fil ska existera")
        
        # Mät storlek på första filen
        file_size1 = os.path.getsize(tfrecord_path1)
        
        # Act - Skapa andra filen med samma namn (skriver över)
        tfrecord_path2 = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test1"
        )
        
        # Assert
        # Verifiera att andra filen skapas
        self.assertIsNotNone(tfrecord_path2, "Andra TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path2), "Andra TFRecord fil ska existera")
        
        # Verifiera att filerna har samma sökväg
        self.assertEqual(tfrecord_path1, tfrecord_path2, "Filer ska ha samma sökväg")
        
        # Verifiera att andra filen har storlek > 0
        file_size2 = os.path.getsize(tfrecord_path2)
        self.assertGreater(file_size2, 0, "Andra TFRecord fil ska ha storlek > 0")
        
        print(f"✅ T040 PASSED: File overwrite för TFRecord files fungerar korrekt")
        print(f"   Fil storlek före: {file_size1 / (1024 * 1024):.2f}MB")
        print(f"   Fil storlek efter: {file_size2 / (1024 * 1024):.2f}MB")
    
    def test_t040_tfrecord_file_creation_large_files(self):
        """
        Verifiera att stora TFRecord-filer kan skapas på disk
        """
        # Arrange
        n_windows = 1000  # Stort dataset
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_large_file")
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "Stor TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "Stor TFRecord fil ska existera på disk")
        
        # Verifiera att fil har storlek > 0
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "Stor TFRecord fil ska ha storlek > 0")
        
        # Verifiera att fil är läsbar
        self.assertTrue(os.access(tfrecord_path, os.R_OK), "Stor TFRecord fil ska vara läsbar")
        
        # Verifiera att fil kan läsas tillbaka
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples från stor fil")
        
        print(f"✅ T040 PASSED: Large TFRecord file creation fungerar korrekt")
        print(f"   Fil storlek: {file_size / (1024 * 1024):.2f}MB")
        print(f"   Antal examples: {len(parsed_data)}")
    
    def test_t040_tfrecord_file_creation_compression(self):
        """
        Verifiera att TFRecord-filer skapas med korrekt compression
        """
        # Arrange
        n_windows = 100
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        # Testa olika compression types
        compression_types = ["GZIP", "ZLIB", None]
        created_files = []
        
        # Act
        for compression_type in compression_types:
            config = TFRecordConfig(compression_type=compression_type)
            tfrecord_creator = MasterPOCTFRecordCreator(config)
            
            output_path = os.path.join(self.temp_dir, f"test_compression_{compression_type or 'none'}")
            tfrecord_path = tfrecord_creator.create_memory_efficient_tfrecord(
                windows_data, static_data, targets_data, output_path, "test"
            )
            created_files.append((compression_type, tfrecord_path, tfrecord_creator))
        
        # Assert
        # Verifiera att alla filer skapas
        self.assertEqual(len(created_files), len(compression_types), "Ska skapa alla compression filer")
        
        for compression_type, tfrecord_path, tfrecord_creator in created_files:
            # Verifiera att fil existerar
            self.assertIsNotNone(tfrecord_path, f"TFRecord fil med {compression_type} compression ska skapas")
            self.assertTrue(os.path.exists(tfrecord_path), f"TFRecord fil med {compression_type} compression ska existera")
            
            # Verifiera att fil har storlek > 0
            file_size = os.path.getsize(tfrecord_path)
            self.assertGreater(file_size, 0, f"TFRecord fil med {compression_type} compression ska ha storlek > 0")
            
            # Verifiera att fil kan läsas med rätt compression type
            parsed_data = tfrecord_creator.read_tfrecord_file(tfrecord_path)
            self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples från {compression_type} fil")
        
        print(f"✅ T040 PASSED: Compression för TFRecord files fungerar korrekt")
        print(f"   Testade {len(compression_types)} compression types")
    
    def test_t040_tfrecord_file_creation_comprehensive(self):
        """
        Omfattande test av TFRecord file creation
        """
        # Arrange
        # Testa med olika dataset storlekar
        test_cases = [
            (1, "Single example"),
            (10, "Small dataset"),
            (100, "Medium dataset"),
            (500, "Large dataset"),
        ]
        
        for n_windows, description in test_cases:
            with self.subTest(case=description):
                # Arrange
                windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
                static_data = np.random.randn(n_windows, 6).astype(np.float32)
                targets_data = np.random.randn(n_windows, 8).astype(np.float32)
                
                output_path = os.path.join(self.temp_dir, f"test_comprehensive_{n_windows}")
                
                # Act
                tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                    windows_data, static_data, targets_data, output_path, "test"
                )
                
                # Assert
                # Verifiera att fil skapas
                self.assertIsNotNone(tfrecord_path, f"TFRecord fil ska skapas för {description}")
                self.assertTrue(os.path.exists(tfrecord_path), f"TFRecord fil ska existera för {description}")
                
                # Verifiera att fil har storlek > 0
                file_size = os.path.getsize(tfrecord_path)
                self.assertGreater(file_size, 0, f"TFRecord fil ska ha storlek > 0 för {description}")
                
                # Verifiera att fil kan läsas
                parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
                self.assertEqual(len(parsed_data), n_windows, f"Ska läsa {n_windows} examples för {description}")
        
        print(f"✅ T040 PASSED: Comprehensive TFRecord file creation test fungerar korrekt")
        print(f"   Verifierade {len(test_cases)} olika dataset storlekar")

if __name__ == '__main__':
    unittest.main()
