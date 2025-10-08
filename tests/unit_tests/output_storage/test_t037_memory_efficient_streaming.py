#!/usr/bin/env python3
"""
T037: Test Memory-Efficient Streaming
Verifiera att TFRecord skrivs streamat utan minnesöverskridning
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf
import psutil
import gc

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

class TestT037MemoryEfficientStreaming(unittest.TestCase):
    """T037: Test Memory-Efficient Streaming"""
    
    def setUp(self):
        """Setup för varje test."""
        self.tfrecord_creator = create_master_poc_tfrecord_creator()
        self.temp_dir = tempfile.mkdtemp()
        self.process = psutil.Process(os.getpid())
    
    def tearDown(self):
        """Cleanup efter varje test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        gc.collect()  # Force garbage collection
    
    def test_t037_memory_efficient_streaming_basic(self):
        """
        T037: Test Memory-Efficient Streaming
        Verifiera att TFRecord skrivs streamat utan minnesöverskridning
        """
        # Arrange
        # Skapa stora datasets för att testa memory efficiency
        n_windows = 1000  # 1000 windows
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_memory_efficient")
        
        # Mät minnesanvändning före
        mem_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Mät minnesanvändning efter
        mem_after = self.process.memory_info().rss / (1024 * 1024)  # MB
        mem_increase = mem_after - mem_before
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        self.assertTrue(os.path.exists(tfrecord_path), "TFRecord fil ska existera på disk")
        
        # Verifiera att minnesökning är rimlig (< 100MB för 1000 windows)
        self.assertLess(mem_increase, 100, 
                       f"Minnesökning ska vara < 100MB, fick {mem_increase:.2f}MB")
        
        # Verifiera att fil har korrekt storlek
        file_size = os.path.getsize(tfrecord_path)
        self.assertGreater(file_size, 0, "TFRecord fil ska ha storlek > 0")
        
        print(f"✅ T037 PASSED: Basic memory-efficient streaming fungerar korrekt")
        print(f"   Minnesökning: {mem_increase:.2f}MB för {n_windows} windows")
        print(f"   Fil storlek: {file_size / (1024 * 1024):.2f}MB")
    
    def test_t037_memory_efficient_streaming_large_dataset(self):
        """
        Verifiera memory efficiency med mycket stora datasets
        """
        # Arrange
        # Skapa mycket stora datasets
        n_windows = 5000  # 5000 windows
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_large_dataset")
        
        # Mät minnesanvändning före
        mem_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Mät minnesanvändning efter
        mem_after = self.process.memory_info().rss / (1024 * 1024)  # MB
        mem_increase = mem_after - mem_before
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas för stora datasets")
        
        # Verifiera att minnesökning är rimlig även för stora datasets
        # För 5000 windows förväntar vi oss högre minnesanvändning men fortfarande kontrollerad
        self.assertLess(mem_increase, 500, 
                       f"Minnesökning för stora datasets ska vara < 500MB, fick {mem_increase:.2f}MB")
        
        # Verifiera att fil kan läsas tillbaka
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, 
                        f"Ska läsa {n_windows} examples från stora dataset")
        
        print(f"✅ T037 PASSED: Large dataset memory-efficient streaming fungerar korrekt")
        print(f"   Minnesökning: {mem_increase:.2f}MB för {n_windows} windows")
    
    def test_t037_memory_efficient_streaming_memory_monitoring(self):
        """
        Verifiera att minnesanvändning övervakas under streaming
        """
        # Arrange
        n_windows = 2000
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_memory_monitoring")
        
        # Mät minnesanvändning vid olika tidpunkter
        mem_start = self.process.memory_info().rss / (1024 * 1024)
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        mem_end = self.process.memory_info().rss / (1024 * 1024)
        mem_total_increase = mem_end - mem_start
        
        # Assert
        # Verifiera att minnesanvändning är kontrollerad
        self.assertLess(mem_total_increase, 200, 
                       f"Total minnesökning ska vara < 200MB, fick {mem_total_increase:.2f}MB")
        
        # Verifiera att fil skapas korrekt
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att data kan läsas tillbaka
        parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, 
                        f"Ska läsa {n_windows} examples")
        
        print(f"✅ T037 PASSED: Memory monitoring fungerar korrekt")
        print(f"   Total minnesökning: {mem_total_increase:.2f}MB")
    
    def test_t037_memory_efficient_streaming_garbage_collection(self):
        """
        Verifiera att garbage collection fungerar korrekt
        """
        # Arrange
        n_windows = 1000
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_garbage_collection")
        
        # Mät minnesanvändning före
        mem_before = self.process.memory_info().rss / (1024 * 1024)
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Force garbage collection
        del windows_data, static_data, targets_data
        gc.collect()
        
        # Mät minnesanvändning efter garbage collection
        mem_after_gc = self.process.memory_info().rss / (1024 * 1024)
        mem_after_gc_increase = mem_after_gc - mem_before
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att minnesanvändning är rimlig efter garbage collection
        self.assertLess(mem_after_gc_increase, 50, 
                       f"Minnesökning efter GC ska vara < 50MB, fick {mem_after_gc_increase:.2f}MB")
        
        print(f"✅ T037 PASSED: Garbage collection fungerar korrekt")
        print(f"   Minnesökning efter GC: {mem_after_gc_increase:.2f}MB")
    
    def test_t037_memory_efficient_streaming_multiple_files(self):
        """
        Verifiera att multiple TFRecord filer kan skapas utan minnesöverskridning
        """
        # Arrange
        n_windows = 1000
        output_path = os.path.join(self.temp_dir, "test_multiple_files")
        
        # Mät minnesanvändning före
        mem_before = self.process.memory_info().rss / (1024 * 1024)
        
        # Act - Skapa flera TFRecord filer
        tfrecord_paths = []
        for i in range(3):  # Skapa 3 filer
            windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
            static_data = np.random.randn(n_windows, 6).astype(np.float32)
            targets_data = np.random.randn(n_windows, 8).astype(np.float32)
            
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                windows_data, static_data, targets_data, 
                f"{output_path}_file_{i}", f"test_{i}"
            )
            tfrecord_paths.append(tfrecord_path)
            
            # Force garbage collection mellan filer
            del windows_data, static_data, targets_data
            gc.collect()
        
        # Mät minnesanvändning efter
        mem_after = self.process.memory_info().rss / (1024 * 1024)
        mem_total_increase = mem_after - mem_before
        
        # Assert
        # Verifiera att alla filer skapas
        self.assertEqual(len(tfrecord_paths), 3, "Ska skapa 3 TFRecord filer")
        for i, tfrecord_path in enumerate(tfrecord_paths):
            self.assertIsNotNone(tfrecord_path, f"TFRecord fil {i} ska skapas")
            self.assertTrue(os.path.exists(tfrecord_path), f"TFRecord fil {i} ska existera")
        
        # Verifiera att minnesanvändning är kontrollerad även för multiple filer
        self.assertLess(mem_total_increase, 150, 
                       f"Minnesökning för multiple filer ska vara < 150MB, fick {mem_total_increase:.2f}MB")
        
        print(f"✅ T037 PASSED: Multiple files memory-efficient streaming fungerar korrekt")
        print(f"   Minnesökning för 3 filer: {mem_total_increase:.2f}MB")
    
    def test_t037_memory_efficient_streaming_custom_config(self):
        """
        Verifiera memory efficiency med custom configuration
        """
        # Arrange
        # Skapa custom config med olika buffer size
        config = TFRecordConfig(
            timeseries_shape=(300, 16),
            static_shape=(6,),
            target_shape=(8,),
            compression_type="GZIP",
            buffer_size=500  # Mindre buffer size
        )
        
        tfrecord_creator_custom = MasterPOCTFRecordCreator(config)
        
        n_windows = 1000
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_custom_config")
        
        # Mät minnesanvändning före
        mem_before = self.process.memory_info().rss / (1024 * 1024)
        
        # Act
        tfrecord_path = tfrecord_creator_custom.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        # Mät minnesanvändning efter
        mem_after = self.process.memory_info().rss / (1024 * 1024)
        mem_increase = mem_after - mem_before
        
        # Assert
        # Verifiera att custom config fungerar
        self.assertIsNotNone(tfrecord_path, "Custom config ska skapa TFRecord fil")
        
        # Verifiera att minnesanvändning är kontrollerad med custom config
        self.assertLess(mem_increase, 100, 
                       f"Custom config minnesökning ska vara < 100MB, fick {mem_increase:.2f}MB")
        
        # Verifiera att fil kan läsas
        parsed_data = tfrecord_creator_custom.read_tfrecord_file(tfrecord_path)
        self.assertEqual(len(parsed_data), n_windows, 
                        f"Custom config ska läsa {n_windows} examples")
        
        print(f"✅ T037 PASSED: Custom config memory-efficient streaming fungerar korrekt")
        print(f"   Minnesökning med custom config: {mem_increase:.2f}MB")
    
    def test_t037_memory_efficient_streaming_performance(self):
        """
        Verifiera att memory-efficient streaming har bra prestanda
        """
        # Arrange
        import time
        
        n_windows = 2000
        windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
        static_data = np.random.randn(n_windows, 6).astype(np.float32)
        targets_data = np.random.randn(n_windows, 8).astype(np.float32)
        
        output_path = os.path.join(self.temp_dir, "test_performance")
        
        # Mät tid och minne
        start_time = time.time()
        mem_before = self.process.memory_info().rss / (1024 * 1024)
        
        # Act
        tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
            windows_data, static_data, targets_data, output_path, "test"
        )
        
        end_time = time.time()
        mem_after = self.process.memory_info().rss / (1024 * 1024)
        
        processing_time = end_time - start_time
        mem_increase = mem_after - mem_before
        
        # Assert
        # Verifiera att fil skapas
        self.assertIsNotNone(tfrecord_path, "TFRecord fil ska skapas")
        
        # Verifiera att prestanda är rimlig (< 10 sekunder för 2000 windows)
        self.assertLess(processing_time, 10, 
                       f"Processing time ska vara < 10s, fick {processing_time:.2f}s")
        
        # Verifiera att minnesanvändning är kontrollerad
        self.assertLess(mem_increase, 200, 
                       f"Minnesökning ska vara < 200MB, fick {mem_increase:.2f}MB")
        
        # Beräkna throughput
        throughput = n_windows / processing_time
        self.assertGreater(throughput, 100, 
                          f"Throughput ska vara > 100 windows/s, fick {throughput:.2f} windows/s")
        
        print(f"✅ T037 PASSED: Performance memory-efficient streaming fungerar korrekt")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} windows/s")
        print(f"   Minnesökning: {mem_increase:.2f}MB")
    
    def test_t037_memory_efficient_streaming_comprehensive(self):
        """
        Omfattande test av memory-efficient streaming
        """
        # Arrange
        # Testa med olika dataset storlekar
        test_cases = [
            (500, "Small dataset"),
            (1000, "Medium dataset"),
            (2000, "Large dataset"),
        ]
        
        output_path = os.path.join(self.temp_dir, "test_comprehensive")
        
        # Act & Assert
        for n_windows, description in test_cases:
            windows_data = np.random.randn(n_windows, 300, 16).astype(np.float32)
            static_data = np.random.randn(n_windows, 6).astype(np.float32)
            targets_data = np.random.randn(n_windows, 8).astype(np.float32)
            
            # Mät minnesanvändning före
            mem_before = self.process.memory_info().rss / (1024 * 1024)
            
            # Skapa TFRecord
            tfrecord_path = self.tfrecord_creator.create_memory_efficient_tfrecord(
                windows_data, static_data, targets_data, 
                f"{output_path}_{n_windows}", f"test_{n_windows}"
            )
            
            # Mät minnesanvändning efter
            mem_after = self.process.memory_info().rss / (1024 * 1024)
            mem_increase = mem_after - mem_before
            
            # Verifiera att fil skapas
            self.assertIsNotNone(tfrecord_path, f"{description} ska skapa TFRecord fil")
            
            # Verifiera att minnesanvändning är kontrollerad
            max_memory = n_windows * 0.1  # Max 0.1MB per window
            self.assertLess(mem_increase, max_memory, 
                           f"{description} minnesökning ska vara < {max_memory:.2f}MB, fick {mem_increase:.2f}MB")
            
            # Verifiera att fil kan läsas
            parsed_data = self.tfrecord_creator.read_tfrecord_file(tfrecord_path)
            self.assertEqual(len(parsed_data), n_windows, 
                           f"{description} ska läsa {n_windows} examples")
            
            # Cleanup
            del windows_data, static_data, targets_data
            gc.collect()
        
        print(f"✅ T037 PASSED: Comprehensive memory-efficient streaming test fungerar korrekt")

if __name__ == '__main__':
    unittest.main()
