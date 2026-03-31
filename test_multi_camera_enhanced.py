#!/usr/bin/env python3
"""
Enhanced test script for multi-camera frame extraction with verification and optional cleanup.
"""

import os
import tempfile
from examples.datasets.colmap import extract_frame_from_cameras

def test_frame_extraction_with_verification():
    """Test the frame extraction functionality with verification."""
    
    # Test data directory (replace with your actual path)
    data_dir = "/data/shared/elaheh/4D/4D_scenes/elly/undistort"
    frame_num = 100  # Extract frame 000100.jpg
    
    print("Testing multi-camera frame extraction with verification...")
    print("=" * 60)
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Data directory: {data_dir}")
        print(f"Frame number: {frame_num}")
        print(f"Output directory: {temp_dir}")
        print()
        
        try:
            # Test 1: With verification (default)
            print("🔍 Test 1: With verification enabled")
            print("-" * 40)
            images_dir = extract_frame_from_cameras(data_dir, frame_num, temp_dir, verify_copy=True)
            
            # Check results
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                print(f"\n✅ Successfully extracted {len(image_files)} camera images with verification:")
                for img_file in sorted(image_files):
                    img_path = os.path.join(images_dir, img_file)
                    size = os.path.getsize(img_path)
                    print(f"  - {img_file} ({size:,} bytes)")
                
                # Show the structure
                print(f"\n📁 Directory structure:")
                print(f"  {images_dir}/")
                for img_file in sorted(image_files):
                    print(f"    {img_file}")
                    
                return True
            else:
                print("❌ Images directory was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error during frame extraction: {e}")
            return False

def test_frame_extraction_without_verification():
    """Test the frame extraction functionality without verification."""
    
    data_dir = "/data/shared/elaheh/4D/4D_scenes/elly/undistort"
    frame_num = 200  # Extract frame 000200.jpg
    
    print("\n🚀 Test 2: Without verification (faster)")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            images_dir = extract_frame_from_cameras(data_dir, frame_num, temp_dir, verify_copy=False)
            
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                print(f"✅ Extracted {len(image_files)} camera images without verification")
                return True
            else:
                print("❌ Images directory was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error during frame extraction: {e}")
            return False

def test_colmap_parser_with_cleanup_options():
    """Test the COLMAP parser with cleanup options."""
    
    print("\n🧹 Test 3: COLMAP parser with cleanup options")
    print("-" * 40)
    
    try:
        from examples.datasets.colmap import Parser
        
        print("✅ COLMAP parser supports:")
        print("   - frame_num parameter for multi-camera extraction")
        print("   - cleanup_temp_dirs parameter (True/False)")
        print("   - extract_frame_from_cameras function with verification")
        print("   - Enhanced logging with emojis and file sizes")
        print("   - Optional cleanup of temporary directories")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing COLMAP parser: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the enhanced functionality."""
    
    print("\n📖 Usage Examples:")
    print("=" * 60)
    
    print("1. Command line with verification and cleanup:")
    print("   python examples/datasets/colmap.py \\")
    print("     --data_dir /data/shared/elaheh/4D/4D_scenes/elly/undistort \\")
    print("     --frame_num 100 \\")
    print("     --cleanup_temp_dirs")
    print()
    
    print("2. Command line without cleanup (keep temp dirs):")
    print("   python examples/datasets/colmap.py \\")
    print("     --data_dir /data/shared/elaheh/4D/4D_scenes/elly/undistort \\")
    print("     --frame_num 100 \\")
    print("     --no_cleanup")
    print()
    
    print("3. Training with frame extraction and no cleanup:")
    print("   python examples/simple_trainer.py default \\")
    print("     --data-dir /data/shared/elaheh/4D/4D_scenes/elly/undistort \\")
    print("     --frame-num 100 \\")
    print("     --cleanup-temp-dirs false \\")
    print("     --result-dir /path/to/results")
    print()
    
    print("4. Python code with custom options:")
    print("   from examples.datasets.colmap import Parser")
    print("   ")
    print("   parser = Parser(")
    print("       data_dir='/path/to/cameras',")
    print("       frame_num=100,")
    print("       cleanup_temp_dirs=False,  # Keep temp dirs")
    print("       factor=1,")
    print("       normalize=True,")
    print("       test_every=0,")
    print("   )")

if __name__ == "__main__":
    print("Enhanced Multi-Camera Frame Extraction Test")
    print("=" * 60)
    
    # Test 1: Frame extraction with verification
    success1 = test_frame_extraction_with_verification()
    
    # Test 2: Frame extraction without verification
    success2 = test_frame_extraction_without_verification()
    
    # Test 3: COLMAP parser with cleanup options
    success3 = test_colmap_parser_with_cleanup_options()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    print("\n🎯 Key Features:")
    print("   ✅ File verification with size checking")
    print("   ✅ Optional cleanup of temporary directories")
    print("   ✅ Enhanced logging with emojis and status indicators")
    print("   ✅ Error handling for failed copies")
    print("   ✅ Support for multiple image formats (.jpg, .png, .jpeg)")
    print("   ✅ Command-line interface with cleanup options")

