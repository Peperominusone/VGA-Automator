#!/usr/bin/env python3
"""
Example usage of VGA-Automator library
This demonstrates how to use the API programmatically
"""
import numpy as np
import cv2
from pathlib import Path

from src.segmentation_detector import (
    ContinuousWallExtractor,
    SegmentationDetector,
    ElementType,
)
from src.dxf_exporter_continuous import DXFExporterContinuous


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Load image
    image_path = "floorplan.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Note: Example image '{image_path}' not found")
        print("This is a demonstration of API usage")
        return
    
    # Preprocess to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Extract elements
    extractor = ContinuousWallExtractor()
    elements = extractor.extract_all_elements(
        image, binary,
        model_path="best.pt",
        confidence=0.5
    )
    
    # Export to DXF
    exporter = DXFExporterContinuous("output.dxf")
    exporter.export_elements(elements, image_height=image.shape[0])
    exporter.save()
    
    print("✓ Conversion complete!")


def example_custom_processing():
    """Example with custom processing"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Processing")
    print("=" * 60)
    
    # Create a simple test image
    print("Creating synthetic test image...")
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    binary = np.zeros((500, 500), dtype=np.uint8)
    
    # Draw some wall-like structures
    cv2.rectangle(binary, (50, 50), (450, 70), 255, -1)  # Horizontal wall
    cv2.rectangle(binary, (50, 50), (70, 450), 255, -1)  # Vertical wall
    cv2.rectangle(binary, (50, 430), (450, 450), 255, -1)  # Bottom wall
    cv2.rectangle(binary, (430, 70), (450, 430), 255, -1)  # Right wall
    
    # Note: This example shows the API structure
    # In real use, you would have a trained YOLO model
    print("Note: Requires a trained YOLO segmentation model")
    print("API usage example only")


def example_accessing_results():
    """Example of accessing detection results"""
    print("\n" + "=" * 60)
    print("Example 3: Accessing Results")
    print("=" * 60)
    
    # This demonstrates how to access the detection results
    # after processing (mock data for demonstration)
    
    from src.segmentation_detector import SegmentedElement
    
    # Create mock result
    mask = np.zeros((100, 100), dtype=np.uint8)
    element = SegmentedElement(element_type=ElementType.WALL, mask=mask)
    element.polylines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(100.0, 0.0), (100.0, 100.0)],
    ]
    
    # Access polylines
    print(f"Element type: {element.element_type.value}")
    print(f"Number of polylines: {len(element.polylines)}")
    
    for i, polyline in enumerate(element.polylines):
        print(f"  Polyline {i + 1}: {len(polyline)} points")
        print(f"    Start: {polyline[0]}")
        print(f"    End: {polyline[-1]}")


def example_custom_export():
    """Example of custom DXF export configuration"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Export")
    print("=" * 60)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "custom_export.dxf"
        
        # Create exporter
        exporter = DXFExporterContinuous(str(output_path))
        
        # Access layer configuration
        print("Available layers:")
        for elem_type, config in exporter.LAYER_CONFIG.items():
            print(f"  {elem_type.value}: {config['name']} (color: {config['color']})")
        
        # Create empty DXF with layers
        exporter.save()
        print(f"\n✓ Empty DXF created at: {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")


def main():
    """Run all examples"""
    print("VGA-Automator API Examples")
    print()
    
    # Run examples
    example_basic_usage()
    example_custom_processing()
    example_accessing_results()
    example_custom_export()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor CLI usage, see:")
    print("  python main_continuous.py --help")


if __name__ == '__main__':
    main()
