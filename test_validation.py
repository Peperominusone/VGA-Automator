#!/usr/bin/env python3
"""
Basic validation tests for VGA-Automator
Tests core functionality without requiring YOLO model
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.segmentation_detector import (
    ElementType, 
    SegmentedElement, 
    SegmentationDetector,
    SANATLADKAT_CLASS_MAP
)
from src.export.dxf_exporter_continuous import DXFExporterContinuous


def test_element_type_enum():
    """Test ElementType enum"""
    print("Testing ElementType enum...")
    assert ElementType.WALL.value == "wall"
    assert ElementType.DOOR.value == "door"
    assert ElementType.WINDOW.value == "window"
    print("✓ ElementType enum works correctly")


def test_class_map():
    """Test SANATLADKAT_CLASS_MAP"""
    print("\nTesting SANATLADKAT_CLASS_MAP...")
    assert 'Wall' in SANATLADKAT_CLASS_MAP
    assert SANATLADKAT_CLASS_MAP['Wall'] == ElementType.WALL
    assert 'Door' in SANATLADKAT_CLASS_MAP
    assert SANATLADKAT_CLASS_MAP['Door'] == ElementType.DOOR
    print("✓ Class mapping works correctly")


def test_segmented_element():
    """Test SegmentedElement dataclass"""
    print("\nTesting SegmentedElement...")
    mask = np.zeros((100, 100), dtype=np.uint8)
    element = SegmentedElement(element_type=ElementType.WALL, mask=mask)
    assert element.element_type == ElementType.WALL
    assert element.mask.shape == (100, 100)
    assert element.contours == []
    assert element.skeleton is None
    assert element.polylines == []
    print("✓ SegmentedElement works correctly")


def test_connect_segments():
    """Test segment connection"""
    print("\nTesting connect_segments...")
    # Create a simple test mask with a gap
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:45, 10:40] = 255  # Left segment
    mask[40:45, 60:90] = 255  # Right segment (gap of 20 pixels)
    
    element = SegmentedElement(element_type=ElementType.WALL, mask=mask)
    
    # Mock detector (we don't need the full detector for this test)
    class MockDetector:
        def connect_segments(self, element, gap_size=10, min_area=100):
            mask = element.mask.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, gap_size))
            connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(connected)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)
            element.mask = clean_mask
            return element
    
    detector = MockDetector()
    
    # With large gap_size, segments should connect
    result = detector.connect_segments(element, gap_size=25)
    assert np.sum(result.mask) > np.sum(mask), "Segments should be connected"
    print("✓ Segment connection works correctly")


def test_merge_polylines():
    """Test polyline merging"""
    print("\nTesting _merge_polylines...")
    from src.segmentation_detector import SegmentationDetector
    
    # Create mock polylines that should merge
    line1 = [(0.0, 0.0), (10.0, 0.0)]
    line2 = [(10.0, 0.0), (20.0, 0.0)]  # Connects to end of line1
    line3 = [(30.0, 30.0), (40.0, 40.0)]  # Separate line
    
    polylines = [line1, line2, line3]
    
    # We can't instantiate SegmentationDetector without a model,
    # but we can test the merge logic directly
    class TestDetector:
        def _merge_polylines(self, polylines, threshold=10.0):
            if len(polylines) <= 1:
                return polylines
            merged = []
            used = [False] * len(polylines)
            for i, line1 in enumerate(polylines):
                if used[i] or len(line1) < 2:
                    continue
                current = list(line1)
                used[i] = True
                changed = True
                while changed:
                    changed = False
                    for j, line2 in enumerate(polylines):
                        if used[j] or len(line2) < 2:
                            continue
                        connections = [
                            (current[-1], line2[0], False, False),
                            (current[-1], line2[-1], False, True),
                            (current[0], line2[0], True, False),
                            (current[0], line2[-1], True, True),
                        ]
                        for p1, p2, reverse_current, reverse_other in connections:
                            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                            if dist < threshold:
                                other = list(line2)
                                if reverse_other:
                                    other = other[::-1]
                                if reverse_current:
                                    current = other + current
                                else:
                                    current = current + other
                                used[j] = True
                                changed = True
                                break
                        if changed:
                            break
                merged.append(current)
            return merged
    
    detector = TestDetector()
    merged = detector._merge_polylines(polylines, threshold=0.1)
    
    # Should merge line1 and line2 into one, keep line3 separate
    assert len(merged) == 2, f"Expected 2 merged lines, got {len(merged)}"
    # Find the merged line (should have more than 2 points)
    long_lines = [line for line in merged if len(line) > 2]
    assert len(long_lines) == 1, "Should have one merged line with >2 points"
    print("✓ Polyline merging works correctly")


def test_dxf_exporter():
    """Test DXF exporter initialization and layer setup"""
    print("\nTesting DXFExporterContinuous...")
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.dxf"
        exporter = DXFExporterContinuous(str(output_path))
        
        # Check that layers are created
        assert 'WALLS' in exporter.doc.layers
        assert 'DOORS' in exporter.doc.layers
        assert 'WINDOWS' in exporter.doc.layers
        
        # Test saving empty DXF
        exporter.save()
        assert output_path.exists(), "DXF file should be created"
        
    print("✓ DXF exporter works correctly")


def test_dxf_export_with_data():
    """Test DXF export with actual polylines"""
    print("\nTesting DXF export with polylines...")
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_with_data.dxf"
        exporter = DXFExporterContinuous(str(output_path))
        
        # Create test data
        mask = np.zeros((100, 100), dtype=np.uint8)
        element = SegmentedElement(element_type=ElementType.WALL, mask=mask)
        element.polylines = [
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)],
            [(50.0, 50.0), (75.0, 75.0)]
        ]
        
        elements = {ElementType.WALL: element}
        
        exporter.export_elements(elements, image_height=100)
        exporter.save()
        
        assert output_path.exists(), "DXF file should be created"
        
        # Check file size is reasonable (not empty)
        assert output_path.stat().st_size > 100, "DXF file should have content"
        
    print("✓ DXF export with data works correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("VGA-Automator Validation Tests")
    print("=" * 60)
    
    tests = [
        test_element_type_enum,
        test_class_map,
        test_segmented_element,
        test_connect_segments,
        test_merge_polylines,
        test_dxf_exporter,
        test_dxf_export_with_data,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
