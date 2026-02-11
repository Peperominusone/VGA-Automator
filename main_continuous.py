#!/usr/bin/env python3
"""
Connected wall lines CLI - Segmentation-based floorplan conversion
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from src.detection.segmentation_detector import ContinuousWallExtractor, ElementType
from src.export.dxf_exporter_continuous import DXFExporterContinuous


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Image preprocessing - generate binary image"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return image, binary


def save_debug_images(elements: dict, output_dir: Path, image_shape: tuple):
    """Save debug mask images"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for elem_type, element in elements.items():
        # Save mask
        mask_path = output_dir / f"debug_{elem_type.value}_mask.png"
        cv2.imwrite(str(mask_path), element.mask)
        
        # Save skeleton (walls only)
        if elem_type == ElementType.WALL and element.skeleton is not None:
            skeleton_path = output_dir / f"debug_{elem_type.value}_skeleton.png"
            cv2.imwrite(str(skeleton_path), element.skeleton)
        
        # Visualize polylines (walls only)
        if elem_type == ElementType.WALL and element.polylines:
            polyline_img = np.zeros(image_shape[:2], dtype=np.uint8)
            for polyline in element.polylines:
                pts = np.array(polyline, dtype=np.int32)
                cv2.polylines(polyline_img, [pts], False, 255, 2)
            polyline_path = output_dir / f"debug_{elem_type.value}_polylines.png"
            cv2.imwrite(str(polyline_path), polyline_img)
        
        # Visualize contours (other elements)
        if elem_type != ElementType.WALL and element.contours:
            contour_img = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_img, element.contours, -1, 255, 2)
            contour_path = output_dir / f"debug_{elem_type.value}_contours.png"
            cv2.imwrite(str(contour_path), contour_img)


def main():
    parser = argparse.ArgumentParser(
        description='Extract connected wall lines - Segmentation-based floorplan conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s floorplan.png -o output.dxf
  %(prog)s floorplan.png -o output.dxf --gap 15 --debug
  %(prog)s floorplan.png -o output.dxf --model custom_model.pt --confidence 0.6
        """
    )
    
    parser.add_argument('input', help='Input floorplan image path')
    parser.add_argument('-o', '--output', required=True, help='Output DXF file path')
    parser.add_argument('--model', default='best.pt', help='YOLO model file path (default: best.pt)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--gap', type=int, default=10, help='Gap size for connecting segments (default: 10)')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    
    args = parser.parse_args()
    
    # Validate file paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file does not exist: {model_path}", file=sys.stderr)
        print(f"   Please provide a YOLO segmentation model file (.pt)", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output)
    
    try:
        # 1. Image preprocessing
        print(f"📄 Loading image: {input_path}")
        image, binary = preprocess_image(str(input_path))
        print(f"   Size: {image.shape[1]}x{image.shape[0]}")
        
        # 2. Extract elements
        print(f"🔍 Detecting elements (model: {model_path}, confidence: {args.confidence})")
        extractor = ContinuousWallExtractor()
        elements = extractor.extract_all_elements(
            image, binary, 
            model_path=str(model_path), 
            confidence=args.confidence
        )
        
        # Result summary
        print(f"\n✓ Detected elements:")
        for elem_type, element in elements.items():
            if elem_type == ElementType.WALL:
                count = len(element.polylines)
                print(f"   - {elem_type.value}: {count} polylines")
            else:
                count = len(element.contours)
                print(f"   - {elem_type.value}: {count} contours")
        
        if not elements:
            print("⚠️  No elements detected.", file=sys.stderr)
            sys.exit(0)
        
        # 3. Export to DXF
        print(f"\n💾 Exporting to DXF: {output_path}")
        exporter = DXFExporterContinuous(str(output_path))
        exporter.export_elements(elements, image_height=image.shape[0])
        exporter.save()
        
        # 4. Save debug images
        if args.debug:
            debug_dir = output_path.parent / f"{output_path.stem}_debug"
            print(f"\n🔧 Saving debug images: {debug_dir}")
            save_debug_images(elements, debug_dir, image.shape)
            print(f"   ✓ Debug images saved")
        
        print(f"\n✅ Conversion complete!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
