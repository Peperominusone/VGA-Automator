#!/usr/bin/env python3
"""
VGA-Automator: Floor Plan to DXF Converter
Main CLI entry point
"""

import argparse
import sys
from pathlib import Path
import cv2

from src.preprocessing.preprocessor import Preprocessor
from src.detection.detector import FloorPlanDetector
from src.postprocessing.contour_extractor import ContourExtractor
from src.export.dxf_exporter import DXFExporter


def main():
    parser = argparse.ArgumentParser(
        description='Convert floor plan images to DXF format for VGA analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py floorplan.png -o output.dxf
  python main.py floorplan.jpg -o output.dxf --confidence 0.5 --debug
  python main.py floorplan.pdf -o output.dxf --model best.pt --scale 10.0
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input',
        type=str,
        help='Input floor plan image (PNG, JPG, or PDF)'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output.dxf',
        help='Output DXF file path (default: output.dxf)'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='best.pt',
        help='Path to YOLOv8 model file (default: best.pt)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    # Processing options
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor from pixels to CAD units (default: 1.0)'
    )
    
    parser.add_argument(
        '--denoise',
        type=int,
        default=10,
        help='Denoising strength (default: 10)'
    )
    
    parser.add_argument(
        '--doors-as-walls',
        action='store_true',
        help='Treat doors as walls instead of openings (VGA impassable)'
    )
    
    parser.add_argument(
        '--bbox-only',
        action='store_true',
        help='Use bounding boxes instead of detailed contours'
    )
    
    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save intermediate debug images'
    )
    
    parser.add_argument(
        '--no-detection',
        action='store_true',
        help='Skip object detection, only preprocess and extract contours'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    print("=" * 60)
    print("VGA-Automator: Floor Plan to DXF Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    try:
        # Step 1: Preprocessing
        print("[1/4] Preprocessing image...")
        preprocessor = Preprocessor()
        processed_image = preprocessor.preprocess(
            input_path,
            denoise_strength=args.denoise
        )
        
        if args.debug:
            debug_dir = Path('debug_output')
            debug_dir.mkdir(exist_ok=True)
            preprocessor.save_image(processed_image, debug_dir / '1_preprocessed.png')
            print(f"  Saved preprocessed image to {debug_dir / '1_preprocessed.png'}")
        
        print("  ✓ Preprocessing complete")
        
        # Get original image for detection
        original_image = preprocessor.original_image
        
        if args.no_detection:
            # Skip detection, just extract contours from preprocessed image
            print("[2/4] Object detection skipped")
            print("[3/4] Extracting contours from preprocessed image...")
            
            extractor = ContourExtractor()
            contours = extractor.find_contours(processed_image, min_area=100)
            simplified = extractor.simplify_contours(contours)
            lines = extractor.extract_lines_from_contours(simplified, min_length=10)
            
            print(f"  Extracted {len(lines)} lines")
            
            if args.debug:
                vis = extractor.visualize_lines(original_image, lines)
                cv2.imwrite(str(debug_dir / '3_lines.png'), vis)
                print(f"  Saved line visualization to {debug_dir / '3_lines.png'}")
            
            print("  ✓ Contour extraction complete")
            
            # Export to DXF
            print("[4/4] Exporting to DXF...")
            exporter = DXFExporter(
                image_height=original_image.shape[0],
                scale_factor=args.scale
            )
            exporter.export_from_lines(
                lines,
                original_image.shape[0],
                output_path
            )
            
        else:
            # Step 2: Object Detection
            print("[2/4] Detecting objects with YOLOv8...")
            
            # Check if model file exists
            model_path = Path(args.model)
            if not model_path.exists():
                print(f"\nError: Model file not found: {model_path}")
                print("\nPlease download the model from:")
                print("https://github.com/sanatladkat/floor-plan-object-detection")
                print("\nOr use --no-detection flag to skip object detection")
                sys.exit(1)
            
            detector = FloorPlanDetector(
                model_path=args.model,
                confidence=args.confidence
            )
            
            detections = detector.detect(original_image)
            
            print(f"  Detected {len(detections)} objects:")
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                print(f"    - {class_name}: {count}")
            
            if args.debug:
                vis = detector.visualize_detections(original_image)
                cv2.imwrite(str(debug_dir / '2_detections.png'), vis)
                print(f"  Saved detection visualization to {debug_dir / '2_detections.png'}")
            
            print("  ✓ Detection complete")
            
            # Step 3: Extract contours/lines (optional, for additional detail)
            print("[3/4] Extracting additional contours...")
            
            # Extract lines from wall detections
            wall_detections = detector.filter_detections_by_class(['Wall', 'Curtain Wall'])
            lines = detector.extract_lines_from_walls(
                processed_image,
                wall_detections,
                threshold=50,
                min_line_length=30,
                max_line_gap=10
            )
            
            print(f"  Extracted {len(lines)} wall lines")
            
            if args.debug and lines:
                extractor = ContourExtractor()
                vis = extractor.visualize_lines(original_image, lines)
                cv2.imwrite(str(debug_dir / '3_wall_lines.png'), vis)
                print(f"  Saved wall lines to {debug_dir / '3_wall_lines.png'}")
            
            print("  ✓ Contour extraction complete")
            
            # Step 4: Export to DXF
            print("[4/4] Exporting to DXF...")
            exporter = DXFExporter(
                image_height=original_image.shape[0],
                scale_factor=args.scale
            )
            
            # Export detections
            exporter.export_from_detections(
                detections,
                original_image,
                output_path,
                doors_as_openings=not args.doors_as_walls,
                as_bbox=args.bbox_only
            )
            
            # Optionally add extracted lines to the same DXF
            # (could be added to a separate layer if needed)
            
        print("  ✓ DXF export complete")
        print()
        print("=" * 60)
        print(f"✓ SUCCESS: DXF file created at {output_path}")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Open the DXF file in AutoCAD or similar CAD software")
        print("  2. Import to depthmapX or other VGA analysis tool")
        print("  3. Run Visibility Graph Analysis")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
