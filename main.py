#!/usr/bin/env python3
"""
VGA-Automator: Unified CLI for Floor Plan to DXF Conversion and Model Training
"""

import argparse
import sys
from pathlib import Path


def cmd_infer(args):
    """Run inference to convert floor plan to DXF"""
    import cv2
    from src.preprocessing.preprocessor import Preprocessor
    from src.detection.segmentation_detector import ContinuousWallExtractor, ElementType
    from src.export.dxf_exporter_continuous import DXFExporterContinuous
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"   Please download or train a model and place it at: {model_path}")
        print(f"   Or specify a different model with --model <path>")
        sys.exit(1)
    
    print("=" * 60)
    print("VGA-Automator: Floor Plan to DXF Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {model_path}")
    print()
    
    try:
        # 1. Load and preprocess image
        print("[1/3] Loading and preprocessing image...")
        preprocessor = Preprocessor()
        image = preprocessor.load_image(str(input_path))
        preprocessed = preprocessor.preprocess(str(input_path))
        print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
        print("   ✓ Preprocessing complete")
        
        # 2. Extract elements
        print(f"[2/3] Detecting elements (confidence: {args.confidence})...")
        extractor = ContinuousWallExtractor()
        elements = extractor.extract_all_elements(
            image,
            preprocessed['binary'],
            model_path=str(model_path),
            confidence=args.confidence,
            gap_size=args.gap
        )
        
        # Result summary
        print("   Detected elements:")
        for elem_type, element in elements.items():
            if elem_type == ElementType.WALL:
                count = len(element.polylines)
                print(f"      - {elem_type.value}: {count} polylines")
            else:
                count = len(element.contours)
                print(f"      - {elem_type.value}: {count} contours")
        
        if not elements:
            print("   ⚠️  No elements detected.")
        
        print("   ✓ Detection complete")
        
        # 3. Export to DXF
        print(f"[3/3] Exporting to DXF...")
        exporter = DXFExporterContinuous(str(output_path))
        exporter.export_elements(elements, image_height=image.shape[0])
        exporter.save()
        print("   ✓ DXF export complete")
        
        # 4. Save debug images if requested
        if args.debug:
            debug_dir = output_path.parent / f"{output_path.stem}_debug"
            print(f"\n   Saving debug images to: {debug_dir}")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for elem_type, element in elements.items():
                # Save mask
                mask_path = debug_dir / f"{elem_type.value}_mask.png"
                cv2.imwrite(str(mask_path), element.mask)
            print("   ✓ Debug images saved")
        
        print()
        print("=" * 60)
        print(f"✅ SUCCESS: DXF file created at {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_train(args):
    """Train YOLOv8 segmentation model"""
    from ultralytics import YOLO
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ Error: Data YAML file not found: {data_yaml}")
        print("\n   To prepare training data, run:")
        print(f"   python main.py convert --cubicasa_root <path> --out_root training/data/yolo")
        sys.exit(1)
    
    print("=" * 60)
    print("VGA-Automator: Model Training")
    print("=" * 60)
    print(f"Data:    {data_yaml}")
    print(f"Model:   {args.model}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"ImgSize: {args.imgsz}")
    print()
    
    try:
        model = YOLO(args.model)
        results = model.train(
            data=str(data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"   Weights saved to: {args.project}/{args.name}/weights/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_convert(args):
    """Convert CubiCasa5k dataset to YOLO-seg format"""
    import json
    import yaml
    from pathlib import Path
    from tqdm import tqdm
    
    cubicasa_root = Path(args.cubicasa_root)
    out_root = Path(args.out_root)
    
    if not cubicasa_root.exists():
        print(f"❌ Error: CubiCasa5k directory not found: {cubicasa_root}")
        sys.exit(1)
    
    print("=" * 60)
    print("VGA-Automator: Dataset Conversion")
    print("=" * 60)
    print(f"Input:  {cubicasa_root}")
    print(f"Output: {out_root}")
    print()
    
    # Load class configuration
    class_config_path = Path(__file__).parent / "training" / "configs" / "classes.json"
    if not class_config_path.exists():
        print(f"❌ Error: Class config not found: {class_config_path}")
        sys.exit(1)
    
    with open(class_config_path, 'r') as f:
        class_config = json.load(f)
    
    print("⚠️  Note: This is a skeleton converter.")
    print("   CubiCasa5k SVG parsing needs to be implemented in:")
    print(f"   {Path(__file__).parent / 'training' / 'scripts' / 'convert_cubicasa_to_yolo_seg.py'}")
    print()
    print("   The converter will create the directory structure and data.yaml,")
    print("   but you need to implement SVG parsing for your specific dataset.")
    print()
    
    # Create output directories
    for split in args.splits:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {out_root / 'images' / split}")
        print(f"✓ Created directory: {out_root / 'labels' / split}")
    
    # Create data.yaml
    data_yaml = {
        "path": str(out_root.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_config["names"])}
    }
    
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    
    print(f"\n✓ Created data.yaml: {yaml_path}")
    print("\n" + "=" * 60)
    print("✅ Directory structure created!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement SVG parsing in training/scripts/convert_cubicasa_to_yolo_seg.py")
    print("2. Run the full conversion script")
    print("3. Verify the generated labels in the output directory")


def main():
    parser = argparse.ArgumentParser(
        prog='VGA-Automator',
        description='Floor Plan to DXF Converter with Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference (convert floor plan to DXF)
  python main.py infer floorplan.png -o output.dxf
  python main.py infer floorplan.png -o output.dxf --model models/best.pt --confidence 0.6
  
  # Training (train YOLOv8-seg model)
  python main.py train --data training/data/yolo/data.yaml --epochs 100
  python main.py train --data training/data/yolo/data.yaml --model yolov8m-seg.pt --batch 16
  
  # Dataset conversion (CubiCasa5k to YOLO format)
  python main.py convert --cubicasa_root data/raw/CubiCasa5k --out_root training/data/yolo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Infer command
    infer_parser = subparsers.add_parser(
        'infer',
        help='Convert floor plan image to DXF',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    infer_parser.add_argument('input', help='Input floor plan image (PNG, JPG, PDF)')
    infer_parser.add_argument('-o', '--output', default='output.dxf', help='Output DXF file (default: output.dxf)')
    infer_parser.add_argument('--model', default='models/best.pt', help='YOLOv8-seg model file (default: models/best.pt)')
    infer_parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence (default: 0.5)')
    infer_parser.add_argument('--gap', type=int, default=10, help='Gap size for connecting segments (default: 10)')
    infer_parser.add_argument('--debug', action='store_true', help='Save debug images')
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train YOLOv8 segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    train_parser.add_argument('--data', required=True, help='Path to data.yaml')
    train_parser.add_argument('--model', default='yolov8n-seg.pt', help='Base model (default: yolov8n-seg.pt)')
    train_parser.add_argument('--imgsz', type=int, default=1024, help='Image size (default: 1024)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    train_parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8)')
    train_parser.add_argument('--device', default=None, help='Device (e.g., 0 or cpu)')
    train_parser.add_argument('--project', default='training/runs/segment', help='Project directory')
    train_parser.add_argument('--name', default='vga_yolov8seg', help='Run name')
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert CubiCasa5k to YOLO-seg format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    convert_parser.add_argument('--cubicasa_root', required=True, help='CubiCasa5k root directory')
    convert_parser.add_argument('--out_root', required=True, help='Output directory for YOLO-seg dataset')
    convert_parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='Dataset splits (default: train val)')
    
    args = parser.parse_args()
    
    if args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'convert':
        cmd_convert(args)
    else:
        parser.print_help()
        print("\n❌ Error: Please specify a command (infer, train, or convert)")
        sys.exit(1)


if __name__ == '__main__':
    main()
