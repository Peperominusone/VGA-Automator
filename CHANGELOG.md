# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-01-28

### Added
- Initial release of VGA-Automator
- Segmentation-based wall extraction using YOLO segmentation models
- Morphological operations for connecting broken wall segments
- Skeletonization for extracting centerlines from wall masks
- Polyline merging algorithm to connect nearby line segments
- DXF export with proper layer organization
- Support for multiple architectural elements:
  - Walls (exported as polylines)
  - Doors (exported as contours)
  - Windows (exported as contours)
  - Columns (exported as contours)
  - Sliding Doors (exported as contours)
  - Stairs (exported as contours)
  - Curtain Walls (exported as contours)
  - Railings (exported as contours)
- CLI tool with configurable parameters:
  - Model path selection
  - Confidence threshold adjustment
  - Gap size for segment connection
  - Debug image output
- Comprehensive test suite
- API examples and documentation
- Security analysis with CodeQL

### Features
- **Pixel-level Accuracy**: Uses segmentation masks instead of bounding boxes
- **Gap Connection**: Automatically connects broken wall segments
- **Centerline Extraction**: Skeletonization produces clean wall centerlines
- **Smart Merging**: Connects nearby polyline endpoints
- **Layer Organization**: Proper DXF layer structure with color coding
- **Debug Mode**: Visual output for algorithm verification

### Documentation
- Comprehensive README with usage examples
- Security summary document
- API usage examples
- Inline code documentation in English

### Testing
- Unit tests for core functionality
- Validation tests for all major components
- CodeQL security analysis (0 vulnerabilities)

### Dependencies
- ultralytics>=8.0.0 (YOLO segmentation)
- opencv-python>=4.8.0 (Image processing)
- numpy>=1.24.0 (Numerical operations)
- scikit-image>=0.21.0 (Skeletonization)
- ezdxf>=1.1.0 (DXF export)
