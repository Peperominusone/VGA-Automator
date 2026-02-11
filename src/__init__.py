"""
VGA-Automator - Floorplan to DXF converter
"""

from .detection.segmentation_detector import (
    ElementType,
    SegmentedElement,
    SegmentationDetector,
    ContinuousWallExtractor,
)
from .export.dxf_exporter_continuous import DXFExporterContinuous

from .preprocessing.preprocessor import Preprocessor
from .detection.detector import FloorPlanDetector
from .postprocessing.contour_extractor import ContourExtractor
from .export.dxf_exporter import DXFExporter

__all__ = [
    'ElementType',
    'SegmentedElement',
    'SegmentationDetector',
    'ContinuousWallExtractor',
    'DXFExporterContinuous',
    'Preprocessor',
    'FloorPlanDetector',
    'ContourExtractor',
    'DXFExporter',
]

__version__ = "1.0.0"
__author__ = "VGA-Automator Contributors"
