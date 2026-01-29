"""
VGA-Automator - Floorplan to DXF converter
"""

from .segmentation_detector import (
    ElementType,
    SegmentedElement,
    SegmentationDetector,
    ContinuousWallExtractor,
)
from .dxf_exporter_continuous import DXFExporterContinuous

from .preprocessor import Preprocessor
from .detector import FloorPlanDetector
from .contour_extractor import ContourExtractor
from .dxf_exporter import DXFExporter

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
