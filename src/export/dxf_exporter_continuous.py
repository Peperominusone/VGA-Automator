"""
Export connected polylines to DXF format
"""
import ezdxf
from ezdxf import colors
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

from ..detection.segmentation_detector import SegmentedElement, ElementType


class DXFExporterContinuous:
    """Exports connected wall lines to DXF format"""
    
    # Layer and color configuration per element type
    LAYER_CONFIG = {
        ElementType.WALL: {
            'name': 'WALLS',
            'color': colors.WHITE,
            'linetype': 'Continuous'
        },
        ElementType.DOOR: {
            'name': 'DOORS',
            'color': colors.CYAN,
            'linetype': 'Continuous'
        },
        ElementType.WINDOW: {
            'name': 'WINDOWS',
            'color': colors.BLUE,
            'linetype': 'Continuous'
        },
        ElementType.COLUMN: {
            'name': 'COLUMNS',
            'color': colors.MAGENTA,
            'linetype': 'Continuous'
        },
        ElementType.SLIDING_DOOR: {
            'name': 'SLIDING_DOORS',
            'color': colors.GREEN,
            'linetype': 'Continuous'
        },
        ElementType.STAIR: {
            'name': 'STAIRS',
            'color': colors.YELLOW,
            'linetype': 'Continuous'
        },
        ElementType.CURTAIN_WALL: {
            'name': 'CURTAIN_WALLS',
            'color': colors.RED,
            'linetype': 'Continuous'
        },
        ElementType.RAILING: {
            'name': 'RAILINGS',
            'color': 8,  # Gray
            'linetype': 'Continuous'
        },
    }
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.doc = ezdxf.new('R2010')
        self.msp = self.doc.modelspace()
        self._setup_layers()
    
    def _setup_layers(self):
        """Create and configure layers"""
        for elem_type, config in self.LAYER_CONFIG.items():
            layer = self.doc.layers.add(config['name'])
            layer.color = config['color']
            layer.linetype = config['linetype']
    
    def export_elements(self, elements: Dict[ElementType, SegmentedElement], image_height: int = None):
        """Export all elements to DXF"""
        for elem_type, element in elements.items():
            layer_name = self.LAYER_CONFIG[elem_type]['name']
            
            if elem_type == ElementType.WALL:
                # Export walls as polylines
                self._export_polylines(element.polylines, layer_name, image_height)
            else:
                # Export other elements as contours
                self._export_contours(element.contours, layer_name, image_height)
    
    def _export_polylines(self, polylines: List[List[Tuple[float, float]]], layer_name: str, image_height: int = None):
        """Export polylines to DXF"""
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            
            points = []
            for x, y in polyline:
                # Flip Y-axis (image coords -> CAD coords)
                if image_height is not None:
                    y = image_height - y
                points.append((x, y))
            
            # Add as LWPOLYLINE (lightweight polyline)
            self.msp.add_lwpolyline(points, dxfattribs={'layer': layer_name})
    
    def _export_contours(self, contours: List[np.ndarray], layer_name: str, image_height: int = None):
        """Export contours to DXF"""
        for contour in contours:
            if len(contour) < 2:
                continue
            
            points = []
            for point in contour:
                x, y = float(point[0][0]), float(point[0][1])
                # Flip Y-axis
                if image_height is not None:
                    y = image_height - y
                points.append((x, y))
            
            # Add as closed polyline
            self.msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
    
    def save(self):
        """Save DXF file"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(self.output_path)
        print(f"✓ DXF file saved: {self.output_path}")
