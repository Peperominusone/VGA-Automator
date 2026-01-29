"""
DXF Export Module
Export detected floor plan elements to AutoCAD DXF format for VGA analysis
"""

import ezdxf
from ezdxf import colors
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


class DXFExporter:
    """Export floor plan data to DXF format optimized for VGA analysis"""
    
    # Layer definitions for different element types
    LAYERS = {
        'WALL': {'color': colors.WHITE, 'linetype': 'Continuous'},
        'DOOR': {'color': colors.BLUE, 'linetype': 'Continuous'},
        'WINDOW': {'color': colors.CYAN, 'linetype': 'Continuous'},
        'COLUMN': {'color': colors.MAGENTA, 'linetype': 'Continuous'},
        'CURTAIN_WALL': {'color': colors.GREEN, 'linetype': 'Continuous'},
        'RAILING': {'color': colors.YELLOW, 'linetype': 'Continuous'},
        'SLIDING_DOOR': {'color': colors.RED, 'linetype': 'Continuous'},
        'STAIR': {'color': 7, 'linetype': 'Continuous'},  # Gray
        'BOUNDARY': {'color': colors.RED, 'linetype': 'Continuous'},
        'OPENING': {'color': colors.GREEN, 'linetype': 'Continuous'},  # Changed from DASHED
    }
    
    def __init__(self, image_height: int = 0, scale_factor: float = 1.0):
        """
        Initialize DXF exporter
        
        Args:
            image_height: Height of source image (for Y-axis flip)
            scale_factor: Scaling factor from pixels to CAD units (e.g., mm)
        """
        self.image_height = image_height
        self.scale_factor = scale_factor
        self.doc = None
        self.msp = None
    
    def create_document(self, version: str = 'R2010') -> ezdxf.document.Drawing:
        """
        Create new DXF document
        
        Args:
            version: AutoCAD version (default: R2010 for compatibility)
            
        Returns:
            DXF document
        """
        self.doc = ezdxf.new(version)
        self.msp = self.doc.modelspace()
        
        # Create layers
        for layer_name, layer_props in self.LAYERS.items():
            self.doc.layers.add(
                name=layer_name,
                color=layer_props['color'],
                linetype=layer_props['linetype']
            )
        
        return self.doc
    
    def image_to_cad_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert image coordinates to CAD coordinates
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
            
        Returns:
            (x, y) in CAD coordinates (Y-axis flipped, scaled)
        """
        # Flip Y-axis (image Y increases down, CAD Y increases up)
        cad_x = x * self.scale_factor
        cad_y = (self.image_height - y) * self.scale_factor
        
        return cad_x, cad_y
    
    def add_line(self, x1: float, y1: float, x2: float, y2: float,
                 layer: str = 'WALL') -> None:
        """
        Add line to DXF document
        
        Args:
            x1, y1: Start point in image coordinates
            x2, y2: End point in image coordinates
            layer: Layer name
        """
        if self.msp is None:
            raise RuntimeError("Document not created. Call create_document() first.")
        
        # Convert to CAD coordinates
        start = self.image_to_cad_coords(x1, y1)
        end = self.image_to_cad_coords(x2, y2)
        
        # Add line to modelspace
        self.msp.add_line(start, end, dxfattribs={'layer': layer})
    
    def add_polyline(self, points: List[Tuple[float, float]], 
                     layer: str = 'WALL',
                     closed: bool = False) -> None:
        """
        Add polyline to DXF document
        
        Args:
            points: List of (x, y) points in image coordinates
            layer: Layer name
            closed: Whether to close the polyline
        """
        if self.msp is None:
            raise RuntimeError("Document not created. Call create_document() first.")
        
        # Convert points to CAD coordinates
        cad_points = [self.image_to_cad_coords(x, y) for x, y in points]
        
        # Add polyline
        polyline = self.msp.add_lwpolyline(
            cad_points,
            dxfattribs={'layer': layer}
        )
        
        if closed:
            polyline.close()
    
    def add_rectangle(self, x1: float, y1: float, x2: float, y2: float,
                     layer: str = 'BOUNDARY') -> None:
        """
        Add rectangle to DXF document
        
        Args:
            x1, y1: Top-left corner in image coordinates
            x2, y2: Bottom-right corner in image coordinates
            layer: Layer name
        """
        points = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]
        self.add_polyline(points, layer=layer, closed=True)
    
    def add_contour(self, contour: np.ndarray, layer: str = 'WALL',
                   simplify: bool = True, epsilon_factor: float = 0.01) -> None:
        """
        Add contour to DXF document
        
        Args:
            contour: OpenCV contour array
            layer: Layer name
            simplify: Whether to simplify the contour
            epsilon_factor: Simplification factor
        """
        import cv2
        
        if simplify:
            perimeter = cv2.arcLength(contour, True)
            epsilon = epsilon_factor * perimeter
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Extract points
        points = contour.reshape(-1, 2)
        points = [(float(p[0]), float(p[1])) for p in points]
        
        self.add_polyline(points, layer=layer, closed=True)
    
    def add_detection(self, detection: Dict, image: np.ndarray,
                     as_bbox: bool = False,
                     doors_as_openings: bool = True) -> None:
        """
        Add detected object to DXF
        
        Args:
            detection: Detection dictionary from detector
            image: Original image
            as_bbox: Add as bounding box instead of contour
            doors_as_openings: Treat doors as openings (passable in VGA)
        """
        class_name = detection['class_name']
        bbox = detection['bbox']
        
        # Map class name to layer
        layer_map = {
            'Wall': 'WALL',
            'Door': 'OPENING' if doors_as_openings else 'DOOR',
            'Window': 'WINDOW',
            'Column': 'COLUMN',
            'Curtain Wall': 'CURTAIN_WALL',
            'Railing': 'RAILING',
            'Sliding Door': 'OPENING' if doors_as_openings else 'SLIDING_DOOR',
            'Stair Case': 'STAIR'
        }
        
        layer = layer_map.get(class_name, 'BOUNDARY')
        
        if as_bbox:
            # Add as rectangle
            x1, y1, x2, y2 = bbox
            self.add_rectangle(x1, y1, x2, y2, layer=layer)
        else:
            # Try to use mask for more accurate representation
            if detection.get('mask') is not None:
                import cv2
                mask = detection['mask']
                x1, y1, x2, y2 = bbox
                
                # Validate bbox coordinates against mask dimensions
                h, w = mask.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                
                roi = mask[y1:y2, x1:x2]
                
                # Find contours in mask
                contours, _ = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    # Offset contour to image coordinates
                    offset_contour = contour.copy()
                    offset_contour[:, 0, 0] += x1
                    offset_contour[:, 0, 1] += y1
                    
                    # Add to DXF
                    self.add_contour(offset_contour, layer=layer)
            else:
                # Fallback to bounding box
                x1, y1, x2, y2 = bbox
                self.add_rectangle(x1, y1, x2, y2, layer=layer)
    
    def add_detections(self, detections: List[Dict], image: np.ndarray,
                      as_bbox: bool = False,
                      doors_as_openings: bool = True) -> None:
        """
        Add multiple detections to DXF
        
        Args:
            detections: List of detection dictionaries
            image: Original image
            as_bbox: Add as bounding boxes instead of contours
            doors_as_openings: Treat doors as openings (passable in VGA)
        """
        for detection in detections:
            self.add_detection(detection, image, as_bbox, doors_as_openings)
    
    def add_lines(self, lines: List[Tuple[int, int, int, int]],
                 layer: str = 'WALL') -> None:
        """
        Add multiple lines to DXF
        
        Args:
            lines: List of lines as (x1, y1, x2, y2) tuples
            layer: Layer name
        """
        for line in lines:
            x1, y1, x2, y2 = line
            self.add_line(x1, y1, x2, y2, layer=layer)
    
    def export_from_detections(self, detections: List[Dict],
                              image: np.ndarray,
                              output_path: Union[str, Path],
                              doors_as_openings: bool = True,
                              as_bbox: bool = False) -> None:
        """
        Complete pipeline: create DXF from detections and save
        
        Args:
            detections: List of detections from detector
            image: Original image
            output_path: Output DXF file path
            doors_as_openings: Treat doors as openings for VGA
            as_bbox: Use bounding boxes instead of detailed contours
        """
        # Set image height for coordinate conversion
        self.image_height = image.shape[0]
        
        # Create document
        self.create_document()
        
        # Add detections
        self.add_detections(detections, image, as_bbox, doors_as_openings)
        
        # Save
        self.save(output_path)
    
    def export_from_lines(self, lines: List[Tuple[int, int, int, int]],
                         image_height: int,
                         output_path: Union[str, Path],
                         layer: str = 'WALL') -> None:
        """
        Create DXF from extracted lines
        
        Args:
            lines: List of lines as (x1, y1, x2, y2) tuples
            image_height: Height of source image
            output_path: Output DXF file path
            layer: Layer name for lines
        """
        # Set image height for coordinate conversion
        self.image_height = image_height
        
        # Create document
        self.create_document()
        
        # Add lines
        self.add_lines(lines, layer=layer)
        
        # Save
        self.save(output_path)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save DXF document to file
        
        Args:
            output_path: Output file path
        """
        if self.doc is None:
            raise RuntimeError("No document to save. Create document first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.doc.saveas(str(output_path))
        print(f"DXF file saved to: {output_path}")
    
    def add_text_annotation(self, text: str, x: float, y: float,
                           height: float = 10.0,
                           layer: str = 'BOUNDARY') -> None:
        """
        Add text annotation to DXF
        
        Args:
            text: Text content
            x, y: Position in image coordinates
            height: Text height in CAD units
            layer: Layer name
        """
        if self.msp is None:
            raise RuntimeError("Document not created. Call create_document() first.")
        
        position = self.image_to_cad_coords(x, y)
        
        self.msp.add_text(
            text,
            dxfattribs={
                'layer': layer,
                'height': height
            }
        ).set_placement(position)
