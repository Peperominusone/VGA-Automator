"""
Object Detection Module using YOLOv8
Integrates sanatladkat/floor-plan-object-detection model
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import os


class FloorPlanDetector:
    """
    Floor plan object detector using YOLOv8 model
    Detects: Wall, Door, Window, Column, Curtain Wall, Railing, Sliding Door, Stair Case
    """
    
    # Class names from sanatladkat/floor-plan-object-detection model
    CLASS_NAMES = {
        0: 'Wall',
        1: 'Door',
        2: 'Window',
        3: 'Column',
        4: 'Curtain Wall',
        5: 'Railing',
        6: 'Sliding Door',
        7: 'Stair Case'
    }
    
    def __init__(self, model_path: Union[str, Path] = 'best.pt', confidence: float = 0.5):
        """
        Initialize detector with YOLOv8 model
        
        Args:
            model_path: Path to the best.pt model file
            confidence: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self.detections = []
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please download best.pt from:\n"
                    f"https://github.com/sanatladkat/floor-plan-object-detection"
                )
            
            self.model = YOLO(str(self.model_path))
            print(f"Model loaded successfully from {self.model_path}")
            
        except ImportError:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )
    
    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in floor plan image
        
        Args:
            image: Input image
            conf: Confidence threshold (overrides self.confidence if provided)
            
        Returns:
            List of detection dictionaries with keys:
                - class_id: Class ID
                - class_name: Class name
                - confidence: Detection confidence
                - bbox: Bounding box [x1, y1, x2, y2]
                - mask: Binary mask (if available)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        confidence = conf if conf is not None else self.confidence
        
        # Run detection
        results = self.model(image, conf=confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract box data
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES.get(class_id, f'Unknown_{class_id}'),
                    'confidence': conf_score,
                    'bbox': [x1, y1, x2, y2],
                    'mask': None
                }
                
                # Extract mask if available
                if hasattr(result, 'masks') and result.masks is not None:
                    try:
                        mask = result.masks[i].data[0].cpu().numpy()
                        # Resize mask to match image dimensions
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        detection['mask'] = (mask > 0.5).astype(np.uint8) * 255
                    except:
                        pass
                
                detections.append(detection)
        
        self.detections = detections
        return detections
    
    def extract_contours_from_detection(self, detection: Dict, 
                                       image: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from a detection's bounding box or mask
        
        Args:
            detection: Detection dictionary
            image: Original image
            
        Returns:
            List of contours
        """
        x1, y1, x2, y2 = detection['bbox']
        
        # Use mask if available
        if detection['mask'] is not None:
            roi = detection['mask'][y1:y2, x1:x2]
        else:
            # Use bounding box region
            roi = image[y1:y2, x1:x2]
            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Offset contours to original image coordinates
        offset_contours = []
        for contour in contours:
            offset_contour = contour.copy()
            offset_contour[:, 0, 0] += x1
            offset_contour[:, 0, 1] += y1
            offset_contours.append(offset_contour)
        
        return offset_contours
    
    def extract_lines_from_walls(self, image: np.ndarray, 
                                 wall_detections: List[Dict],
                                 threshold: int = 50,
                                 min_line_length: int = 30,
                                 max_line_gap: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        Extract straight lines from wall regions using Hough transform
        
        Args:
            image: Original image
            wall_detections: List of wall detections
            threshold: Hough transform threshold
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        lines = []
        
        for detection in wall_detections:
            if detection['class_name'] not in ['Wall', 'Curtain Wall']:
                continue
            
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract region
            if detection['mask'] is not None:
                roi = detection['mask'][y1:y2, x1:x2]
            else:
                roi = image[y1:y2, x1:x2]
                if len(roi.shape) == 3:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(roi, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            detected_lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180,
                threshold=threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
            
            if detected_lines is not None:
                for line in detected_lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    # Offset to original image coordinates
                    lines.append((lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1))
        
        return lines
    
    def visualize_detections(self, image: np.ndarray, 
                            detections: Optional[List[Dict]] = None,
                            show_labels: bool = True,
                            show_confidence: bool = True) -> np.ndarray:
        """
        Draw detections on image for visualization
        
        Args:
            image: Input image
            detections: List of detections (uses self.detections if None)
            show_labels: Show class labels
            show_confidence: Show confidence scores
            
        Returns:
            Image with drawn detections
        """
        if detections is None:
            detections = self.detections
        
        result_image = image.copy()
        
        # Color mapping for different classes
        colors = {
            'Wall': (0, 255, 0),           # Green
            'Door': (255, 0, 0),            # Blue
            'Window': (0, 255, 255),        # Yellow
            'Column': (255, 0, 255),        # Magenta
            'Curtain Wall': (0, 128, 255),  # Orange
            'Railing': (128, 0, 128),       # Purple
            'Sliding Door': (255, 128, 0),  # Cyan
            'Stair Case': (128, 128, 128)   # Gray
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = class_name
                if show_confidence:
                    label += f" {confidence:.2f}"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result_image,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return result_image
    
    def filter_detections_by_class(self, class_names: List[str],
                                   detections: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Filter detections by class name
        
        Args:
            class_names: List of class names to keep
            detections: List of detections (uses self.detections if None)
            
        Returns:
            Filtered list of detections
        """
        if detections is None:
            detections = self.detections
        
        return [d for d in detections if d['class_name'] in class_names]
