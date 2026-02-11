"""
Segmentation-based object detection and connected contour extraction
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from pathlib import Path


class ElementType(Enum):
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    COLUMN = "column"
    STAIR = "stair"
    CURTAIN_WALL = "curtain_wall"
    RAILING = "railing"
    SLIDING_DOOR = "sliding_door"


SANATLADKAT_CLASS_MAP = {
    'Wall': ElementType.WALL,
    'Door': ElementType.DOOR,
    'Window': ElementType.WINDOW,
    'Column': ElementType.COLUMN,
    'Stair Case': ElementType.STAIR,
    'Curtain Wall': ElementType.CURTAIN_WALL,
    'Railing': ElementType.RAILING,
    'Sliding Door': ElementType.SLIDING_DOOR,
}


@dataclass
class SegmentedElement:
    """Segmented element from detection"""
    element_type: ElementType
    mask: np.ndarray
    contours: List[np.ndarray] = field(default_factory=list)
    skeleton: Optional[np.ndarray] = None
    polylines: List[List[Tuple[float, float]]] = field(default_factory=list)


class SegmentationDetector:
    """Segmentation-based floorplan element detector"""
    
    def __init__(self, model_path: str = "best.pt", confidence: float = 0.5, use_segmentation: bool = True, debug: bool = False):
        self.model_path = model_path
        self.confidence = confidence
        self.use_segmentation = use_segmentation
        self.debug = debug
        self.model = None
        self._load_model()
    
    def _load_model(self):
        from ultralytics import YOLO
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = YOLO(self.model_path)
        if self.debug:
            print(f"[DEBUG][Model] loaded: {self.model_path} | "f"use_segmentation={self.use_segmentation}")
    
    def detect_with_masks(self, image: np.ndarray, binary: np.ndarray, target_classes: List[str] = None) -> Dict[ElementType, SegmentedElement]:
        if target_classes is None:
            target_classes = ['Wall', 'Door', 'Window', 'Column', 'Sliding Door']
        
        h, w = image.shape[:2]
        class_masks: Dict[ElementType, np.ndarray] = {}
        
        for class_name in target_classes:
            if class_name in SANATLADKAT_CLASS_MAP:
                elem_type = SANATLADKAT_CLASS_MAP[class_name]
                class_masks[elem_type] = np.zeros((h, w), dtype=np.uint8)
        
        results = self.model.predict(image, conf=self.confidence, verbose=False)

        if self.debug:
            print(f"[DEBUG] use_segmentation={self.use_segmentation} | results={len(results)}")

        for r_i, result in enumerate(results):
            has_masks = hasattr(result,"masks") and (result.masks is not None)
            if self.debug and self.use_segmentation and not has_masks:
                print("[WARN] use_segmentation=True but result.masks is None. "
                      "This model is likely a DET model, not SEG.")
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    if class_name not in target_classes:
                        continue
                    elem_type = SANATLADKAT_CLASS_MAP[class_name]
                    mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) * 255
                    class_masks[elem_type] = cv2.bitwise_or(class_masks[elem_type], mask_resized)
            else:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    if class_name not in target_classes:
                        continue
                    elem_type = SANATLADKAT_CLASS_MAP[class_name]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    roi = binary[y1:y2, x1:x2]
                    if roi.size > 0:
                        class_masks[elem_type][y1:y2, x1:x2] = cv2.bitwise_or(class_masks[elem_type][y1:y2, x1:x2], roi)
        
        elements = {}
        for elem_type, mask in class_masks.items():
            if np.sum(mask) > 0:
                elements[elem_type] = SegmentedElement(element_type=elem_type, mask=mask)
        return elements
    
    def connect_segments(self, element: SegmentedElement, gap_size: int = 10, min_area: int = 100) -> SegmentedElement:
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
    
    def extract_skeleton(self, element: SegmentedElement) -> SegmentedElement:
        mask = element.mask.copy()
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary // 255)
        skeleton = img_as_ubyte(skeleton)
        element.skeleton = skeleton
        return element
    
    def skeleton_to_polylines(self, element: SegmentedElement, simplify_epsilon: float = 2.0) -> SegmentedElement:
        if element.skeleton is None:
            return element
        skeleton = element.skeleton.copy()
        polylines = []
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 2:
                continue
            simplified = cv2.approxPolyDP(contour, simplify_epsilon, closed=False)
            points = [(float(p[0][0]), float(p[0][1])) for p in simplified]
            if len(points) >= 2:
                polylines.append(points)
        element.polylines = self._merge_polylines(polylines)
        return element
    
    def extract_contours(self, element: SegmentedElement, simplify_epsilon: float = 2.0) -> SegmentedElement:
        mask = element.mask.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            simplified = cv2.approxPolyDP(contour, simplify_epsilon, closed=True)
            simplified_contours.append(simplified)
        element.contours = simplified_contours
        return element
    
    def _merge_polylines(self, polylines: List[List[Tuple[float, float]]], threshold: float = 10.0) -> List[List[Tuple[float, float]]]:
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


class ContinuousWallExtractor:
    """Wall-specific extractor - extracts fully connected wall lines"""
    
    def __init__(self):
        self.detector = None
    
    def extract_all_elements(self, image: np.ndarray, binary: np.ndarray, model_path: str = "best.pt", confidence: float = 0.5, gap_size : int =10, wall_gap_size: int | None = None) -> Dict[ElementType, SegmentedElement]:
        self.detector = SegmentationDetector(model_path=model_path, confidence=confidence)
        elements = self.detector.detect_with_masks(image, binary, target_classes=['Wall', 'Door', 'Window', 'Column', 'Sliding Door'])
        processed = {}
        for elem_type, element in elements.items():
            gs = wall_gap_size if (elem_type == ElementType.WALL and wall_gap_size is not None) else gap_size
            element = self.detector.connect_segments(element, gap_size=gap_size)
            if elem_type == ElementType.WALL:
                element = self.detector.extract_skeleton(element)
                element = self.detector.skeleton_to_polylines(element)
            else:
                element = self.detector.extract_contours(element)
            processed[elem_type] = element
        return processed
