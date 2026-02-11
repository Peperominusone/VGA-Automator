"""
Contour Extraction and Simplification Module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import math


class ContourExtractor:
    """Extract and simplify contours from binary images"""
    
    def __init__(self):
        self.contours = []
        self.simplified_contours = []
        self.lines = []
    
    def find_contours(self, binary_image: np.ndarray,
                     min_area: int = 100) -> List[np.ndarray]:
        """
        Find contours in binary image
        
        Args:
            binary_image: Binary input image
            min_area: Minimum contour area to keep
            
        Returns:
            List of contours
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by minimum area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)
        
        self.contours = filtered_contours
        return filtered_contours
    
    def simplify_contours(self, contours: Optional[List[np.ndarray]] = None,
                         epsilon_factor: float = 0.01) -> List[np.ndarray]:
        """
        Simplify contours using Douglas-Peucker algorithm
        
        Args:
            contours: List of contours (uses self.contours if None)
            epsilon_factor: Approximation accuracy factor (smaller = more accurate)
            
        Returns:
            List of simplified contours
        """
        if contours is None:
            contours = self.contours
        
        simplified = []
        for contour in contours:
            # Calculate epsilon based on contour perimeter
            perimeter = cv2.arcLength(contour, True)
            epsilon = epsilon_factor * perimeter
            
            # Approximate contour
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified.append(approx)
        
        self.simplified_contours = simplified
        return simplified
    
    def extract_lines_from_contours(self, contours: Optional[List[np.ndarray]] = None,
                                   min_length: float = 10.0) -> List[Tuple[int, int, int, int]]:
        """
        Extract line segments from contours
        
        Args:
            contours: List of contours (uses self.simplified_contours if None)
            min_length: Minimum line length to keep
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        if contours is None:
            contours = self.simplified_contours
        
        lines = []
        for contour in contours:
            # Extract points from contour
            points = contour.reshape(-1, 2)
            
            # Create line segments between consecutive points
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                # Calculate line length
                length = np.linalg.norm(p2 - p1)
                
                if length >= min_length:
                    lines.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        self.lines = lines
        return lines
    
    def extract_lines_hough(self, binary_image: np.ndarray,
                           threshold: int = 50,
                           min_line_length: int = 30,
                           max_line_gap: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        Extract lines using Hough transform
        
        Args:
            binary_image: Binary input image
            threshold: Accumulator threshold
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        # Apply edge detection
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
        
        self.lines = line_list
        return line_list
    
    def merge_nearby_lines(self, lines: Optional[List[Tuple[int, int, int, int]]] = None,
                          distance_threshold: float = 10.0,
                          angle_threshold: float = 5.0) -> List[Tuple[int, int, int, int]]:
        """
        Merge lines that are close and parallel
        
        Args:
            lines: List of lines (uses self.lines if None)
            distance_threshold: Maximum distance between lines to merge
            angle_threshold: Maximum angle difference (degrees) to merge
            
        Returns:
            List of merged lines
        """
        if lines is None:
            lines = self.lines
        
        if not lines:
            return []
        
        def line_angle(line):
            """Calculate line angle in degrees"""
            x1, y1, x2, y2 = line
            return math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        def point_to_line_distance(point, line):
            """Calculate perpendicular distance from point to line"""
            x0, y0 = point
            x1, y1, x2, y2 = line
            
            # Handle vertical and horizontal lines separately
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # Calculate distance using cross product
            numerator = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
            denominator = math.sqrt(dx**2 + dy**2)
            
            return numerator / denominator
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            angle1 = line_angle(line1)
            similar_lines = [line1]
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                angle2 = line_angle(line2)
                
                # Check if angles are similar
                angle_diff = abs(angle1 - angle2) % 180
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                if angle_diff <= angle_threshold:
                    # Check if lines are close
                    x1_l2, y1_l2, x2_l2, y2_l2 = line2
                    
                    dist1 = point_to_line_distance((x1_l2, y1_l2), line1)
                    dist2 = point_to_line_distance((x2_l2, y2_l2), line1)
                    
                    if dist1 <= distance_threshold or dist2 <= distance_threshold:
                        similar_lines.append(line2)
                        used.add(j)
            
            # Merge similar lines by finding extreme points
            if len(similar_lines) == 1:
                merged.append(line1)
            else:
                all_points = []
                for line in similar_lines:
                    x1, y1, x2, y2 = line
                    all_points.extend([(x1, y1), (x2, y2)])
                
                # Find extreme points
                all_points = np.array(all_points)
                
                # Use PCA to find line direction
                mean = np.mean(all_points, axis=0)
                centered = all_points - mean
                
                if len(centered) > 1:
                    cov = np.cov(centered.T)
                    
                    # Check if covariance matrix is valid
                    if np.linalg.matrix_rank(cov) < 2:
                        # Degenerate case: all points are collinear
                        # Just use the extreme points along the first dimension
                        idx_sort = np.argsort(centered[:, 0])
                        p1 = all_points[idx_sort[0]]
                        p2 = all_points[idx_sort[-1]]
                    else:
                        eigenvalues, eigenvectors = np.linalg.eig(cov)
                        
                        # Principal direction
                        principal_dir = eigenvectors[:, np.argmax(eigenvalues)]
                        
                        # Project points onto principal direction
                        projections = np.dot(centered, principal_dir)
                        
                        # Find extreme projections
                        min_idx = np.argmin(projections)
                        max_idx = np.argmax(projections)
                        
                        p1 = all_points[min_idx]
                        p2 = all_points[max_idx]
                    
                    merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
                else:
                    merged.append(line1)
        
        self.lines = merged
        return merged
    
    def filter_short_lines(self, lines: Optional[List[Tuple[int, int, int, int]]] = None,
                          min_length: float = 10.0) -> List[Tuple[int, int, int, int]]:
        """
        Remove lines shorter than minimum length
        
        Args:
            lines: List of lines (uses self.lines if None)
            min_length: Minimum line length
            
        Returns:
            Filtered list of lines
        """
        if lines is None:
            lines = self.lines
        
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                filtered.append(line)
        
        self.lines = filtered
        return filtered
    
    def visualize_contours(self, image: np.ndarray,
                          contours: Optional[List[np.ndarray]] = None,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Draw contours on image
        
        Args:
            image: Input image
            contours: List of contours (uses self.contours if None)
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn contours
        """
        if contours is None:
            contours = self.contours
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def visualize_lines(self, image: np.ndarray,
                       lines: Optional[List[Tuple[int, int, int, int]]] = None,
                       color: Tuple[int, int, int] = (0, 0, 255),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw lines on image
        
        Args:
            image: Input image
            lines: List of lines (uses self.lines if None)
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn lines
        """
        if lines is None:
            lines = self.lines
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        
        return result
