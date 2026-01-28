"""
Image Preprocessing Module
Handles loading, noise reduction, and binarization of floor plan images
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import os


class Preprocessor:
    """Image preprocessor for floor plan images"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file (PNG, JPG, PDF supported)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        image_path = Path(image_path)
        self.image_path = image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Handle PDF files
        if image_path.suffix.lower() == '.pdf':
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(image_path), dpi=300)
                if not images:
                    raise ValueError("No images found in PDF")
                # Convert PIL Image to numpy array
                image = np.array(images[0])
                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except ImportError:
                raise ImportError("pdf2image is required for PDF support. Install with: pip install pdf2image")
        else:
            # Load regular image files
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        
        self.original_image = image
        return image
    
    def convert_to_grayscale(self, image: np.ndarray = None) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input image (uses loaded image if None)
            
        Returns:
            Grayscale image
        """
        if image is None:
            image = self.original_image
            
        if image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Check if already grayscale
        if len(image.shape) == 2:
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def denoise(self, image: np.ndarray, h: int = 10) -> np.ndarray:
        """
        Remove noise from image using fastNlMeansDenoising
        
        Args:
            image: Input image
            h: Filter strength (higher = more denoising)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        
        return denoised
    
    def binarize(self, image: np.ndarray, method: str = 'adaptive', 
                 threshold: int = 127, block_size: int = 11, C: int = 2) -> np.ndarray:
        """
        Binarize image using thresholding
        
        Args:
            image: Input grayscale image
            method: 'adaptive' or 'otsu' or 'simple'
            threshold: Threshold value for simple thresholding
            block_size: Block size for adaptive thresholding (must be odd)
            C: Constant subtracted from mean in adaptive thresholding
            
        Returns:
            Binary image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, C
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'simple':
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
        
        return binary
    
    def morphological_operations(self, image: np.ndarray, 
                                 operation: str = 'close',
                                 kernel_size: Tuple[int, int] = (3, 3),
                                 iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to clean up lines
        
        Args:
            image: Input binary image
            operation: 'close', 'open', 'erode', 'dilate'
            kernel_size: Size of the structuring element
            iterations: Number of times to apply operation
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        if operation == 'close':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'erode':
            result = cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilate':
            result = cv2.dilate(image, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        return result
    
    def preprocess(self, image_path: Union[str, Path], 
                   denoise_strength: int = 10,
                   threshold_method: str = 'adaptive',
                   block_size: int = 11,
                   morph_operation: str = 'close',
                   morph_kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            denoise_strength: Strength of denoising filter
            threshold_method: Binarization method
            block_size: Block size for adaptive thresholding
            morph_operation: Morphological operation to apply
            morph_kernel_size: Kernel size for morphological operation
            
        Returns:
            Preprocessed binary image
        """
        # Load image
        image = self.load_image(image_path)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Denoise
        denoised = self.denoise(gray, h=denoise_strength)
        
        # Binarize
        binary = self.binarize(denoised, method=threshold_method, block_size=block_size)
        
        # Apply morphological operations to clean up
        cleaned = self.morphological_operations(
            binary, 
            operation=morph_operation,
            kernel_size=morph_kernel_size
        )
        
        self.processed_image = cleaned
        return cleaned
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]):
        """
        Save processed image to file
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
