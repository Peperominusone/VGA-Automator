"""
연결된 폴리라인을 DXF로 내보내기
"""
import ezdxf
from ezdxf import colors
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

from .segmentation_detector import SegmentedElement, ElementType


class DXFExporterContinuous:
    """연결된 벽체선을 DXF로 내보내기"""
    
    # Element type별 레이어 및 색상 설정
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
        """레이어 생성 및 설정"""
        for elem_type, config in self.LAYER_CONFIG.items():
            layer = self.doc.layers.add(config['name'])
            layer.color = config['color']
            layer.linetype = config['linetype']
    
    def export_elements(self, elements: Dict[ElementType, SegmentedElement], image_height: int = None):
        """모든 요소를 DXF로 내보내기"""
        for elem_type, element in elements.items():
            layer_name = self.LAYER_CONFIG[elem_type]['name']
            
            if elem_type == ElementType.WALL:
                # 벽체는 폴리라인으로 내보내기
                self._export_polylines(element.polylines, layer_name, image_height)
            else:
                # 기타 요소는 윤곽선으로 내보내기
                self._export_contours(element.contours, layer_name, image_height)
    
    def _export_polylines(self, polylines: List[List[Tuple[float, float]]], layer_name: str, image_height: int = None):
        """폴리라인을 DXF로 내보내기"""
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            
            points = []
            for x, y in polyline:
                # Y축 반전 (이미지 좌표 -> CAD 좌표)
                if image_height is not None:
                    y = image_height - y
                points.append((x, y))
            
            # LWPOLYLINE으로 추가 (경량 폴리라인)
            self.msp.add_lwpolyline(points, dxfattribs={'layer': layer_name})
    
    def _export_contours(self, contours: List[np.ndarray], layer_name: str, image_height: int = None):
        """윤곽선을 DXF로 내보내기"""
        for contour in contours:
            if len(contour) < 2:
                continue
            
            points = []
            for point in contour:
                x, y = float(point[0][0]), float(point[0][1])
                # Y축 반전
                if image_height is not None:
                    y = image_height - y
                points.append((x, y))
            
            # 닫힌 폴리라인으로 추가
            self.msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
    
    def save(self):
        """DXF 파일 저장"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(self.output_path)
        print(f"✓ DXF 파일 저장: {self.output_path}")
