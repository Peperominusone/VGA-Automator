#!/usr/bin/env python3
"""
연결된 벽체선 버전 CLI - Segmentation 기반 도면 변환
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from src.segmentation_detector import ContinuousWallExtractor, ElementType
from src.dxf_exporter_continuous import DXFExporterContinuous


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """이미지 전처리 - 바이너리 이미지 생성"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return image, binary


def save_debug_images(elements: dict, output_dir: Path, image_shape: tuple):
    """디버그용 마스크 이미지 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for elem_type, element in elements.items():
        # 마스크 저장
        mask_path = output_dir / f"debug_{elem_type.value}_mask.png"
        cv2.imwrite(str(mask_path), element.mask)
        
        # 스켈레톤 저장 (벽체만)
        if elem_type == ElementType.WALL and element.skeleton is not None:
            skeleton_path = output_dir / f"debug_{elem_type.value}_skeleton.png"
            cv2.imwrite(str(skeleton_path), element.skeleton)
        
        # 폴리라인 시각화 (벽체만)
        if elem_type == ElementType.WALL and element.polylines:
            polyline_img = np.zeros(image_shape[:2], dtype=np.uint8)
            for polyline in element.polylines:
                pts = np.array(polyline, dtype=np.int32)
                cv2.polylines(polyline_img, [pts], False, 255, 2)
            polyline_path = output_dir / f"debug_{elem_type.value}_polylines.png"
            cv2.imwrite(str(polyline_path), polyline_img)
        
        # 윤곽선 시각화 (기타 요소)
        if elem_type != ElementType.WALL and element.contours:
            contour_img = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_img, element.contours, -1, 255, 2)
            contour_path = output_dir / f"debug_{elem_type.value}_contours.png"
            cv2.imwrite(str(contour_path), contour_img)


def main():
    parser = argparse.ArgumentParser(
        description='연결된 벽체선 추출 - Segmentation 기반 도면 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  %(prog)s floorplan.png -o output.dxf
  %(prog)s floorplan.png -o output.dxf --gap 15 --debug
  %(prog)s floorplan.png -o output.dxf --model custom_model.pt --confidence 0.6
        """
    )
    
    parser.add_argument('input', help='입력 도면 이미지 경로')
    parser.add_argument('-o', '--output', required=True, help='출력 DXF 파일 경로')
    parser.add_argument('--model', default='best.pt', help='YOLO 모델 파일 경로 (기본값: best.pt)')
    parser.add_argument('--confidence', type=float, default=0.5, help='감지 신뢰도 임계값 (기본값: 0.5)')
    parser.add_argument('--gap', type=int, default=10, help='연결할 갭 크기 (기본값: 10)')
    parser.add_argument('--debug', action='store_true', help='디버그 이미지 저장')
    
    args = parser.parse_args()
    
    # 파일 경로 검증
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 입력 파일이 존재하지 않습니다: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}", file=sys.stderr)
        print(f"   YOLO segmentation 모델 파일(.pt)을 준비해주세요.", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output)
    
    try:
        # 1. 이미지 전처리
        print(f"📄 이미지 로드 중: {input_path}")
        image, binary = preprocess_image(str(input_path))
        print(f"   크기: {image.shape[1]}x{image.shape[0]}")
        
        # 2. 요소 추출
        print(f"🔍 요소 감지 중 (모델: {model_path}, 신뢰도: {args.confidence})")
        extractor = ContinuousWallExtractor()
        elements = extractor.extract_all_elements(
            image, binary, 
            model_path=str(model_path), 
            confidence=args.confidence
        )
        
        # 결과 요약
        print(f"\n✓ 감지된 요소:")
        for elem_type, element in elements.items():
            if elem_type == ElementType.WALL:
                count = len(element.polylines)
                print(f"   - {elem_type.value}: {count}개 폴리라인")
            else:
                count = len(element.contours)
                print(f"   - {elem_type.value}: {count}개 윤곽선")
        
        if not elements:
            print("⚠️  감지된 요소가 없습니다.", file=sys.stderr)
            sys.exit(0)
        
        # 3. DXF 내보내기
        print(f"\n💾 DXF 내보내기 중: {output_path}")
        exporter = DXFExporterContinuous(str(output_path))
        exporter.export_elements(elements, image_height=image.shape[0])
        exporter.save()
        
        # 4. 디버그 이미지 저장
        if args.debug:
            debug_dir = output_path.parent / f"{output_path.stem}_debug"
            print(f"\n🔧 디버그 이미지 저장 중: {debug_dir}")
            save_debug_images(elements, debug_dir, image.shape)
            print(f"   ✓ 디버그 이미지 저장 완료")
        
        print(f"\n✅ 변환 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
