# VGA-Automator
Floorplan to DXF converter using AI-powered segmentation

## 개요

VGA-Automator는 건축 도면 이미지를 DXF 파일로 자동 변환하는 도구입니다. YOLO segmentation 모델을 사용하여 벽체, 문, 창문 등의 건축 요소를 정확하게 감지하고 추출합니다.

## 주요 기능

### Segmentation 기반 연결된 벽체선 추출

1. **Segmentation 기반 감지**: 바운딩 박스 대신 픽셀 단위 마스크로 정확한 형태 추출
2. **연결된 세그먼트**: 끊어진 벽체를 모폴로지 연산으로 연결
3. **골격화(Skeletonize)**: 벽체 마스크에서 중심선 추출
4. **폴리라인 병합**: 끝점이 가까운 선분들을 하나로 연결

## 설치

### 요구사항

- Python 3.8 이상
- YOLO segmentation 모델 파일 (`best.pt`)

### 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
python main_continuous.py floorplan.png -o output.dxf
```

### 고급 옵션

```bash
# 모델 및 신뢰도 설정
python main_continuous.py floorplan.png -o output.dxf --model custom_model.pt --confidence 0.6

# 갭 연결 크기 조정
python main_continuous.py floorplan.png -o output.dxf --gap 15

# 디버그 이미지 생성
python main_continuous.py floorplan.png -o output.dxf --debug
```

### 명령행 옵션

- `input`: 입력 도면 이미지 경로 (필수)
- `-o, --output`: 출력 DXF 파일 경로 (필수)
- `--model`: YOLO 모델 파일 경로 (기본값: best.pt)
- `--confidence`: 감지 신뢰도 임계값 (기본값: 0.5)
- `--gap`: 연결할 갭 크기 (기본값: 10)
- `--debug`: 디버그 이미지 저장

## 알고리즘 흐름

```
원본 도면 → YOLO 감지 → 클래스별 마스크 병합 → 갭 연결 (모폴로지)
    → 골격화 (Skeletonize) → 폴리라인 변환 → 끝점 병합 → DXF 저장
```

## 지원 요소

- 벽체 (Wall) - 폴리라인으로 변환
- 문 (Door)
- 창문 (Window)
- 기둥 (Column)
- 슬라이딩 도어 (Sliding Door)
- 계단 (Stair Case)
- 커튼월 (Curtain Wall)
- 난간 (Railing)

## 프로젝트 구조

```
VGA-Automator/
├── src/
│   ├── __init__.py
│   ├── segmentation_detector.py      # Segmentation 기반 요소 감지
│   └── dxf_exporter_continuous.py    # DXF 내보내기
├── main_continuous.py                 # CLI 진입점
├── requirements.txt                   # 의존성
└── README.md
```

## 라이센스

MIT License
