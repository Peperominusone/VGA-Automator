# VGA-Automator

건축 도면을 Space Syntax VGA 분석용 DXF 파일로 자동 변환하고, YOLOv8 세그멘테이션 모델을 학습할 수 있는 통합 도구

Automatically convert architectural floor plans to DXF files for Space Syntax VGA (Visibility Graph Analysis) with integrated model training capabilities

---

## 📋 목차 / Table of Contents

- [한국어](#korean)
- [English](#english)

---

<a name="korean"></a>
## 🇰🇷 한국어

### 프로젝트 개요

VGA-Automator는 건축 도면 이미지(PNG/JPG/PDF)를 Space Syntax VGA(Visibility Graph Analysis) 분석을 위한 DXF 파일로 자동 변환하는 통합 도구입니다. YOLOv8 세그멘테이션 기반 객체 인식을 통해 벽체, 문, 창문 등을 감지하고 정밀한 DXF 파일을 생성합니다. 또한 CubiCasa5k 데이터셋을 활용한 모델 학습 기능도 제공합니다.

### 주요 기능

1. **도면 변환 (Inference)**
   - 이미지 전처리: 도면 로드, 노이즈 제거, 이진화
   - 객체 인식: YOLOv8-seg 모델을 활용한 세그멘테이션 기반 인식
   - 윤곽선 추출: 감지된 요소에서 정밀한 폴리라인 추출
   - DXF 내보내기: 레이어별로 구분된 AutoCAD 호환 DXF 파일 생성

2. **모델 학습 (Training)**
   - CubiCasa5k 데이터셋 변환 (YOLO-seg 형식)
   - YOLOv8 세그멘테이션 모델 학습
   - 커스텀 데이터셋 지원

3. **데스크톱 앱**
   - PyQt6 기반 GUI 애플리케이션
   - 드래그 앤 드롭 지원
   - 실시간 진행 상황 표시

### 감지 가능한 객체

- Wall (벽체)
- Door (문)
- Window (창문)
- Column (기둥)

### 설치 방법

#### 1. 저장소 클론

```bash
git clone https://github.com/Peperominusone/VGA-Automator.git
cd VGA-Automator
```

#### 2. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

#### 3. 모델 파일 준비

학습된 모델을 `models/` 디렉토리에 배치하거나, 직접 학습합니다:

```bash
# 학습된 모델 다운로드 또는
# 아래 학습 섹션 참조하여 직접 학습
```

---

## 사용법

### 1. 도면 변환 (Inference)

#### 기본 사용

```bash
python main.py infer floorplan.png -o output.dxf
```

#### 고급 옵션

```bash
# 커스텀 모델 및 신뢰도 설정
python main.py infer floorplan.png -o output.dxf --model models/best.pt --confidence 0.6

# 갭 연결 크기 조정
python main.py infer floorplan.png -o output.dxf --gap 15

# 디버그 이미지 생성
python main.py infer floorplan.png -o output.dxf --debug
```

#### 명령행 옵션

- `input`: 입력 도면 이미지 경로 (필수)
- `-o, --output`: 출력 DXF 파일 경로 (기본값: output.dxf)
- `--model`: YOLO 모델 파일 경로 (기본값: models/best.pt)
- `--confidence`: 감지 신뢰도 임계값 (기본값: 0.5)
- `--gap`: 연결할 갭 크기 (기본값: 10)
- `--debug`: 디버그 이미지 저장

### 2. 모델 학습 (Training)

#### 데이터 준비

CubiCasa5k 데이터셋을 YOLO 형식으로 변환:

```bash
python main.py convert \
  --cubicasa_root data/raw/CubiCasa5k \
  --out_root training/data/yolo
```

**참고**: `convert` 명령은 디렉토리 구조와 `data.yaml`을 생성합니다. CubiCasa5k SVG 파싱은 `training/scripts/convert_cubicasa_to_yolo_seg.py`에서 구현해야 합니다.

#### 모델 학습

```bash
# 기본 학습
python main.py train --data training/data/yolo/data.yaml --epochs 100

# 고급 학습 설정
python main.py train \
  --data training/data/yolo/data.yaml \
  --model yolov8m-seg.pt \
  --epochs 200 \
  --batch 16 \
  --imgsz 1024
```

#### 학습 옵션

- `--data`: data.yaml 경로 (필수)
- `--model`: 기본 모델 (기본값: yolov8n-seg.pt)
- `--epochs`: 학습 에폭 수 (기본값: 100)
- `--batch`: 배치 크기 (기본값: 8)
- `--imgsz`: 이미지 크기 (기본값: 1024)
- `--device`: 디바이스 (예: 0, cpu)
- `--project`: 프로젝트 디렉토리 (기본값: training/runs/segment)
- `--name`: 실행 이름 (기본값: vga_yolov8seg)

학습 완료 후 `training/runs/segment/<name>/weights/best.pt`에 모델이 저장됩니다.

### 3. 데스크톱 앱

GUI 애플리케이션 실행:

```bash
python app_desktop.py
```

기능:
- 드래그 앤 드롭으로 도면 업로드
- 실시간 설정 조정 (신뢰도, 갭 크기, 스케일)
- 진행 상황 표시
- 변환 결과 통계 표시

---

## 알고리즘 흐름

```
원본 도면 → YOLO 세그멘테이션 → 클래스별 마스크 병합 → 갭 연결 (모폴로지)
    → 골격화 (Skeletonize) → 폴리라인 변환 → 끝점 병합 → DXF 저장
```

### 상세 프로세스

1. **전처리**: 이미지 로드, 그레이스케일 변환, 적응형 임계값 처리
2. **세그멘테이션**: YOLOv8-seg 모델로 건축 요소 감지 및 세그멘테이션
3. **후처리**:
   - 클래스별 마스크 병합
   - 모폴로지 연산으로 갭 연결
   - 골격화를 통한 중심선 추출 (벽체)
   - 윤곽선 추출 (문, 창문)
4. **내보내기**: 레이어별 DXF 파일 생성

---

## 프로젝트 구조

```
VGA-Automator/
├── src/                                  # 소스 코드
│   ├── preprocessing/                    # 전처리 모듈
│   │   ├── __init__.py
│   │   └── preprocessor.py              # 이미지 전처리
│   ├── detection/                        # 감지 모듈
│   │   ├── __init__.py
│   │   ├── detector.py                  # 객체 감지기
│   │   └── segmentation_detector.py     # 세그멘테이션 감지기
│   ├── postprocessing/                   # 후처리 모듈
│   │   ├── __init__.py
│   │   └── contour_extractor.py         # 윤곽선 추출
│   ├── export/                           # 내보내기 모듈
│   │   ├── __init__.py
│   │   ├── dxf_exporter.py              # DXF 내보내기
│   │   └── dxf_exporter_continuous.py   # 연속 라인 DXF 내보내기
│   └── __init__.py
├── training/                             # 학습 관련
│   ├── scripts/                          # 학습 스크립트
│   │   ├── convert_cubicasa_to_yolo_seg.py
│   │   └── train_yolov8_seg.py
│   ├── configs/                          # 설정 파일
│   │   ├── classes.json                 # 클래스 정의
│   │   └── data.yaml.template           # 데이터 설정 템플릿
│   ├── src/                              # 학습 유틸리티
│   │   └── svg_utils.py
│   └── __init__.py
├── models/                               # 모델 파일 디렉토리
│   └── .gitkeep
├── main.py                               # 통합 CLI 진입점
├── main_continuous.py                    # 레거시 CLI (연속 라인 모드)
├── main_legacy.py                        # 레거시 CLI (기본 모드)
├── app_desktop.py                        # PyQt6 데스크톱 앱
├── examples.py                           # API 사용 예제
├── test_validation.py                    # 검증 테스트
├── requirements.txt                      # 의존성
├── README.md                             # 문서
└── .gitignore                            # Git 무시 파일
```

---

## 개발 및 테스트

### 검증 테스트 실행

```bash
python test_validation.py
```

### 예제 코드 실행

```bash
python examples.py
```

---

## 레거시 CLI

이전 버전과의 호환성을 위해 레거시 CLI도 제공됩니다:

```bash
# 기본 모드
python main_legacy.py floorplan.png -o output.dxf

# 연속 라인 모드 (세그멘테이션 기반)
python main_continuous.py floorplan.png -o output.dxf
```

---

## 기여

이슈 및 풀 리퀘스트 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 라이센스

MIT License

---

<a name="english"></a>
## 🇬🇧 English

### Project Overview

VGA-Automator is an integrated tool that automatically converts architectural floor plan images (PNG/JPG/PDF) to DXF files for Space Syntax VGA (Visibility Graph Analysis). It uses YOLOv8 segmentation-based object detection to identify walls, doors, windows, and generates precise DXF files. It also provides model training capabilities using the CubiCasa5k dataset.

### Key Features

1. **Floor Plan Conversion (Inference)**
   - Image preprocessing: loading, denoising, binarization
   - Object detection: YOLOv8-seg model for segmentation-based detection
   - Contour extraction: precise polyline extraction from detected elements
   - DXF export: layer-separated AutoCAD-compatible DXF file generation

2. **Model Training**
   - CubiCasa5k dataset conversion (YOLO-seg format)
   - YOLOv8 segmentation model training
   - Custom dataset support

3. **Desktop Application**
   - PyQt6-based GUI application
   - Drag-and-drop support
   - Real-time progress display

### Detectable Objects

- Wall
- Door
- Window
- Column

### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/Peperominusone/VGA-Automator.git
cd VGA-Automator
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Prepare Model

Place a trained model in the `models/` directory or train your own:

```bash
# Download a pre-trained model or
# See training section below to train your own
```

---

## Usage

### 1. Floor Plan Conversion (Inference)

#### Basic Usage

```bash
python main.py infer floorplan.png -o output.dxf
```

#### Advanced Options

```bash
# Custom model and confidence
python main.py infer floorplan.png -o output.dxf --model models/best.pt --confidence 0.6

# Adjust gap connection size
python main.py infer floorplan.png -o output.dxf --gap 15

# Generate debug images
python main.py infer floorplan.png -o output.dxf --debug
```

#### Command-line Options

- `input`: Input floor plan image path (required)
- `-o, --output`: Output DXF file path (default: output.dxf)
- `--model`: YOLO model file path (default: models/best.pt)
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--gap`: Gap size for connecting segments (default: 10)
- `--debug`: Save debug images

### 2. Model Training

#### Prepare Data

Convert CubiCasa5k dataset to YOLO format:

```bash
python main.py convert \
  --cubicasa_root data/raw/CubiCasa5k \
  --out_root training/data/yolo
```

**Note**: The `convert` command creates directory structure and `data.yaml`. CubiCasa5k SVG parsing needs to be implemented in `training/scripts/convert_cubicasa_to_yolo_seg.py`.

#### Train Model

```bash
# Basic training
python main.py train --data training/data/yolo/data.yaml --epochs 100

# Advanced training settings
python main.py train \
  --data training/data/yolo/data.yaml \
  --model yolov8m-seg.pt \
  --epochs 200 \
  --batch 16 \
  --imgsz 1024
```

#### Training Options

- `--data`: Path to data.yaml (required)
- `--model`: Base model (default: yolov8n-seg.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 8)
- `--imgsz`: Image size (default: 1024)
- `--device`: Device (e.g., 0, cpu)
- `--project`: Project directory (default: training/runs/segment)
- `--name`: Run name (default: vga_yolov8seg)

After training, the model is saved to `training/runs/segment/<name>/weights/best.pt`.

### 3. Desktop Application

Run GUI application:

```bash
python app_desktop.py
```

Features:
- Drag-and-drop floor plan upload
- Real-time settings adjustment (confidence, gap size, scale)
- Progress display
- Conversion result statistics

---

## Algorithm Flow

```
Original Floor Plan → YOLO Segmentation → Merge Class Masks → Connect Gaps (Morphology)
    → Skeletonize → Convert to Polylines → Merge Endpoints → Save DXF
```

### Detailed Process

1. **Preprocessing**: Load image, grayscale conversion, adaptive thresholding
2. **Segmentation**: Detect and segment architectural elements using YOLOv8-seg model
3. **Post-processing**:
   - Merge masks by class
   - Connect gaps using morphological operations
   - Extract centerlines via skeletonization (walls)
   - Extract contours (doors, windows)
4. **Export**: Generate layer-separated DXF file

---

## Development and Testing

### Run Validation Tests

```bash
python test_validation.py
```

### Run Examples

```bash
python examples.py
```

---

## Legacy CLI

For backward compatibility, legacy CLIs are also provided:

```bash
# Basic mode
python main_legacy.py floorplan.png -o output.dxf

# Continuous line mode (segmentation-based)
python main_continuous.py floorplan.png -o output.dxf
```

---

## Contributing

Issues and pull requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License
