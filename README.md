# VGA-Automator

건축 도면을 Space Syntax VGA 분석용 DXF 파일로 자동 변환하는 도구

Automatically convert architectural floor plans to DXF files for Space Syntax VGA (Visibility Graph Analysis)

---

## 📋 목차 / Table of Contents

- [한국어](#korean)
- [English](#english)

---

<a name="korean"></a>
## 🇰🇷 한국어

### 프로젝트 개요

VGA-Automator는 건축 도면 이미지(PNG/JPG/PDF)를 Space Syntax VGA(Visibility Graph Analysis) 분석을 위한 DXF 파일로 자동 변환하는 도구입니다. YOLOv8 기반 객체 인식을 통해 벽체, 문, 창문, 기둥 등을 감지하고 정밀한 DXF 파일을 생성합니다.

### 주요 기능

1. **이미지 전처리**: 도면 로드, 노이즈 제거, 이진화
2. **객체 인식**: YOLOv8 모델을 활용한 벽체, 문, 창문, 기둥 등 인식
3. **윤곽선 추출**: 감지된 요소에서 정밀한 라인 추출
4. **DXF 내보내기**: 레이어별로 구분된 AutoCAD 호환 DXF 파일 생성

### 감지 가능한 객체

- Wall (벽체)
- Door (문)
- Window (창문)
- Column (기둥)
- Curtain Wall (커튼월)
- Railing (난간)
- Sliding Door (미닫이문)
- Stair Case (계단)

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

#### 3. YOLOv8 모델 다운로드

본 프로젝트는 [sanatladkat/floor-plan-object-detection](https://github.com/sanatladkat/floor-plan-object-detection) 모델을 사용합니다.

1. 저장소를 방문하여 `best.pt` 모델 파일을 다운로드
2. 다운로드한 `best.pt` 파일을 프로젝트 루트 디렉토리에 배치

또는 다음 명령어로 직접 다운로드:

```bash
# 모델 다운로드 예시 (실제 URL은 저장소에서 확인)
wget https://github.com/sanatladkat/floor-plan-object-detection/raw/main/best.pt
```

### 사용 방법

#### 기본 사용법

```bash
python main.py input_floorplan.png -o output.dxf
```

#### 고급 옵션

```bash
python main.py input_floorplan.jpg -o output.dxf \
    --confidence 0.5 \
    --scale 10.0 \
    --debug
```

#### 명령줄 옵션

- `input`: 입력 도면 이미지 (PNG, JPG, PDF)
- `-o, --output`: 출력 DXF 파일 경로 (기본값: output.dxf)
- `--model`: YOLOv8 모델 파일 경로 (기본값: best.pt)
- `--confidence`: 객체 감지 신뢰도 임계값 (기본값: 0.5)
- `--scale`: 픽셀-CAD 단위 변환 스케일 (기본값: 1.0)
- `--denoise`: 노이즈 제거 강도 (기본값: 10)
- `--doors-as-walls`: 문을 개구부가 아닌 벽으로 처리
- `--bbox-only`: 상세 윤곽선 대신 바운딩 박스 사용
- `--debug`: 중간 과정 이미지 저장
- `--no-detection`: 객체 감지 건너뛰고 윤곽선만 추출

#### 예제

```bash
# 기본 변환
python main.py samples/floorplan.png -o output/result.dxf

# 디버그 모드로 중간 과정 확인
python main.py samples/floorplan.jpg -o output/result.dxf --debug

# PDF 도면 처리
python main.py samples/floorplan.pdf -o output/result.dxf --scale 10

# 객체 감지 없이 윤곽선만 추출
python main.py samples/floorplan.png -o output/result.dxf --no-detection
```

### 프로젝트 구조

```
VGA-Automator/
├── README.md                 # 프로젝트 설명
├── requirements.txt          # 의존성 패키지
├── main.py                   # CLI 메인 진입점
├── src/
│   ├── __init__.py          
│   ├── preprocessor.py       # 이미지 전처리 모듈
│   ├── detector.py           # YOLOv8 기반 객체 인식
│   ├── contour_extractor.py  # 윤곽선 추출 및 단순화
│   └── dxf_exporter.py       # DXF 파일 생성
├── notebooks/
│   └── colab_demo.ipynb      # Google Colab 데모
├── samples/                  # 샘플 도면 폴더
└── .gitignore
```

### DXF 레이어 구조

생성된 DXF 파일은 다음과 같은 레이어로 구성됩니다:

- `WALL`: 벽체 (흰색)
- `DOOR`: 문 (파란색) 
- `WINDOW`: 창문 (청록색)
- `COLUMN`: 기둥 (마젠타)
- `CURTAIN_WALL`: 커튼월 (녹색)
- `RAILING`: 난간 (노란색)
- `SLIDING_DOOR`: 미닫이문 (빨간색)
- `STAIR`: 계단 (회색)
- `OPENING`: 개구부 (녹색 점선) - 문을 개구부로 처리할 때
- `BOUNDARY`: 경계 (빨간색)

### VGA 분석 도구 연동

생성된 DXF 파일은 다음 도구에서 사용할 수 있습니다:

1. **depthmapX**: Space Syntax VGA 분석 전문 도구
   - https://github.com/SpaceGroupUCL/depthmapX
   
2. **AutoCAD / DraftSight**: DXF 파일 편집 및 검증

3. **QGIS**: 공간 분석 및 시각화

### Google Colab 데모

모델 다운로드부터 DXF 변환까지 전 과정을 Google Colab에서 실행할 수 있습니다:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_demo.ipynb)

### 기술 스택

- **Python 3.8+**
- **OpenCV**: 이미지 전처리 및 윤곽선 추출
- **ultralytics (YOLOv8)**: 객체 인식
- **ezdxf**: DXF 파일 생성
- **NumPy, Pillow**: 이미지 처리
- **pdf2image**: PDF 지원 (선택사항)

### 라이선스

MIT License

### 참고 자료

- YOLOv8 모델 출처: [sanatladkat/floor-plan-object-detection](https://github.com/sanatladkat/floor-plan-object-detection)
- Space Syntax: [UCL Space Syntax](https://www.spacesyntax.net/)
- depthmapX: [SpaceGroupUCL/depthmapX](https://github.com/SpaceGroupUCL/depthmapX)

---

<a name="english"></a>
## 🇬🇧 English

### Project Overview

VGA-Automator is a tool that automatically converts architectural floor plan images (PNG/JPG/PDF) into DXF files for Space Syntax VGA (Visibility Graph Analysis). It uses YOLOv8-based object detection to identify walls, doors, windows, columns, and generates precise DXF files.

### Key Features

1. **Image Preprocessing**: Load, denoise, and binarize floor plans
2. **Object Detection**: Detect walls, doors, windows, columns using YOLOv8
3. **Contour Extraction**: Extract precise lines from detected elements
4. **DXF Export**: Generate layer-separated AutoCAD-compatible DXF files

### Detectable Objects

- Wall
- Door
- Window
- Column
- Curtain Wall
- Railing
- Sliding Door
- Stair Case

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

#### 3. Download YOLOv8 Model

This project uses the [sanatladkat/floor-plan-object-detection](https://github.com/sanatladkat/floor-plan-object-detection) model.

1. Visit the repository and download the `best.pt` model file
2. Place the downloaded `best.pt` file in the project root directory

Or download directly:

```bash
# Example download command (check repository for actual URL)
wget https://github.com/sanatladkat/floor-plan-object-detection/raw/main/best.pt
```

### Usage

#### Basic Usage

```bash
python main.py input_floorplan.png -o output.dxf
```

#### Advanced Options

```bash
python main.py input_floorplan.jpg -o output.dxf \
    --confidence 0.5 \
    --scale 10.0 \
    --debug
```

#### Command Line Options

- `input`: Input floor plan image (PNG, JPG, PDF)
- `-o, --output`: Output DXF file path (default: output.dxf)
- `--model`: YOLOv8 model file path (default: best.pt)
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--scale`: Pixel to CAD unit scale factor (default: 1.0)
- `--denoise`: Denoising strength (default: 10)
- `--doors-as-walls`: Treat doors as walls instead of openings
- `--bbox-only`: Use bounding boxes instead of detailed contours
- `--debug`: Save intermediate debug images
- `--no-detection`: Skip object detection, extract contours only

#### Examples

```bash
# Basic conversion
python main.py samples/floorplan.png -o output/result.dxf

# Debug mode to check intermediate steps
python main.py samples/floorplan.jpg -o output/result.dxf --debug

# Process PDF floor plan
python main.py samples/floorplan.pdf -o output/result.dxf --scale 10

# Extract contours without object detection
python main.py samples/floorplan.png -o output/result.dxf --no-detection
```

### Project Structure

```
VGA-Automator/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── main.py                   # CLI main entry point
├── src/
│   ├── __init__.py          
│   ├── preprocessor.py       # Image preprocessing module
│   ├── detector.py           # YOLOv8-based object detection
│   ├── contour_extractor.py  # Contour extraction and simplification
│   └── dxf_exporter.py       # DXF file generation
├── notebooks/
│   └── colab_demo.ipynb      # Google Colab demo
├── samples/                  # Sample floor plans
└── .gitignore
```

### DXF Layer Structure

Generated DXF files are organized into layers:

- `WALL`: Walls (white)
- `DOOR`: Doors (blue)
- `WINDOW`: Windows (cyan)
- `COLUMN`: Columns (magenta)
- `CURTAIN_WALL`: Curtain walls (green)
- `RAILING`: Railings (yellow)
- `SLIDING_DOOR`: Sliding doors (red)
- `STAIR`: Stairs (gray)
- `OPENING`: Openings (green dashed) - when doors are treated as openings
- `BOUNDARY`: Boundaries (red)

### VGA Analysis Tool Integration

The generated DXF files can be used with:

1. **depthmapX**: Professional Space Syntax VGA analysis tool
   - https://github.com/SpaceGroupUCL/depthmapX
   
2. **AutoCAD / DraftSight**: DXF file editing and validation

3. **QGIS**: Spatial analysis and visualization

### Google Colab Demo

Run the complete pipeline from model download to DXF conversion in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_demo.ipynb)

### Tech Stack

- **Python 3.8+**
- **OpenCV**: Image preprocessing and contour extraction
- **ultralytics (YOLOv8)**: Object detection
- **ezdxf**: DXF file generation
- **NumPy, Pillow**: Image processing
- **pdf2image**: PDF support (optional)

### License

MIT License

### References

- YOLOv8 Model: [sanatladkat/floor-plan-object-detection](https://github.com/sanatladkat/floor-plan-object-detection)
- Space Syntax: [UCL Space Syntax](https://www.spacesyntax.net/)
- depthmapX: [SpaceGroupUCL/depthmapX](https://github.com/SpaceGroupUCL/depthmapX)
