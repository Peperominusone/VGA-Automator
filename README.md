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
