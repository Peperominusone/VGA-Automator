"""
VGA Automator - PyQt6 데스크톱 앱
드래그 앤 드롭 지원
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFileDialog, QMessageBox,
    QSlider, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, filename="debug.log", encoding="utf-8")

class DropZone(QLabel):
    """드래그 앤 드롭 영역"""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(300)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 15px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #666;
            }
        """)
        self.setText("📂 도면 이미지를 여기에 드래그하세요\n\n또는 클릭하여 파일 선택")
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #4CAF50;
                    border-radius: 15px;
                    background-color: #f0fff0;
                    font-size: 16px;
                    color: #666;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 15px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #666;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            file_path = files[0]
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                self.file_dropped.emit(file_path)
            else:
                QMessageBox.warning(self, "지원하지 않는 형식", "PNG, JPG, PDF 파일만 지원합니다.")
        self.dragLeaveEvent(event)
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "도면 선택",
            "",
            "이미지 파일 (*.png *.jpg *.jpeg *.pdf)"
        )
        if file_path:
            self.file_dropped.emit(file_path)
    
    def set_preview(self, pixmap: QPixmap, filename: str):
        """미리보기 이미지 설정"""
        scaled = pixmap.scaled(
            self.width() - 20, self.height() - 60,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setText("")  # Clear text when showing image


class ConversionWorker(QThread):
    """변환 작업 스레드 (UI 블로킹 방지)"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, dict)  # output_path, stats
    error = pyqtSignal(str)
    
    def __init__(self, image_path: str, output_path: str, settings: dict):
        super().__init__()
        self.image_path = image_path
        self.output_path = output_path
        self.settings = settings
    
    def run(self):
        try:
            from src.preprocessor import Preprocessor
            from src.segmentation_detector import ContinuousWallExtractor, ElementType
            from src.dxf_exporter_continuous import DXFExporterContinuous
            
            stats = {'walls': 0, 'doors': 0, 'windows': 0}
            
            # 1. 이미지 로드 및 전처리
            self.progress.emit(10, "이미지 로드 중...")
            preprocessor = Preprocessor()
            image = preprocessor.load_image(self.image_path)
            
            self.progress.emit(25, "이미지 전처리 중...")
            preprocessed = preprocessor.preprocess(self.image_path)
            
            # 2. 요소 추출
            self.progress.emit(40, "건축 요소 감지 중...")
            extractor = ContinuousWallExtractor()
            elements = extractor.extract_all_elements(
                image,
                preprocessed['binary'],
                model_path=self.settings.get('model_path', 'best.pt'),
                confidence=self.settings.get('confidence', 0.5)
            )

            logging.info(f"elements type: {type(elements)}")
            logging.info(f"elements keys: {getattr(elements, 'keys', lambda: 'No keys method')()}")
            
            # 통계 수집
            self.progress.emit(70, "윤곽선 처리 중...")
            for elem_type, element in elements.items():
                if elem_type == ElementType.WALL and element.polylines:
                    stats['walls'] = len(element.polylines)
                elif elem_type == ElementType.DOOR and element.contours:
                    stats['doors'] = len(element.contours)
                elif elem_type == ElementType.WINDOW and element.contours:
                    stats['windows'] = len(element.contours)
            
           # 3. DXF 생성
            self.progress.emit(85, "DXF 파일 생성 중...")
            h, w = image.shape[:2]
            exporter = DXFExporterContinuous(self.output_path)  # 참고: 생성자에서 output_path 필요
            exporter.export_elements(elements, image_height=h)
            
            # 저장
            self.progress.emit(95, "파일 저장 중...")
            exporter.save()
            
            self.progress.emit(100, "완료!")
            self.finished.emit(self.output_path, stats)
            
        except FileNotFoundError as e:
            self.error.emit(f"모델 파일을 찾을 수 없습니다.\n{e}\n\nbest.pt 파일을 프로젝트 폴더에 넣어주세요.")
        except Exception as e:
            self.error.emit(f"변환 중 오류 발생:\n{str(e)}")


class SettingsPanel(QGroupBox):
    """설정 패널"""
    
    def __init__(self):
        super().__init__("⚙️ 설정")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 감지 신뢰도
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("감지 신뢰도:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        layout.addLayout(conf_layout)
        
        # 갭 연결 크기
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("갭 연결 크기 (px):"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(5, 50)
        self.gap_spin.setValue(15)
        gap_layout.addWidget(self.gap_spin)
        layout.addLayout(gap_layout)
        
        # 출력 스케일
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("출력 스케일:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        scale_layout.addWidget(self.scale_spin)
        layout.addLayout(scale_layout)
        
        # 디버그 옵션
        self.debug_check = QCheckBox("디버그 이미지 저장")
        layout.addWidget(self.debug_check)
        
        self.setLayout(layout)
    
    def get_settings(self) -> dict:
        return {
            'confidence': self.conf_slider.value() / 100,
            'gap_size': self.gap_spin.value(),
            'scale': self.scale_spin.value(),
            'debug': self.debug_check.isChecked()
        }


class ResultPanel(QGroupBox):
    """결과 표시 패널"""
    
    def __init__(self):
        super().__init__("📊 변환 결과")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 통계 라벨들
        self.wall_label = QLabel("벽체: -")
        self.door_label = QLabel("문: -")
        self.window_label = QLabel("창문: -")
        
        for label in [self.wall_label, self.door_label, self.window_label]:
            label.setStyleSheet("font-size: 14px; padding: 5px;")
            layout.addWidget(label)
        
        # 상태 메시지
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_stats(self, stats: dict):
        self.wall_label.setText(f"🧱 벽체: {stats.get('walls', 0)}개 폴리라인")
        self.door_label.setText(f"🚪 문: {stats.get('doors', 0)}개")
        self.window_label.setText(f"🪟 창문: {stats.get('windows', 0)}개")
    
    def set_status(self, message: str, is_error: bool = False):
        color = "#d32f2f" if is_error else "#666"
        self.status_label.setStyleSheet(f"color: {color}; font-style: italic;")
        self.status_label.setText(message)
    
    def clear(self):
        self.wall_label.setText("🧱 벽체: -")
        self.door_label.setText("🚪 문: -")
        self.window_label.setText("🪟 창문: -")
        self.status_label.setText("")


class VGAAutomatorApp(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGA Automator - Space Syntax 도면 변환")
        self.setMinimumSize(900, 650)
        self.current_image_path = None
        self.worker = None
        
        self.init_ui()
        self.apply_styles()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 헤더
        header = QLabel("🏗️ VGA Automator")
        header.setFont(QFont("", 28, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        subtitle = QLabel("Space Syntax VGA 분석용 도면 자동 변환 도구")
        subtitle.setStyleSheet("color: #666; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #ddd;")
        main_layout.addWidget(line)
        
        # 메인 컨텐츠 (좌우 분할)
        content_layout = QHBoxLayout()
        
        # 왼쪽: 드롭존
        left_panel = QVBoxLayout()
        
        drop_label = QLabel("📤 도면 업로드")
        drop_label.setFont(QFont("", 14, QFont.Weight.Bold))
        left_panel.addWidget(drop_label)
        
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_dropped)
        left_panel.addWidget(self.drop_zone)
        
        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet("color: #888; font-size: 12px;")
        left_panel.addWidget(self.file_info_label)
        
        content_layout.addLayout(left_panel, stretch=2)
        
        # 오른쪽: 설정 및 결과
        right_panel = QVBoxLayout()
        
        # 설정 패널
        self.settings_panel = SettingsPanel()
        right_panel.addWidget(self.settings_panel)
        
        # 변환 버튼
        self.convert_btn = QPushButton("🚀 DXF 변환 시작")
        self.convert_btn.setMinimumHeight(50)
        self.convert_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.start_conversion)
        right_panel.addWidget(self.convert_btn)
        
        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        right_panel.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #666;")
        right_panel.addWidget(self.progress_label)
        
        # 결과 패널
        self.result_panel = ResultPanel()
        right_panel.addWidget(self.result_panel)
        
        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)
        
        # 상태바
        self.statusBar().showMessage("도면 이미지를 드래그하여 시작하세요")
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
    
    def on_file_dropped(self, file_path: str):
        """파일 드롭/선택 시 호출"""
        self.current_image_path = file_path
        
        # 미리보기 표시
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.drop_zone.set_preview(pixmap, Path(file_path).name)
            
            # 파일 정보
            file_size = Path(file_path).stat().st_size / 1024  # KB
            self.file_info_label.setText(
                f"📄 {Path(file_path).name} ({file_size:.1f} KB) | "
                f"📐 {pixmap.width()} x {pixmap.height()} px"
            )
        else:
            # PDF 또는 읽을 수 없는 이미지
            file_size = Path(file_path).stat().st_size / 1024  # KB
            file_ext = Path(file_path).suffix.upper()
            if file_ext == '.PDF':
                self.file_info_label.setText(
                    f"📄 {Path(file_path).name} ({file_size:.1f} KB) | "
                    f"PDF 파일 (미리보기 불가)"
                )
            else:
                self.file_info_label.setText(
                    f"📄 {Path(file_path).name} ({file_size:.1f} KB) | "
                    f"⚠️ 이미지를 읽을 수 없습니다"
                )
                QMessageBox.warning(
                    self,
                    "이미지 로드 실패",
                    f"이미지 파일을 읽을 수 없습니다.\n다른 파일을 선택해주세요."
                )
                return
        
        self.convert_btn.setEnabled(True)
        self.result_panel.clear()
        self.statusBar().showMessage(f"로드 완료: {file_path}")
    
    def start_conversion(self):
        """변환 시작"""
        if not self.current_image_path:
            return
        
        # 저장 경로 선택
        default_name = Path(self.current_image_path).stem + ".dxf"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "DXF 파일 저장",
            default_name,
            "DXF 파일 (*.dxf)"
        )
        
        if not output_path:
            return
        
        # UI 상태 변경
        self.convert_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("준비 중...")
        self.result_panel.clear()
        
        # 워커 스레드 시작
        settings = self.settings_panel.get_settings()
        self.worker = ConversionWorker(
            self.current_image_path,
            output_path,
            settings
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, value: int, message: str):
        """진행 상태 업데이트"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_finished(self, output_path: str, stats: dict):
        """변환 완료"""
        self.convert_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        self.result_panel.update_stats(stats)
        self.result_panel.set_status(f"✅ 저장 완료: {Path(output_path).name}")
        
        self.statusBar().showMessage(f"변환 완료: {output_path}")
        
        # 워커 정리
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        # 완료 메시지
        reply = QMessageBox.information(
            self,
            "변환 완료",
            f"DXF 파일이 성공적으로 저장되었습니다!\n\n"
            f"📁 {output_path}\n\n"
            f"🧱 벽체: {stats['walls']}개\n"
            f"🚪 문: {stats['doors']}개\n"
            f"🪟 창문: {stats['windows']}개\n\n"
            f"파일 위치를 열까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            import subprocess
            import platform
            
            folder = str(Path(output_path).parent)
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", folder], check=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", folder], check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", folder], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                QMessageBox.warning(
                    self,
                    "폴더 열기 실패",
                    f"파일 탐색기를 열 수 없습니다.\n수동으로 폴더를 열어주세요:\n{folder}"
                )
    
    def on_error(self, error_msg: str):
        """오류 발생"""
        self.convert_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        self.result_panel.set_status(f"❌ 오류 발생", is_error=True)
        self.statusBar().showMessage("오류 발생")
        
        # 워커 정리
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        QMessageBox.critical(self, "오류", error_msg)
    
    def closeEvent(self, event):
        """앱 종료 시"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "종료 확인",
                "변환이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            # 워커 종료 및 대기
            self.worker.requestInterruption()
            if not self.worker.wait(2000):  # 2초 대기
                self.worker.terminate()
                self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 크로스 플랫폼 일관된 스타일
    
    window = VGAAutomatorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
