import sys
import os
import time
import numpy as np
import pydicom
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme,
                            PrimaryPushButton, CardWidget, PushButton, FluentIcon, ProgressBar)

# Saját modulok importálása (Ellenőrizd az elérési utat!)
from src.core.data_manager import DataManager
from src.core.segmentation.lung_segmenter import LungSegmenter
from src.utils.logger import setup_logger

log = setup_logger("Processor")


class BatchWorker(QThread):
    """Végigmegy az összes páron, szegmentál és logol."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, valid_pairs, segmenter):
        super().__init__()
        self.valid_pairs = valid_pairs
        self.segmenter = segmenter
        self.log_file = 'logged.txt'

    def write_to_file(self, message):
        """Időbélyeggel ellátott log mentése a fájlba."""
        timestamp = time.strftime("%Y.%m.%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    # A main_window.py elején a importok maradnak, csak a BatchWorker.run változik:

    def run(self):
        total = len(self.valid_pairs)
        self.log_signal.emit(f"🚀 Adatfeldolgozás indítása: {total} eset...")
        self.write_to_file(f"FUTTATÁS INDÍTÁSA - Összesen {total} fájl")

        for i, (d_path, x_path) in enumerate(self.valid_pairs):
            try:
                # --- ITT HÍVJUK AZ ÚJ BETÖLTŐT ---
                # Ez a sor helyettesíti a sima pydicom.dcmread-et
                ds, img_sitk, img_array, frame_num, width, height, ch = self.segmenter.load_file(d_path)

                # Metaadatok kinyerése a betanításhoz (DS objektumból)
                p_id = ds.PatientID if 'PatientID' in ds else "Ismeretlen"
                thickness = ds.SliceThickness if 'SliceThickness' in ds else 0.0
                spacing = ds.PixelSpacing if 'PixelSpacing' in ds else [0.0, 0.0]

                # Szegmentálás futtatása (a kinyert numpy tömbbel)
                mask = self.segmenter.segment_mask(img_array)
                px_count = np.sum(mask > 0)

                status = "✅ OK" if px_count > 0 else "⚠️ ÜRES"

                # Log üzenet: Most már tartalmazza a Dimenzókat is (width x height)
                log_msg = (f"[{i + 1}/{total}] {d_path.name} | ID: {p_id} | "
                           f"Dim: {width}x{height} | Spacing: {spacing[0]:.2f}mm | "
                           f"{status} ({px_count} px)")

                # Adatok ideiglenes tárolása (ha kellene később)
                # Itt a ciklus végén a változók felszabadulnak, így nem eszi meg a RAM-ot

                self.log_signal.emit(log_msg)
                self.write_to_file(log_msg)

            except Exception as e:
                err_msg = f"❌ HIBA ({d_path.name}): {str(e)}"
                self.log_signal.emit(err_msg)
                self.write_to_file(err_msg)

            # Progress bar frissítése
            self.progress_signal.emit(int(((i + 1) / total) * 100))

        self.log_signal.emit("✨ Feldolgozási folyamat befejeződött.")
        self.write_to_file("FELDOLGOZÁS VÉGE\n" + "=" * 60)
        self.finished.emit()


class DashboardInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("processor_interface")
        self.segmenter = LungSegmenter()
        self.dicom_dir = None
        self.xml_dir = None

        self.layout = QVBoxLayout(self)
        self._init_ui()

    def _init_ui(self):
        # Vezérlő kártya
        self.top_card = CardWidget(self)
        h_ly = QHBoxLayout(self.top_card)

        self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM mappa")
        self.xml_btn = PushButton(FluentIcon.FOLDER, "XML mappa")
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "Batch Indítása")
        self.run_btn.setEnabled(False)

        h_ly.addWidget(self.dicom_btn)
        h_ly.addWidget(self.xml_btn)
        h_ly.addStretch(1)
        h_ly.addWidget(self.run_btn)
        self.layout.addWidget(self.top_card)

        # Progress bar
        self.progress_bar = ProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        # Terminál log
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00FF00;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.layout.addWidget(SubtitleLabel("Rendszernapló (logged.txt)"))
        self.layout.addWidget(self.log_display)

        # Eseménykezelők
        self.dicom_btn.clicked.connect(self.select_dicom)
        self.xml_btn.clicked.connect(self.select_xml)
        self.run_btn.clicked.connect(self.start_batch)

    def select_dicom(self):
        p = QFileDialog.getExistingDirectory(self, "DICOM mappa")
        if p:
            self.dicom_dir = p
            self.log_display.append(f"📁 DICOM set: {p}")
            self.check_ready()

    def select_xml(self):
        p = QFileDialog.getExistingDirectory(self, "XML mappa")
        if p:
            self.xml_dir = p
            self.log_display.append(f"📝 XML set: {p}")
            self.check_ready()

    def check_ready(self):
        if self.dicom_dir and self.xml_dir:
            self.mgr = DataManager(self.dicom_dir, self.xml_dir)
            self.mgr.index_files()
            count = len(self.mgr.valid_pairs)
            self.log_display.append(f"🔍 Talált érvényes párok: {count}")
            if count > 0: self.run_btn.setEnabled(True)

    def start_batch(self):
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # Kezdő log mentése
        with open('logged.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 10} ÚJ FUTTATÁS: {time.ctime()} {'=' * 10}\n")

        self.worker = BatchWorker(self.mgr.valid_pairs, self.segmenter)
        self.worker.log_signal.connect(self.log_display.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungDx Processor v3.0")
        self.resize(800, 600)
        self.dashboard = DashboardInterface(self)
        self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Feldolgozás')
        setTheme(Theme.DARK)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())