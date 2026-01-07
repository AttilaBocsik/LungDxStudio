import sys
import os
import pydicom
import time
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme,
                            PrimaryPushButton, CardWidget, PushButton, FluentIcon, ProgressBar)

# Saját modulok importálása
from src.core.data_manager import DataManager
from src.core.segmentation.lung_segmenter import LungSegmenter
from src.core.data_prep.annotation_parser import AnnotationParser


class BatchWorker(QThread):
    """
    Memóriakímélő indexelő: Csak a metaadatokat olvassa be.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    data_ready_signal = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, valid_pairs):
        super().__init__()
        self.valid_pairs = valid_pairs
        self.patient_store = {}
        self.log_file = "app.log"

    def write_to_log_file(self, message):
        """Időbélyeggel ellátott üzenet írása az app.log fájlba."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def run(self):
        total = len(self.valid_pairs)
        msg = f"🚀 Metaadatok és Annotációk indexelése {total} szelethez..."
        self.log_signal.emit(msg)
        self.write_to_log_file(msg)

        for i, (d_path, x_path) in enumerate(self.valid_pairs):
            try:
                # 1. DICOM Metaadatok (Lazy Loading)
                ds_meta = pydicom.dcmread(str(d_path), stop_before_pixels=True)
                p_id = ds_meta.PatientID if 'PatientID' in ds_meta else "Ismeretlen"

                # 2. XML Annotációk kinyerése (Pascal VOC)
                # Itt használjuk az új AnnotationParser-t
                annotations = AnnotationParser.parse_voc_xml(str(x_path))

                # 3. Adatcsomag összeállítása
                slice_meta = {
                    "patient_id": p_id,
                    "img_name": os.path.basename(d_path),
                    "path": str(d_path),
                    "xml_path": str(x_path),
                    "width": getattr(ds_meta, 'Rows', 512),
                    "height": getattr(ds_meta, 'Columns', 512),
                    "annotations": annotations,  # <--- Új: Itt vannak a daganat adatok
                    "has_tumor": len(annotations) > 0,  # Gyors szűréshez
                    "thickness": float(getattr(ds_meta, 'SliceThickness', 0.0)),
                    "spacing": getattr(ds_meta, 'PixelSpacing', [1.0, 1.0])
                }

                if p_id not in self.patient_store:
                    self.patient_store[p_id] = []
                self.patient_store[p_id].append(slice_meta)

                if i % 20 == 0 or i == total - 1:
                    status_msg = f"Indexelve: {i + 1}/{total} (Talált daganat: {len(annotations) > 0})"
                    self.log_signal.emit(status_msg)
                    self.write_to_log_file(status_msg)

            except Exception as e:
                err_msg = f"⚠️ Hiba: {os.path.basename(d_path)} - {str(e)}"
                self.log_signal.emit(err_msg)
                self.write_to_log_file(err_msg)

            self.progress_signal.emit(int(((i + 1) / total) * 100))

        self.data_ready_signal.emit(self.patient_store)
        self.finished.emit()


class DashboardInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("dashboard_interface")

        self.layout = QVBoxLayout(self)
        self.dicom_dir = None
        self.xml_dir = None
        self.mgr = None
        self.log_file = "app.log"

        self._init_ui()

    def write_to_log_file(self, message):
        """Segédfüggvény a GUI műveletek naplózásához."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [GUI] {message}\n")

    def _init_ui(self):
        self.top_card = CardWidget(self)
        h_ly = QHBoxLayout(self.top_card)

        self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM Mappa")
        self.xml_btn = PushButton(FluentIcon.FOLDER, "XML Mappa")
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "Indexelés Indítása")
        self.run_btn.setEnabled(False)

        h_ly.addWidget(self.dicom_btn)
        h_ly.addWidget(self.xml_btn)
        h_ly.addStretch(1)
        h_ly.addWidget(self.run_btn)
        self.layout.addWidget(self.top_card)

        self.progress_bar = ProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                border-radius: 4px;
            }
        """)
        self.layout.addWidget(SubtitleLabel("Feldolgozási Napló (app.log)"))
        self.layout.addWidget(self.log_display)

        self.dicom_btn.clicked.connect(self.select_dicom)
        self.xml_btn.clicked.connect(self.select_xml)
        self.run_btn.clicked.connect(self.start_index)

    def select_dicom(self):
        p = QFileDialog.getExistingDirectory(self, "Válassz DICOM mappát")
        if p:
            self.dicom_dir = p
            msg = f"📁 DICOM forrás kijelölve: {p}"
            self.log_display.append(msg)
            self.write_to_log_file(msg)
            self.check_ready()

    def select_xml(self):
        p = QFileDialog.getExistingDirectory(self, "Válassz XML mappát")
        if p:
            self.xml_dir = p
            msg = f"📝 XML forrás kijelölve: {p}"
            self.log_display.append(msg)
            self.write_to_log_file(msg)
            self.check_ready()

    def check_ready(self):
        if self.dicom_dir and self.xml_dir:
            msg = "🔍 Érvényes párok keresése..."
            self.log_display.append(msg)
            self.write_to_log_file(msg)

            self.mgr = DataManager(self.dicom_dir, self.xml_dir)
            self.mgr.index_files()
            count = len(self.mgr.valid_pairs)

            res_msg = f"✅ Talált érvényes párok: {count}"
            self.log_display.append(res_msg)
            self.write_to_log_file(res_msg)

            if count > 0:
                self.run_btn.setEnabled(True)

    def start_index(self):
        self.run_btn.setEnabled(False)
        # self.log_display.clear()  <-- ELTÁVOLÍTVA: Így megmarad a korábbi szöveg
        self.log_display.append(f"\n--- Új folyamat indult: {time.ctime()} ---")
        self.write_to_log_file("--- BATCH INDEXELÉS START ---")

        self.progress_bar.setValue(0)

        self.worker = BatchWorker(self.mgr.valid_pairs)
        self.worker.log_signal.connect(self.log_display.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.data_ready_signal.connect(self.show_summary)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def show_summary(self, patient_store):
        total_p = len(patient_store)
        total_s = sum(len(s) for s in patient_store.values())
        # Megszámoljuk a daganatos szeleteket
        tumor_s = sum(1 for slices in patient_store.values() for s in slices if s['has_tumor'])

        summary = [
            "\n" + "=" * 40,
            "📊 ADATHALMAZ STATISZTIKA (ANNOTÁLT)",
            f"Összes egyedi páciens: {total_p}",
            f"Összes szelet: {total_s}",
            f"Daganatos szeletek száma: {tumor_s}  <--",
            "=" * 40
        ]

        for p_id, slices in list(patient_store.items())[:15]:
            t_count = sum(1 for s in slices if s['has_tumor'])
            summary.append(f"• {p_id}: {len(slices)} szelet (Ebből daganatos: {t_count})")

        for line in summary:
            self.log_display.append(line)
            self.write_to_log_file(line)


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungDx Data Manager Pro")
        self.resize(900, 700)
        self.dashboard = DashboardInterface(self)
        self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Indexelés')
        setTheme(Theme.DARK)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())