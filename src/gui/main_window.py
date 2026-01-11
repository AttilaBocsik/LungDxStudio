import sys
import os
import time
import pydicom
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme,
                            PrimaryPushButton, CardWidget, PushButton, FluentIcon, ProgressBar)

# Saját modulok importálása
from src.core.data_manager import DataManager
from src.core.segmentation.lung_segmenter import LungSegmenter
from src.core.data_prep.annotation_parser import AnnotationParser
from src.core.processing.tumor_processor import TumorProcessor
# --- ÚJ IMPORT ---
from src.core.learning.feature_extractor import FeatureExtractor


# --- 1. Worker az indexeléshez ---
class BatchWorker(QThread):
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
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def run(self):
        total = len(self.valid_pairs)
        msg = f"🚀 Metaadatok indexelése {total} szelethez..."
        self.log_signal.emit(msg)
        self.write_to_log_file(msg)

        for i, (d_path, x_path) in enumerate(self.valid_pairs):
            try:
                ds_meta = pydicom.dcmread(str(d_path), stop_before_pixels=True)
                p_id = ds_meta.PatientID if 'PatientID' in ds_meta else "Ismeretlen"
                annotations = AnnotationParser.parse_voc_xml(str(x_path))

                slice_meta = {
                    "patient_id": p_id,
                    "img_name": os.path.basename(d_path),
                    "path": str(d_path),
                    "xml_path": str(x_path),
                    "width": getattr(ds_meta, 'Rows', 512),
                    "height": getattr(ds_meta, 'Columns', 512),
                    "annotations": annotations,
                    "has_tumor": len(annotations) > 0,
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


# --- 2. Worker a Feature Extraction-höz (ÚJ OSZTÁLY) ---
class FeatureWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.log_file = "app.log"

    def write_to_log_file(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def run(self):
        self.log_signal.emit("📊 Jellemzők kinyerésének indítása (Gabor filterek)...")
        self.write_to_log_file("--- FEATURE EXTRACTION START ---")

        try:
            # Példányosítjuk a FeatureExtractor-t
            extractor = FeatureExtractor(data_dir="processed_data")

            # Lefuttatjuk a kinyerést
            df = extractor.extract_features()

            if df is not None and not df.empty:
                self.log_signal.emit(f"✅ Siker! {len(df)} sor generálva.")

                # Mentés CSV-be
                csv_path = "training_data_pixelwise.csv"
                extractor.save_to_csv(df, csv_path)

                msg = f"💾 CSV mentve: {csv_path}"
                self.log_signal.emit(msg)
                self.write_to_log_file(msg)
            else:
                self.log_signal.emit("⚠️ Nem keletkezett adat (üres DataFrame).")

        except Exception as e:
            err = f"❌ Hiba a feature kinyerésnél: {str(e)}"
            self.log_signal.emit(err)
            self.write_to_log_file(err)
            import traceback
            print(traceback.format_exc())

        self.finished.emit()


# --- 3. GUI Felület ---
class DashboardInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("dashboard_interface")
        self.layout = QVBoxLayout(self)
        self.dicom_dir = None
        self.xml_dir = None
        self.mgr = None
        self.patient_store = None
        self.log_file = "app.log"
        self._init_ui()

    def write_to_log_file(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [GUI] {message}\n")

    def _init_ui(self):
        self.top_card = CardWidget(self)
        h_ly = QHBoxLayout(self.top_card)

        self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM")
        self.xml_btn = PushButton(FluentIcon.FOLDER, "XML")

        # Gombok sorrendben:
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "1. Indexelés")
        self.run_btn.setEnabled(False)

        self.process_btn = PushButton(FluentIcon.SYNC, "2. Feldolgozás")
        self.process_btn.setEnabled(False)

        # ÚJ GOMB:
        self.export_btn = PushButton(FluentIcon.SAVE, "3. CSV Export")
        self.export_btn.setEnabled(False)  # Csak feldolgozás után aktív

        h_ly.addWidget(self.dicom_btn)
        h_ly.addWidget(self.xml_btn)
        h_ly.addStretch(1)
        h_ly.addWidget(self.process_btn)
        h_ly.addWidget(self.export_btn)  # Hozzáadjuk a sorhoz
        h_ly.addWidget(self.run_btn)

        self.layout.addWidget(self.top_card)
        self.progress_bar = ProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: #1a1a1a; color: #00ff00; font-family: Consolas;")
        self.layout.addWidget(SubtitleLabel("Napló"))
        self.layout.addWidget(self.log_display)

        # Signalok
        self.dicom_btn.clicked.connect(self.select_dicom)
        self.xml_btn.clicked.connect(self.select_xml)
        self.run_btn.clicked.connect(self.start_index)
        self.process_btn.clicked.connect(self.start_processing)

        # ÚJ SIGNAL:
        self.export_btn.clicked.connect(self.start_export)

    def select_dicom(self):
        p = QFileDialog.getExistingDirectory(self, "DICOM Mappa")
        if p:
            self.dicom_dir = p;
            self.check_ready()

    def select_xml(self):
        p = QFileDialog.getExistingDirectory(self, "XML Mappa")
        if p:
            self.xml_dir = p;
            self.check_ready()

    def check_ready(self):
        if self.dicom_dir and self.xml_dir:
            self.mgr = DataManager(self.dicom_dir, self.xml_dir)
            self.mgr.index_files()
            if len(self.mgr.valid_pairs) > 0: self.run_btn.setEnabled(True)
            self.log_display.append(f"✅ Párok: {len(self.mgr.valid_pairs)}")

    def start_index(self):
        self.run_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log_display.append("\n--- Indexelés Start ---")
        self.worker = BatchWorker(self.mgr.valid_pairs)
        self.worker.log_signal.connect(self.log_display.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.data_ready_signal.connect(self.on_index_finished)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def on_index_finished(self, patient_store):
        self.patient_store = patient_store
        tumor_s = sum(1 for slices in patient_store.values() for s in slices if s['has_tumor'])
        self.log_display.append(f"📊 Daganatos szeletek: {tumor_s}")
        if tumor_s > 0:
            self.process_btn.setEnabled(True)  # Engedélyezzük a 2. lépést

    def start_processing(self):
        if not self.patient_store: return
        self.process_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log_display.append("\n--- Feldolgozás Start ---")

        self.processor = TumorProcessor(self.patient_store)
        self.processor.log_signal.connect(self.log_display.append)
        self.processor.log_signal.connect(self.write_to_log_file)
        self.processor.progress_signal.connect(self.progress_bar.setValue)

        # Ha végzett, engedélyezzük a 3. lépést (Export)
        self.processor.finished.connect(lambda: self.process_btn.setEnabled(True))
        self.processor.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.processor.finished.connect(self.on_processing_finished)  # ÚJ

        self.processor.start()

    def on_processing_finished(self):
        """Ez fut le, ha a GVF/ROI feldolgozás kész."""
        self.log_display.append("\n✅ Feldolgozás kész! Most már exportálhatod a CSV-t.")
        self.export_btn.setEnabled(True)  # Gomb aktiválása

    def start_export(self):
        """Ez indítja a FeatureExtraction folyamatot."""
        self.export_btn.setEnabled(False)
        self.log_display.append("\n--- CSV Export Start ---")

        # Indítjuk a FeatureWorkert
        self.feat_worker = FeatureWorker()
        self.feat_worker.log_signal.connect(self.log_display.append)
        # Nincs progress bar signal, mert a FeatureExtractorban a tqdm konzolra ír,
        # de a log_signal-on kapunk infót a végén.

        self.feat_worker.finished.connect(lambda: self.export_btn.setEnabled(True))
        self.feat_worker.start()


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungDx Data Manager Pro")
        self.resize(950, 700)
        self.dashboard = DashboardInterface(self)
        self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Indexelés')
        setTheme(Theme.DARK)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())