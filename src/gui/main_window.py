# src/gui/main_window.py
import sys
import os
import time
import pydicom
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme,
                            PrimaryPushButton, PushButton, CardWidget, FluentIcon, ProgressBar)

# Saját modulok importálása
from src.core.data_manager import DataManager
from src.core.processing.tumor_processor import TumorProcessor
from src.core.learning.feature_extractor import FeatureExtractor
from src.core.data_prep.annotation_parser import AnnotationParser


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
                # Metaadatok beolvasása (ID kinyerése)
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

                # Logolás: Páciens ID feltüntetése!
                if i % 20 == 0 or i == total - 1:
                    status_msg = f"[{p_id}] Feldolgozva: {os.path.basename(d_path)} ({i + 1}/{total})"
                    self.log_signal.emit(status_msg)

            except Exception as e:
                # Hibánál is próbáljuk kiírni a fájl nevét
                err_msg = f"⚠️ Hiba [{os.path.basename(d_path)}]: {str(e)}"
                self.log_signal.emit(err_msg)
                self.write_to_log_file(err_msg)

            self.progress_signal.emit(int(((i + 1) / total) * 100))

        self.data_ready_signal.emit(self.patient_store)
        self.finished.emit()


# --- 2. Worker a Feature Extraction-höz ---
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
            extractor = FeatureExtractor(data_dir="processed_data")
            df = extractor.extract_features()

            if df is not None and not df.empty:
                self.log_signal.emit(f"✅ Siker! {len(df)} sor generálva.")
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
        # Felső vezérlő panel
        self.top_card = CardWidget(self)
        h_ly = QHBoxLayout(self.top_card)

        # Mappa választó gombok (Bal oldal)
        self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM")
        self.xml_btn = PushButton(FluentIcon.FOLDER, "XML")

        # Folyamat gombok (Jobb oldal, sorrendben)
        # PrimaryPushButton-t használunk, hogy kékek legyenek, ha aktívak
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "1. Indexelés")
        self.process_btn = PrimaryPushButton(FluentIcon.SYNC, "2. Feldolgozás")
        self.export_btn = PrimaryPushButton(FluentIcon.SAVE, "3. CSV Export")

        # Kezdeti állapot: minden folyamat gomb inaktív (szürke)
        self.run_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        # Elrendezés hozzáadása
        h_ly.addWidget(self.dicom_btn)
        h_ly.addWidget(self.xml_btn)

        h_ly.addStretch(1)  # Távtartó középre

        # Balról jobbra sorrend:
        h_ly.addWidget(self.run_btn)
        h_ly.addWidget(self.process_btn)
        h_ly.addWidget(self.export_btn)

        self.layout.addWidget(self.top_card)

        # Progress Bar
        self.progress_bar = ProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        # Napló kijelző
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        # Sötét háttér, zöld betűk, monospace font
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a; 
                color: #00ff00; 
                font-family: Consolas;
                font-size: 12px;
            }
        """)
        self.layout.addWidget(SubtitleLabel("Rendszernapló"))
        self.layout.addWidget(self.log_display)

        # Signalok bekötése
        self.dicom_btn.clicked.connect(self.select_dicom)
        self.xml_btn.clicked.connect(self.select_xml)
        self.run_btn.clicked.connect(self.start_index)
        self.process_btn.clicked.connect(self.start_processing)
        self.export_btn.clicked.connect(self.start_export)

    def select_dicom(self):
        p = QFileDialog.getExistingDirectory(self, "DICOM Mappa")
        if p:
            self.dicom_dir = p
            msg = f"📂 DICOM Mappa kiválasztva: {self.dicom_dir}"
            self.log_display.append(msg)
            self.write_to_log_file(msg)
            self.check_ready()

    def select_xml(self):
        p = QFileDialog.getExistingDirectory(self, "XML Mappa")
        if p:
            self.xml_dir = p
            msg = f"📝 XML Mappa kiválasztva: {self.xml_dir}"
            self.log_display.append(msg)
            self.write_to_log_file(msg)
            self.check_ready()

    def check_ready(self):
        if self.dicom_dir and self.xml_dir:
            self.log_display.append("🔍 Fájlok ellenőrzése...")
            self.mgr = DataManager(self.dicom_dir, self.xml_dir)
            self.mgr.index_files()

            count = len(self.mgr.valid_pairs)
            msg = f"✅ Talált párok száma: {count}"
            self.log_display.append(msg)

            if count > 0:
                self.run_btn.setEnabled(True)  # 1. Gomb aktiválása (Kék lesz)
                self.log_display.append("➡️ Kattints az '1. Indexelés' gombra!")

    def start_index(self):
        # Gombok tiltása futás közben
        self.run_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        self.log_display.append("\n--- 1. INDEXELÉS INDÍTÁSA ---")
        self.worker = BatchWorker(self.mgr.valid_pairs)
        self.worker.log_signal.connect(self.log_display.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.data_ready_signal.connect(self.on_index_finished)

        # Ha kész, visszakapcsoljuk az indexelést (ha újra akarnánk futtatni)
        # De a fő cél a következő gomb aktiválása
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def on_index_finished(self, patient_store):
        self.patient_store = patient_store

        # Statisztika kiírása
        total_p = len(patient_store)
        tumor_s = sum(1 for slices in patient_store.values() for s in slices if s['has_tumor'])

        self.log_display.append("=" * 30)
        self.log_display.append(f"📊 Összes páciens: {total_p}")
        self.log_display.append(f"📊 Daganatos szeletek: {tumor_s}")
        self.log_display.append("=" * 30)

        if tumor_s > 0:
            self.process_btn.setEnabled(True)  # 2. Gomb aktiválása (Kék lesz)
            self.log_display.append("\n➡️ Az indexelés kész. Kattints a '2. Feldolgozás' gombra!")
        else:
            self.log_display.append("\n⚠️ Nem találtam daganatot, a folyamat itt megáll.")

    def start_processing(self):
        if not self.patient_store: return

        # --- ÚJ RÉSZ: TAKARÍTÁS ---
        import shutil
        processed_dir = "processed_data"
        if os.path.exists(processed_dir):
            # Töröljük a régi fájlokat, hogy ne keveredjenek
            self.log_display.append("🧹 Régi feldolgozott adatok törlése...")
            for filename in os.listdir(processed_dir):
                file_path = os.path.join(processed_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Nem sikerült törölni: {file_path}. Ok: {e}")
        # --------------------------

        # Gombok tiltása futás közben
        self.process_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        self.log_display.append("\n--- 2. FELDOLGOZÁS (ROI/GVF) INDÍTÁSA ---")
        self.progress_bar.setValue(0)

        self.processor = TumorProcessor(self.patient_store)
        self.processor.log_signal.connect(self.log_display.append)
        self.processor.log_signal.connect(self.write_to_log_file)
        self.processor.progress_signal.connect(self.progress_bar.setValue)

        # Ha kész:
        self.processor.finished.connect(lambda: self.process_btn.setEnabled(True))
        self.processor.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.processor.finished.connect(self.on_processing_finished)

        self.processor.start()

    def on_processing_finished(self):
        self.log_display.append("\n✅ Feldolgozás és mentés (.npz) kész!")
        self.export_btn.setEnabled(True)  # 3. Gomb aktiválása (Kék lesz)
        self.log_display.append("➡️ Kattints a '3. CSV Export' gombra!")

    def start_export(self):
        self.export_btn.setEnabled(False)  # Futás alatt inaktív
        self.log_display.append("\n--- 3. CSV EXPORT (JELLEMZŐK KINYERÉSE) ---")
        self.progress_bar.setValue(0)  # Itt nem tudunk pontos %-ot, de jelezzük hogy indult

        self.feat_worker = FeatureWorker()
        self.feat_worker.log_signal.connect(self.log_display.append)

        # Ha kész, újra aktív
        self.feat_worker.finished.connect(lambda: self.export_btn.setEnabled(True))
        self.feat_worker.start()


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungDx Data Manager Pro")
        self.resize(1000, 750)
        self.dashboard = DashboardInterface(self)
        self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Adatkezelés')
        setTheme(Theme.DARK)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())