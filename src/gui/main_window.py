# src/gui/main_window_v2.py
import sys
import os
import time
import pydicom
import shutil  # Fontos a takarításhoz
from dask.distributed import Client  # Dask kliens a párhuzamosításhoz

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme,
                            PrimaryPushButton, PushButton, CardWidget, FluentIcon, ProgressBar)

# Saját modulok importálása
from src.core.data_manager import DataManager
from src.core.processing.tumor_processor import TumorProcessor
from src.core.learning.feature_extractor import FeatureExtractor
from src.core.data_prep.annotation_parser import AnnotationParser

# --- Itt importáljuk a tanítási logikát (az előző beszélgetésből) ---
# Feltételezem, hogy ezt a fájlt létrehoztad: src/core/learning/training_logic.py
try:
    from src.core.learning.training_logic import XGBoostTrainer
except ImportError:
    print("HIBA: Nem található a src.core.learning.training_logic modul! Ellenőrizd a fájlt.")
    XGBoostTrainer = None  # Placeholder, hogy ne szálljon el az import hiba miatt azonnal


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
                    status_msg = f"[{p_id}] Feldolgozva: {os.path.basename(d_path)} ({i + 1}/{total})"
                    self.log_signal.emit(status_msg)

            except Exception as e:
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


# --- 3. ÚJ Worker a Modell Tanításhoz ---
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)  # Üzenetek küldése a GUI-nak
    finished_signal = pyqtSignal(bool)  # Jelzés, ha kész (siker/hiba)

    def __init__(self, trainer_class, *args, **kwargs):
        super().__init__()
        # Itt dinamikusan példányosítjuk a kapott tréner osztályt (Strategy pattern)
        if trainer_class is None:
            raise ValueError("Nincs Trainer osztály megadva!")
        self.trainer = trainer_class(*args, **kwargs)

    def run(self):
        # Ez a metódus fut a háttérszálon, így nem fagy le a GUI
        success = self.trainer.train(self.emit_log)
        self.finished_signal.emit(success)

    def emit_log(self, message):
        # Callback függvény, amit átadunk a trénernek
        self.log_signal.emit(message)


# --- 4. GUI Felület ---
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

        # Konfiguráció a modellhez
        self.config = {'model-name': 'lung_xgb.pkl'}
        self.resource_folder = "resources"  # Hozzunk létre egy mappát a modelleknek
        if not os.path.exists(self.resource_folder):
            os.makedirs(self.resource_folder)

        # Dask Client indítása (egyszer az alkalmazás elején)
        try:
            # LocalCluster-t indít automatikusan
            self.dask_client = Client(processes=False)
            print(f"Dask Dashboard link: {self.dask_client.dashboard_link}")
        except Exception as e:
            print(f"Nem sikerült elindítani a Dask klienst: {e}")
            self.dask_client = None

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
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "1. Indexelés")
        self.process_btn = PrimaryPushButton(FluentIcon.SYNC, "2. Feldolgozás")
        self.export_btn = PrimaryPushButton(FluentIcon.SAVE, "3. CSV Export")

        # --- ÚJ GOMB ---
        self.train_btn = PrimaryPushButton(FluentIcon.ROBOT, "4. Modell Tanítás")

        # Kezdeti állapot: minden folyamat gomb inaktív
        self.run_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.train_btn.setEnabled(False)  # Alapból tiltva

        # Elrendezés hozzáadása
        h_ly.addWidget(self.dicom_btn)
        h_ly.addWidget(self.xml_btn)

        h_ly.addStretch(1)  # Távtartó

        # Balról jobbra sorrend:
        h_ly.addWidget(self.run_btn)
        h_ly.addWidget(self.process_btn)
        h_ly.addWidget(self.export_btn)
        h_ly.addWidget(self.train_btn)  # Hozzáadva a sor végére

        self.layout.addWidget(self.top_card)

        # Progress Bar
        self.progress_bar = ProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        # Napló kijelző
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
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
        # Új signal
        self.train_btn.clicked.connect(self.start_training_process)

    # ... (select_dicom, select_xml, check_ready, start_index, on_index_finished - VÁLTOZATLANOK) ...
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
                self.run_btn.setEnabled(True)
                self.log_display.append("➡️ Kattints az '1. Indexelés' gombra!")

    def start_index(self):
        self.run_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.train_btn.setEnabled(False)

        self.log_display.append("\n--- 1. INDEXELÉS INDÍTÁSA ---")
        self.worker = BatchWorker(self.mgr.valid_pairs)
        self.worker.log_signal.connect(self.log_display.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.data_ready_signal.connect(self.on_index_finished)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def on_index_finished(self, patient_store):
        self.patient_store = patient_store
        total_p = len(patient_store)
        tumor_s = sum(1 for slices in patient_store.values() for s in slices if s['has_tumor'])

        self.log_display.append("=" * 30)
        self.log_display.append(f"📊 Összes páciens: {total_p}")
        self.log_display.append(f"📊 Daganatos szeletek: {tumor_s}")
        self.log_display.append("=" * 30)

        if tumor_s > 0:
            self.process_btn.setEnabled(True)
            self.log_display.append("\n➡️ Az indexelés kész. Kattints a '2. Feldolgozás' gombra!")
        else:
            self.log_display.append("\n⚠️ Nem találtam daganatot, a folyamat itt megáll.")

    def start_processing(self):
        if not self.patient_store: return

        # Takarítás
        processed_dir = "processed_data"
        if os.path.exists(processed_dir):
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

        self.process_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.train_btn.setEnabled(False)

        self.log_display.append("\n--- 2. FELDOLGOZÁS (ROI/GVF) INDÍTÁSA ---")
        self.progress_bar.setValue(0)

        self.processor = TumorProcessor(self.patient_store)
        self.processor.log_signal.connect(self.log_display.append)
        self.processor.log_signal.connect(self.write_to_log_file)
        self.processor.progress_signal.connect(self.progress_bar.setValue)

        self.processor.finished.connect(lambda: self.process_btn.setEnabled(True))
        self.processor.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.processor.finished.connect(self.on_processing_finished)

        self.processor.start()

    def on_processing_finished(self):
        self.log_display.append("\n✅ Feldolgozás és mentés (.npz) kész!")
        self.export_btn.setEnabled(True)
        self.log_display.append("➡️ Kattints a '3. CSV Export' gombra!")

    def start_export(self):
        self.export_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.log_display.append("\n--- 3. CSV EXPORT (JELLEMZŐK KINYERÉSE) ---")
        self.progress_bar.setValue(0)

        self.feat_worker = FeatureWorker()
        self.feat_worker.log_signal.connect(self.log_display.append)

        # Ha kész, újra aktív
        self.feat_worker.finished.connect(lambda: self.export_btn.setEnabled(True))
        # !!! ITT KAPCSOLJUK BE A TANÍTÁS GOMBOT !!!
        self.feat_worker.finished.connect(self.on_export_finished)
        self.feat_worker.start()

    def on_export_finished(self):
        # Ez a metódus hívódik meg, ha a CSV generálás kész
        self.log_display.append("\n✅ CSV Export kész!")
        self.train_btn.setEnabled(True)  # 4. Gomb aktiválása
        self.log_display.append("➡️ Kattints a '4. Modell Tanítás' gombra!")

    # --- 5. ÚJ METÓDUS: Tanítás indítása ---
    def start_training_process(self):
        csv_path = "training_data_pixelwise.csv"

        # 1. Ellenőrzés
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            QMessageBox.warning(self, "Folyamatban", "A tanítás már fut!")
            return

        if not self.dask_client:
            QMessageBox.critical(self, "Hiba", "A Dask kliens nincs inicializálva! Nem lehet tanítani.")
            return

        if not XGBoostTrainer:
            QMessageBox.critical(self, "Import Hiba", "Nem található a tanító logika (XGBoostTrainer)!")
            return

        # 2. UI frissítés
        self.train_btn.setEnabled(False)
        self.log_display.append("\n--- 4. MODELL TANÍTÁS INDÍTÁSA (XGBoost) ---")
        self.progress_bar.setValue(0)  # Tanításnál nem tudjuk a %-ot pontosan, 0-n tartjuk vagy pulzálhat

        # 3. Worker indítása
        # Itt adjuk át a paramétereket az XGBoostTrainer __init__-jének
        self.training_worker = TrainingWorker(
            XGBoostTrainer,  # Osztály referenciája
            csv_file_path=csv_path,
            resource_folder=self.resource_folder,
            config=self.config,
            client=self.dask_client
        )

        self.training_worker.log_signal.connect(self.log_display.append)
        self.training_worker.log_signal.connect(self.write_to_log_file)
        self.training_worker.finished_signal.connect(self.on_training_finished)

        self.training_worker.start()

    def on_training_finished(self, success):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        if success:
            msg = f"🎉 SIKER! A modell mentve ide: {self.resource_folder}/{self.config['model-name']}"
            self.log_display.append(msg)
            QMessageBox.information(self, "Kész", "A modell tanítása sikeresen befejeződött!")
        else:
            self.log_display.append("❌ Hiba történt a tanítás során.")
            QMessageBox.critical(self, "Hiba", "A modell tanítása közben hiba lépett fel. Lásd a logot.")


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungDx Data Manager Pro")
        self.resize(1100, 800)  # Kicsit szélesebb ablak a 4 gomb miatt
        self.dashboard = DashboardInterface(self)
        self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Adatkezelés')
        setTheme(Theme.DARK)

    # Dask kliens bezárása kilépéskor
    def closeEvent(self, event):
        if hasattr(self.dashboard, 'dask_client') and self.dashboard.dask_client:
            print("Dask kliens leállítása...")
            self.dashboard.dask_client.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())