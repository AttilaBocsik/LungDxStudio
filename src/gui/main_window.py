# src/gui/main_window.py
import sys
import os


# Alapvető író osztály a konzol nélküli módhoz (későbbi használatra)
class NullWriter:
    """Megakadályozza a 'NoneType' hibát GUI módban."""

    def write(self, arg): pass

    def flush(self): pass


def run_application():
    """Ebben a függvényben történik minden import és inicializálás,
    hogy a hibákat el tudjuk kapni."""

    # 1. NEHÉZ IMPORTÁLÁSOK (XGBoost, Dask, PyQt6)
    import time
    import pydicom
    import shutil
    import dask
    from dask.distributed import Client
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox
    from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme, BodyLabel,
                                PrimaryPushButton, PushButton, CardWidget, FluentIcon, ProgressBar, SwitchButton)

    # 2. SAJÁT MODULOK IMPORTÁLÁSA
    try:
        from src.core.data_manager import DataManager
        from src.core.processing.tumor_processor import TumorProcessor
        from src.core.learning.feature_extractor import FeatureExtractor
        from src.core.data_prep.annotation_parser import AnnotationParser
        from src.core.learning.training_logic import XGBoostTrainer
    except ImportError as e:
        print(f"HIBA: Nem sikerült betölteni a belső modulokat: {e}")
        raise

    # --- WORKER OSZTÁLYOK ---

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
            self.log_signal.emit("📊 Jellemzők kinyerése (Parquet készítés folyamatban)...")
            try:
                extractor = FeatureExtractor(data_dir="processed_data")
                df = extractor.extract_features()

                if df is not None and not df.empty:
                    parquet_path = "training_data_pixelwise.parquet"
                    extractor.save_to_parquet(df, parquet_path)
                    msg = f"✅ Parquet mentve: {parquet_path} ({len(df)} sor)"
                    self.log_signal.emit(msg)
                    self.write_to_log_file(msg)
                else:
                    self.log_signal.emit("⚠️ Hiba: Nem generálódott adat.")
            except Exception as e:
                self.log_signal.emit(f"❌ Hiba: {str(e)}")

            self.finished.emit()

    class TrainingWorker(QThread):
        log_signal = pyqtSignal(str)
        finished_signal = pyqtSignal(bool)

        def __init__(self, trainer_class, do_split, *args, **kwargs):
            super().__init__()
            self.do_split = do_split
            self.trainer = trainer_class(*args, **kwargs) if trainer_class else None

        def run(self):
            if not self.trainer:
                self.log_signal.emit("❌ Hiba: XGBoostTrainer nem található!")
                self.finished_signal.emit(False)
                return

            success = self.trainer.train(self.emit_log, do_split=self.do_split)
            self.finished_signal.emit(success)

        def emit_log(self, message):
            self.log_signal.emit(message)

    # --- UI INTERFÉSZEK ---

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
            self.config = {'model-name': 'lung_dx_model.pkl'}
            self.resource_folder = "resources"

            if not os.path.exists(self.resource_folder):
                os.makedirs(self.resource_folder)

            dask.config.set({
                "distributed.comm.timeouts.connect": "60s",
                "distributed.comm.timeouts.tcp": "60s",
                "distributed.worker.memory.target": 0.6,
                "distributed.worker.memory.spill": 0.7,
                "distributed.worker.memory.pause": 0.8,
                "distributed.worker.memory.terminate": 0.95
            })

            try:
                self.dask_client = Client(n_workers=1, threads_per_worker=2, processes=True, memory_limit='4GB')
            except Exception as e:
                self.dask_client = None

            self._init_ui()

        def _init_ui(self):
            self.top_card = CardWidget(self)
            h_ly = QHBoxLayout(self.top_card)

            self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM")
            self.xml_btn = PushButton(FluentIcon.FOLDER, "XML")
            self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "1. Indexelés")
            self.process_btn = PrimaryPushButton(FluentIcon.SYNC, "2. Feldolgozás")
            self.export_btn = PrimaryPushButton(FluentIcon.SAVE, "3. Parquet fájl készítés")

            self.train_mode_layout = QVBoxLayout()
            self.test_mode_switch = SwitchButton()
            self.test_mode_switch.setOnText("Teszt (80/20)")
            self.test_mode_switch.setOffText("Végleges (100%)")
            self.test_mode_switch.setChecked(True)

            self.train_mode_layout.addWidget(BodyLabel("Tanítási mód:"))
            self.train_mode_layout.addWidget(self.test_mode_switch)
            self.train_btn = PrimaryPushButton(FluentIcon.ROBOT, "4. Model tanítás")

            for btn in [self.run_btn, self.process_btn, self.export_btn, self.train_btn]:
                btn.setEnabled(False)

            h_ly.addWidget(self.dicom_btn)
            h_ly.addWidget(self.xml_btn)
            h_ly.addStretch(1)
            h_ly.addWidget(self.run_btn)
            h_ly.addWidget(self.process_btn)
            h_ly.addWidget(self.export_btn)
            h_ly.addSpacing(20)
            h_ly.addLayout(self.train_mode_layout)
            h_ly.addWidget(self.train_btn)

            self.layout.addWidget(self.top_card)
            self.progress_bar = ProgressBar(self)
            self.layout.addWidget(self.progress_bar)

            self.log_display = QTextEdit()
            self.log_display.setReadOnly(True)
            self.log_display.setStyleSheet(
                "background-color: #1a1a1a; color: #00ff00; font-family: Consolas; font-size: 12px;")
            self.layout.addWidget(SubtitleLabel("Rendszernapló"))
            self.layout.addWidget(self.log_display)

            self.dicom_btn.clicked.connect(self.select_dicom)
            self.xml_btn.clicked.connect(self.select_xml)
            self.run_btn.clicked.connect(self.start_index)
            self.process_btn.clicked.connect(self.start_processing)
            self.export_btn.clicked.connect(self.start_export)
            self.train_btn.clicked.connect(self.start_training_process)

        def select_dicom(self):
            p = QFileDialog.getExistingDirectory(self, "DICOM Mappa")
            if p:
                self.dicom_dir = p
                self.log_display.append(f"📂 DICOM: {p}")
                self.check_ready()

        def select_xml(self):
            p = QFileDialog.getExistingDirectory(self, "XML Mappa")
            if p:
                self.xml_dir = p
                self.log_display.append(f"📝 XML: {p}")
                self.check_ready()

        def check_ready(self):
            if self.dicom_dir and self.xml_dir:
                self.mgr = DataManager(self.dicom_dir, self.xml_dir)
                self.mgr.index_files()
                if len(self.mgr.valid_pairs) > 0:
                    self.run_btn.setEnabled(True)
                    self.log_display.append(f"✅ Talált párok: {len(self.mgr.valid_pairs)}. Mehet az indexelés!")

        def start_index(self):
            self.run_btn.setEnabled(False)
            self.log_display.append("\n--- 1. INDEXELÉS ---")
            self.worker = BatchWorker(self.mgr.valid_pairs)
            self.worker.log_signal.connect(self.log_display.append)
            self.worker.progress_signal.connect(self.progress_bar.setValue)
            self.worker.data_ready_signal.connect(lambda data: setattr(self, 'patient_store', data))
            self.worker.finished.connect(self.on_index_finished)
            self.worker.start()

        def on_index_finished(self):
            self.run_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            self.log_display.append("➡️ Kész! Mehet a feldolgozás.")

        def start_processing(self):
            self.process_btn.setEnabled(False)
            self.log_display.append("\n--- 2. FELDOLGOZÁS ---")
            self.processor = TumorProcessor(self.patient_store)
            self.processor.log_signal.connect(self.log_display.append)
            self.processor.progress_signal.connect(self.progress_bar.setValue)
            self.processor.finished.connect(self.on_processing_finished)
            self.processor.start()

        def on_processing_finished(self):
            self.process_btn.setEnabled(False)
            self.export_btn.setEnabled(True)
            self.log_display.append("➡️ Kész! Mehet a Parquet fájl készítés.")

        def start_export(self):
            self.export_btn.setEnabled(False)
            self.log_display.append("\n--- 3. Parquet fájl készítés ---")
            self.feat_worker = FeatureWorker()
            self.feat_worker.log_signal.connect(self.log_display.append)
            self.feat_worker.finished.connect(self.on_export_finished)
            self.feat_worker.start()

        def on_export_finished(self):
            self.train_btn.setEnabled(True)
            self.log_display.append("➡️ Kész! Mehet a Modell Tanítás.")

        def start_training_process(self):
            if not XGBoostTrainer:
                QMessageBox.critical(self, "Hiba", "Nincs meg a TrainingLogic modul!")
                return
            do_split = self.test_mode_switch.isChecked()
            self.train_btn.setEnabled(False)
            self.log_display.append("\n🧠 TANÍTÁS INDÍTÁSA...")
            self.train_worker = TrainingWorker(
                XGBoostTrainer, do_split=do_split, csv_file_path="training_data_pixelwise.parquet",
                resource_folder=self.resource_folder, config=self.config, client=self.dask_client
            )
            self.train_worker.log_signal.connect(self.log_display.append)
            self.train_worker.finished_signal.connect(self.on_training_finished)
            self.train_worker.start()

        def on_training_finished(self, success):
            self.run_btn.setEnabled(True)
            self.train_btn.setEnabled(False)
            if success:
                self.log_display.append("✅ Sikeres tanítás!")
                self.cleanup_temp_files()
            else:
                self.log_display.append("❌ Hiba a tanítás során.")

        def cleanup_temp_files(self):
            try:
                if os.path.exists("training_data_pixelwise.parquet"):
                    os.remove("training_data_pixelwise.parquet")
                if os.path.exists("processed_data"):
                    shutil.rmtree("processed_data")
                    os.makedirs("processed_data")
                self.log_display.append("🧹 Takarítás kész.")
            except Exception as e:
                self.log_display.append(f"❌ Takarítási hiba: {e}")

    class MainWindow(MSFluentWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LungDx Studio Pro")
            self.resize(1200, 800)
            self.dashboard = DashboardInterface(self)
            self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, 'Adatkezelés & Tanítás')
            setTheme(Theme.DARK)

        def closeEvent(self, event):
            if hasattr(self.dashboard, 'dask_client') and self.dashboard.dask_client:
                self.dashboard.dask_client.close()
            event.accept()

    # --- ALKALMAZÁS INDÍTÁSA ---
    app = QApplication(sys.argv)

    # Ha nincs konzol, átirányítjuk a kimenetet, de csak itt bent
    if sys.stdout is None:
        sys.stdout = NullWriter()
    if sys.stderr is None:
        sys.stderr = NullWriter()

    print("Inicilaizálás: MainWindow...")
    w = MainWindow()
    w.show()

    print("Alkalmazás fut...")
    return app.exec()


# --- A PROGRAM VALÓDI BELÉPÉSI PONTJA ---
if __name__ == '__main__':
    # 1. Kötelező freeze_support az XGBoost/Dask multiprocessing miatt
    from multiprocessing import freeze_support

    freeze_support()

    # 2. Védett indítás
    try:
        exit_code = run_application()
        sys.exit(exit_code)
    except Exception as e:
        # Ez a rész ment meg minket: ha bármi hiba van, itt kiírjuk
        print("\n" + "!" * 60)
        print("KRITIKUS HIBA AZ INDÍTÁSKOR:")
        print(f"Hiba: {e}")
        print("-" * 60)

        import traceback

        traceback.print_exc()

        print("!" * 60)
        # Megállítjuk az ablakot, hogy el tudd olvasni a hibaüzenetet
        input("\nA hiba elolvasása után nyomj ENTER-t a bezáráshoz...")