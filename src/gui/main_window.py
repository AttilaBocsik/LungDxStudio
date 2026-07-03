# src/gui/main_window.py
import sys
import os
import traceback


class NullWriter:
    """Megakadályozza a 'NoneType' hibát GUI módban."""

    def write(self, arg): pass

    def flush(self): pass


# Globális változó deklarálása
XGBoostTrainer = None


def run_application():
    """Ebben a függvényben történik minden import és inicializálás."""
    global XGBoostTrainer

    try:
        # 1. NEHÉZ IMPORTÁLÁSOK
        import time
        import pydicom
        import shutil
        import dask
        import pyarrow
        from dask.distributed import Client
        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox
        from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme, BodyLabel,
                                    PrimaryPushButton, PushButton, CardWidget, FluentIcon, ProgressBar, SwitchButton)

        # 2. SAJÁT MODULOK IMPORTÁLÁSA
        from src.core.data_manager import DataManager
        from src.core.processing.tumor_processor import TumorProcessor
        from src.core.learning.feature_extractor import FeatureExtractor
        from src.core.data_prep.annotation_parser import AnnotationParser

        try:
            from src.core.learning.training_logic import XGBoostTrainer as TrainerClass, DagsHubConnectionError
            XGBoostTrainer = TrainerClass
            print("✅ XGBoostTrainer sikeresen betöltve.")
        except ImportError as e:
            print(f"⚠️ Hiba a TrainingLogic betöltésekor: {e}")
            XGBoostTrainer = None

            # Tartalék osztály, ha nem sikerülne importálni
            class DagsHubConnectionError(Exception):
                pass

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

        # --- 2. Worker a Feature Extraction-höz (Export) ---
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

        # --- 3. Worker a Modell Tanításhoz ---
        class TrainingWorker(QThread):
            log_signal = pyqtSignal(str)
            finished_signal = pyqtSignal(bool)
            dagshub_error_signal = pyqtSignal(str)  # Szignál a hálózati kapcsolat hibájának átadásához

            def __init__(self, trainer_class, do_split, *args, **kwargs):
                super().__init__()
                self.do_split = do_split
                self.trainer = trainer_class(*args, **kwargs) if trainer_class else None

            def run(self):
                if not self.trainer:
                    self.log_signal.emit("❌ Hiba: XGBoostTrainer nem található!")
                    self.finished_signal.emit(False)
                    return

                try:
                    # A tanítás indítása a választott móddal
                    success = self.trainer.train(self.log_signal.emit, do_split=self.do_split)
                    self.finished_signal.emit(success)
                except DagsHubConnectionError as de:
                    # Elkapjuk a specifikus DAGsHub hibát, és jelezzük a főablaknak
                    self.dagshub_error_signal.emit(str(de))
                except Exception as e:
                    self.log_signal.emit(f"❌ Váratlan hiba a háttérszálon: {str(e)}")
                    self.finished_signal.emit(False)

            def emit_log(self, message):
                self.log_signal.emit(message)

        # --- DASHBOARD INTERFACE ---

        class DashboardInterface(QFrame):
            def __init__(self, parent=None):
                super().__init__(parent=parent)
                self.setObjectName("dashboard_interface")
                self.layout = QVBoxLayout(self)

                # Adatok tárolása
                self.dicom_dir = None
                self.xml_dir = None
                self.license_file_path = None  # ÚJ: Licenc fájl tárolója
                self.mgr = None
                self.patient_store = None
                self.log_file = "app.log"

                # Modell konfig
                self.config = {'model-name': 'lung_dx_model.pkl'}
                self.resource_folder = "resources"
                if not os.path.exists(self.resource_folder):
                    os.makedirs(self.resource_folder)

                # Dask konfiguráció finomhangolása
                dask.config.set({
                    "distributed.comm.timeouts.connect": "60s",
                    "distributed.comm.timeouts.tcp": "60s",
                    "distributed.worker.memory.target": 0.6,
                    "distributed.worker.memory.spill": 0.7,
                    "distributed.worker.memory.pause": 0.8,
                    "distributed.worker.memory.terminate": 0.95
                })

                try:
                    self.dask_client = Client(
                        n_workers=1,
                        threads_per_worker=2,
                        processes=True,
                        memory_limit='8GB'
                    )
                except Exception as e:
                    self.write_to_log_file(f"Dask Error: {e}")
                    self.dask_client = None

                self._init_ui()

            def write_to_log_file(self, message):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [GUI] {message}\n")

            def _init_ui(self):
                # Felső vezérlő panel kártyán
                self.top_card = CardWidget(self)
                h_ly = QHBoxLayout(self.top_card)

                # Forrás mappák és a licenc gomb
                self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM")
                self.xml_btn = PushButton(FluentIcon.FOLDER, "XML")
                self.license_btn = PushButton(FluentIcon.VPN, "Licenc (.json)")  # ÚJ: Licenc kiválasztó gomb

                # Folyamat gombok (sorrendben)
                self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "1. Indexelés")
                self.process_btn = PrimaryPushButton(FluentIcon.SYNC, "2. Feldolgozás")
                self.export_btn = PrimaryPushButton(FluentIcon.SAVE, "3. Parquet fájl készítés")

                # Tanítás szekció (Kapcsoló + Gomb)
                self.train_mode_layout = QVBoxLayout()
                self.test_mode_switch = SwitchButton()
                self.test_mode_switch.setOnText("Teszt (80/20)")
                self.test_mode_switch.setOffText("Végleges (100%)")
                self.test_mode_switch.setChecked(True)

                self.train_mode_layout.addWidget(BodyLabel("Tanítási mód:"))
                self.train_mode_layout.addWidget(self.test_mode_switch)

                self.train_btn = PrimaryPushButton(FluentIcon.ROBOT, "4. Model tanítás")

                # Gombok tiltása az elején
                for btn in [self.run_btn, self.process_btn, self.export_btn, self.train_btn]:
                    btn.setEnabled(False)

                # UI elemek elrendezése (Új gomb beillesztve)
                h_ly.addWidget(self.dicom_btn)
                h_ly.addWidget(self.xml_btn)
                h_ly.addWidget(self.license_btn)
                h_ly.addStretch(1)
                h_ly.addWidget(self.run_btn)
                h_ly.addWidget(self.process_btn)
                h_ly.addWidget(self.export_btn)
                h_ly.addSpacing(20)
                h_ly.addLayout(self.train_mode_layout)
                h_ly.addWidget(self.train_btn)

                self.layout.addWidget(self.top_card)

                # Progress Bar
                self.progress_bar = ProgressBar(self)
                self.layout.addWidget(self.progress_bar)

                # Log ablak
                self.log_display = QTextEdit()
                self.log_display.setReadOnly(True)
                self.log_display.setStyleSheet(
                    "background-color: #1a1a1a; color: #00ff00; font-family: Consolas; font-size: 12px;")
                self.layout.addWidget(SubtitleLabel("Rendszernapló"))
                self.layout.addWidget(self.log_display)

                # Signal bekötések
                self.dicom_btn.clicked.connect(self.select_dicom)
                self.xml_btn.clicked.connect(self.select_xml)
                self.license_btn.clicked.connect(self.select_license)  # ÚJ: Signal bekötés
                self.run_btn.clicked.connect(self.start_index)
                self.process_btn.clicked.connect(self.start_processing)
                self.export_btn.clicked.connect(self.start_export)
                self.train_btn.clicked.connect(self.start_training_process)

            # --- LOGIKA ---

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

            def select_license(self):
                """ÚJ: Fájlválasztó ablak a licenc JSON fájl betallózásához."""
                p, _ = QFileDialog.getOpenFileName(self, "Licenc / Hitelesítési fájl megnyitása", "", "JSON fájlok (*.json)")
                if p:
                    self.license_file_path = p
                    self.log_display.append(f"🔑 Licenc betöltve: {p}")

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
                self.run_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                self.export_btn.setEnabled(True)
                self.log_display.append("➡️ Kész! Mehet a Parquet fájl készítés.")

            def start_export(self):
                self.run_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                self.export_btn.setEnabled(False)
                self.log_display.append("\n--- 3. Parquet fájl készítés ---")
                self.feat_worker = FeatureWorker()
                self.feat_worker.log_signal.connect(self.log_display.append)
                self.feat_worker.finished.connect(self.on_export_finished)
                self.feat_worker.start()

            def on_export_finished(self):
                self.run_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                self.export_btn.setEnabled(False)
                self.train_btn.setEnabled(True)
                self.log_display.append("➡️ Kész! Mehet a Modell Tanítás.")

            def start_training_process(self):
                if not XGBoostTrainer:
                    QMessageBox.critical(self, "Hiba", "Nincs meg a TrainingLogic modul!")
                    return

                # ÚJ: Ellenőrizzük, hogy be van-e tallózva a licenc fájl
                if not self.license_file_path:
                    # Megnézzük, hogy létezik-e az alapértelmezett fallback útvonal
                    from pathlib import Path
                    default_path = Path.home() / ".pulmoflow" / "credentials.json"
                    if not default_path.exists():
                        QMessageBox.warning(
                            self, "Hiányzó licenc",
                            "A tanítás indítása előtt kérjük tallózza be a licenc (.json) fájlt!"
                        )
                        return
                    else:
                        self.license_file_path = str(default_path)

                do_split = self.test_mode_switch.isChecked()

                self.train_btn.setEnabled(False)
                self.log_display.append("\n" + "=" * 40)
                self.log_display.append(
                    f"🧠 TANÍTÁS INDÍTÁSA | Mód: {'TESZTELÉS (80/20)' if do_split else 'VÉGLEGES (100%)'}")
                self.log_display.append("=" * 40)

                # MÓDOSÍTVA: Átadjuk a credentials_path paramétert a workernek
                self.train_worker = TrainingWorker(
                    XGBoostTrainer,
                    do_split=do_split,
                    csv_file_path="training_data_pixelwise.parquet",
                    resource_folder=self.resource_folder,
                    config=self.config,
                    client=self.dask_client,
                    credentials_path=self.license_file_path
                )
                self.train_worker.log_signal.connect(self.log_display.append)
                self.train_worker.finished_signal.connect(self.on_training_finished)
                self.train_worker.dagshub_error_signal.connect(self.on_dagshub_error)
                self.train_worker.start()

            def on_training_finished(self, success):
                self.run_btn.setEnabled(True)
                self.process_btn.setEnabled(False)
                self.export_btn.setEnabled(False)
                self.train_btn.setEnabled(False)
                self.progress_bar.setValue(0)

                if success:
                    self.log_display.append("\n🧹 Átmeneti fájlok takarítása...")
                    self.cleanup_temp_files()
                    QMessageBox.information(self, "Siker",
                                            "A modell tanítása sikeres! A köztes adatok törlésre kerültek.")
                else:
                    self.log_display.append("\n⚠️ A tanítás nem volt sikeres, a fájlok megmaradtak a hibakereséshez.")
                    QMessageBox.warning(self, "Hiba", "Hiba történt a tanítás során.")

            def on_dagshub_error(self, error_msg):
                self.log_display.append(f"\n❌ Kapcsolódási hiba: {error_msg}")
                self.progress_bar.setValue(0)

                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("DAGsHub Elérési Hiba")
                msg.setText(
                    "A modell feltöltése sikertelen, mert a távoli DAGsHub szerver nem elérhető 3 próbálkozás után sem.")
                msg.setInformativeText(f"Részletek: {error_msg}\n\nKérjük válasszon a következő műveletek közül:")

                close_btn = msg.addButton("Program bezárása", QMessageBox.ButtonRole.DestructiveRole)
                retry_later_btn = msg.addButton("Későbbi próbálkozás", QMessageBox.ButtonRole.AcceptRole)

                msg.exec()

                if msg.clickedButton() == close_btn:
                    self.log_display.append("🚪 Kilépés a felhasználó kérésére...")
                    QApplication.quit()
                else:
                    self.log_display.append("ℹ️ Későbbi próbálkozás kiválasztva. A tanítási adatok megmaradtak.")
                    self.run_btn.setEnabled(True)
                    self.process_btn.setEnabled(False)
                    self.export_btn.setEnabled(False)
                    self.train_btn.setEnabled(True)

            def cleanup_temp_files(self):
                csv_file = "training_data_pixelwise.parquet"
                try:
                    if os.path.exists(csv_file):
                        os.remove(csv_file)
                        self.log_display.append(f"✅ Törölve: {csv_file}")
                except Exception as e:
                    self.log_display.append(f"❌ Nem sikerült a Parquet fájl törlése: {e}")

                processed_dir = "processed_data"
                try:
                    if os.path.exists(processed_dir):
                        shutil.rmtree(processed_dir)
                        os.makedirs(processed_dir)
                        self.log_display.append(f"✅ Törölve: {processed_dir} tartalma")
                except Exception as e:
                    self.log_display.append(f"❌ Hiba a mappa takarításakor: {e}")

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

        app = QApplication(sys.argv)
        if sys.stdout is None: sys.stdout = NullWriter()
        if sys.stderr is None: sys.stderr = NullWriter()

        w = MainWindow()
        w.show()
        return app.exec()

    except Exception as e:
        traceback.print_exc()
        input(f"Hiba: {e}. Enter a kilépéshez...")
        return 1


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(run_application())