# src/gui/main_window.py
import sys
import os
import json
import traceback


class NullWriter:
    """Megakadályozza a 'NoneType' hibát GUI módban."""

    def write(self, arg): pass

    def flush(self): pass


# Globális változó deklarálása
XGBoostTrainer = None


# --- 1. I18N NYELVKEZELŐ ---
class I18nManager:
    """Nyelvi szótárat kezelő osztály JSON alapon, beépített fallback támogatással."""

    DEFAULT_TRANSLATIONS = {
        "hu": {
            "window_title": "LungDx Studio Pro",
            "nav_data_prep": "Adatkezelés & Tanítás",
            "btn_dicom": "DICOM",
            "btn_xml": "XML",
            "btn_license": "Licenc (.json)",
            "btn_indexing": "1. Indexelés",
            "btn_processing": "2. Feldolgozás",
            "btn_parquet": "3. Parquet fájl készítés",
            "btn_training": "4. Model tanítás",
            "train_mode_label": "Tanítási mód:",
            "switch_test": "Teszt (80/20)",
            "switch_final": "Végleges (100%)",
            "log_title": "Rendszernapló",
            "msg_missing_license_title": "Hiányzó licenc",
            "msg_missing_license_body": "A tanítás indítása előtt kérjük tallózza be a licenc (.json) fájlt!",
            "msg_error_title": "Hiba",
            "msg_no_trainer_module": "Nincs meg a TrainingLogic modul!",
            "msg_success_title": "Siker",
            "msg_success_body": "A modell tanítása sikeres! A köztes adatok törlésre kerültek.",
            "msg_train_error_body": "Hiba történt a tanítás során.",
            "dagshub_title": "DAGsHub Elérési Hiba",
            "dagshub_text": "A modell feltöltése sikertelen, mert a távoli DAGsHub szerver nem elérhető 3 próbálkozás után sem.",
            "btn_close_app": "Program bezárása",
            "btn_try_later": "Későbbi próbálkozás"
        },
        "en": {
            "window_title": "LungDx Studio Pro",
            "nav_data_prep": "Data Management & Training",
            "btn_dicom": "DICOM",
            "btn_xml": "XML",
            "btn_license": "License (.json)",
            "btn_indexing": "1. Indexing",
            "btn_processing": "2. Processing",
            "btn_parquet": "3. Create Parquet File",
            "btn_training": "4. Model Training",
            "train_mode_label": "Training Mode:",
            "switch_test": "Test (80/20)",
            "switch_final": "Final (100%)",
            "log_title": "System Log",
            "msg_missing_license_title": "Missing License",
            "msg_missing_license_body": "Please select a license (.json) file before starting training!",
            "msg_error_title": "Error",
            "msg_no_trainer_module": "TrainingLogic module not found!",
            "msg_success_title": "Success",
            "msg_success_body": "Model training successful! Intermediate data cleaned up.",
            "msg_train_error_body": "An error occurred during training.",
            "dagshub_title": "DAGsHub Access Error",
            "dagshub_text": "Model upload failed because the remote DAGsHub server is unreachable after 3 attempts.",
            "btn_close_app": "Close Application",
            "btn_try_later": "Try Later"
        }
    }

    def __init__(self, locales_dir="src/locales", default_lang="hu"):
        self.locales_dir = locales_dir
        self.current_lang = default_lang
        self.translations = {}
        self.listeners = []
        self.load_language(default_lang)

    def register_listener(self, callback):
        if callback not in self.listeners:
            self.listeners.append(callback)

    def load_language(self, lang_code: str) -> bool:
        filepath = os.path.join(self.locales_dir, f"{lang_code}.json")
        loaded_from_file = False

        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self.translations = json.load(f)
                loaded_from_file = True
            except Exception as e:
                print(f"⚠️ Hiba a nyelvi JSON fájl betöltésekor [{filepath}]: {e}")

        if not loaded_from_file:
            self.translations = self.DEFAULT_TRANSLATIONS.get(lang_code, self.DEFAULT_TRANSLATIONS.get("hu", {}))

        self.current_lang = lang_code
        self._notify_listeners()
        return True

    def _notify_listeners(self):
        for callback in self.listeners:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ Hiba a nyelvváltási értesítéskor: {e}")

    def t(self, key: str, default: str = "") -> str:
        return self.translations.get(key, default if default else key)


i18n = I18nManager()


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
                                    PrimaryPushButton, PushButton, CardWidget, FluentIcon, ProgressBar, SwitchButton, ComboBox)

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
                self.license_file_path = None  # Licenc fájl tárolója
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
                        memory_limit='4GB'
                    )
                except Exception as e:
                    self.write_to_log_file(f"Dask Error: {e}")
                    self.dask_client = None

                self._init_ui()
                i18n.register_listener(self.retranslate_ui)

            def write_to_log_file(self, message):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [GUI] {message}\n")

            def _init_ui(self):
                # Felső vezérlő panel kártyán
                self.top_card = CardWidget(self)
                h_ly = QHBoxLayout(self.top_card)

                # Forrás mappák és a licenc gomb
                self.dicom_btn = PushButton(FluentIcon.FOLDER, i18n.t("btn_dicom"))
                self.xml_btn = PushButton(FluentIcon.FOLDER, i18n.t("btn_xml"))
                self.license_btn = PushButton(FluentIcon.VPN, i18n.t("btn_license"))

                # Folyamat gombok (sorrendben)
                self.run_btn = PrimaryPushButton(FluentIcon.PLAY, i18n.t("btn_indexing"))
                self.process_btn = PrimaryPushButton(FluentIcon.SYNC, i18n.t("btn_processing"))
                self.export_btn = PrimaryPushButton(FluentIcon.SAVE, i18n.t("btn_parquet"))

                # Tanítás szekció (Kapcsoló + Gomb)
                self.train_mode_layout = QVBoxLayout()
                self.test_mode_switch = SwitchButton()
                self.test_mode_switch.setOnText(i18n.t("switch_test"))
                self.test_mode_switch.setOffText(i18n.t("switch_final"))
                self.test_mode_switch.setChecked(True)

                self.mode_label = BodyLabel(i18n.t("train_mode_label"))
                self.train_mode_layout.addWidget(self.mode_label)
                self.train_mode_layout.addWidget(self.test_mode_switch)

                self.train_btn = PrimaryPushButton(FluentIcon.ROBOT, i18n.t("btn_training"))

                # Gombok tiltása az elején
                for btn in [self.run_btn, self.process_btn, self.export_btn, self.train_btn]:
                    btn.setEnabled(False)

                # UI elemek elrendezése
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
                self.log_title_label = SubtitleLabel(i18n.t("log_title"))
                self.layout.addWidget(self.log_title_label)
                self.layout.addWidget(self.log_display)

                # Signal bekötések
                self.dicom_btn.clicked.connect(self.select_dicom)
                self.xml_btn.clicked.connect(self.select_xml)
                self.license_btn.clicked.connect(self.select_license)
                self.run_btn.clicked.connect(self.start_index)
                self.process_btn.clicked.connect(self.start_processing)
                self.export_btn.clicked.connect(self.start_export)
                self.train_btn.clicked.connect(self.start_training_process)

            def retranslate_ui(self):
                """UI szövegek frissítése nyelvváltáskor."""
                self.dicom_btn.setText(i18n.t("btn_dicom"))
                self.xml_btn.setText(i18n.t("btn_xml"))
                self.license_btn.setText(i18n.t("btn_license"))
                self.run_btn.setText(i18n.t("btn_indexing"))
                self.process_btn.setText(i18n.t("btn_processing"))
                self.export_btn.setText(i18n.t("btn_parquet"))
                self.train_btn.setText(i18n.t("btn_training"))
                self.mode_label.setText(i18n.t("train_mode_label"))
                self.test_mode_switch.setOnText(i18n.t("switch_test"))
                self.test_mode_switch.setOffText(i18n.t("switch_final"))
                self.log_title_label.setText(i18n.t("log_title"))

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
                """Fájlválasztó ablak a licenc JSON fájl betallózásához."""
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
                    QMessageBox.critical(self, i18n.t("msg_error_title"), i18n.t("msg_no_trainer_module"))
                    return

                if not self.license_file_path:
                    from pathlib import Path
                    default_path = Path.home() / ".pulmoflow" / "credentials.json"
                    if not default_path.exists():
                        QMessageBox.warning(
                            self, i18n.t("msg_missing_license_title"),
                            i18n.t("msg_missing_license_body")
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
                    QMessageBox.information(self, i18n.t("msg_success_title"),
                                            i18n.t("msg_success_body"))
                else:
                    self.log_display.append("\n⚠️ A tanítás nem volt sikeres, a fájlok megmaradtak a hibakereséshez.")
                    QMessageBox.warning(self, i18n.t("msg_error_title"), i18n.t("msg_train_error_body"))

            def on_dagshub_error(self, error_msg):
                self.log_display.append(f"\n❌ Kapcsolódási hiba: {error_msg}")
                self.progress_bar.setValue(0)

                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle(i18n.t("dagshub_title"))
                msg.setText(i18n.t("dagshub_text"))
                msg.setInformativeText(f"Részletek: {error_msg}\n\nKérjük válasszon a következő műveletek közül:")

                close_btn = msg.addButton(i18n.t("btn_close_app"), QMessageBox.ButtonRole.DestructiveRole)
                retry_later_btn = msg.addButton(i18n.t("btn_try_later"), QMessageBox.ButtonRole.AcceptRole)

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
                self.setWindowTitle(i18n.t("window_title"))
                self.resize(1200, 800)

                self.dashboard = DashboardInterface(self)
                self.addSubInterface(self.dashboard, FluentIcon.ACCEPT, i18n.t("nav_data_prep"))

                # Nyelvválasztó ComboBox hozzáadása a címsorhoz
                self.lang_combo = ComboBox(self)
                self.lang_combo.addItems(["Magyar", "English"])
                self.lang_combo.setCurrentIndex(0 if i18n.current_lang == "hu" else 1)
                self.lang_combo.currentIndexChanged.connect(self.on_language_changed)

                if hasattr(self, 'titleBar') and self.titleBar:
                    self.titleBar.layout().insertWidget(0, self.lang_combo)

                i18n.register_listener(self.retranslate_ui)
                setTheme(Theme.DARK)

            def on_language_changed(self, index: int):
                lang_code = "hu" if index == 0 else "en"
                i18n.load_language(lang_code)

            def retranslate_ui(self):
                self.setWindowTitle(i18n.t("window_title"))
                nav_item = self.navigationInterface.widget(self.dashboard.objectName())
                if nav_item:
                    nav_item.setText(i18n.t("nav_data_prep"))

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