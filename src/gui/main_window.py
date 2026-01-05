# src/gui/main_window.py
import sys
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QVBoxLayout, QFileDialog
from qfluentwidgets import (MSFluentWindow, SubtitleLabel, setTheme, Theme, ListWidget,
                            PrimaryPushButton, CardWidget, CaptionLabel, ImageLabel,
                            PushButton, FluentIcon)

from src.core.data_manager import DataManager
from src.utils.logger import setup_logger

log = setup_logger("GUI")


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.init_window()

        # Kezdetben nincs adatkezelő, amíg nincs mappa választva
        self.data_manager = None

        self.dashboard_interface = DashboardInterface(self)
        self.addSubInterface(self.dashboard_interface, FluentIcon.HOME, 'Elemzés')

        setTheme(Theme.DARK)

    def init_window(self):
        self.resize(1100, 750)
        self.setWindowTitle('Lung Cancer Detection System - AI Enterprise')


class DashboardInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Dashboard")

        self.dicom_path = None
        self.xml_path = None

        # Fő elrendezés
        self.h_layout = QHBoxLayout(self)
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # --- FELSŐ RÉSZ: Tallózó gombok ---
        self.setup_browser_section()

        # --- KÖZÉPSŐ RÉSZ: Lista ---
        self.list_label = SubtitleLabel("Elérhető felvételek", self)
        self.list_widget = ListWidget(self)
        self.list_widget.itemSelectionChanged.connect(self.on_item_selected)

        self.left_layout.addWidget(self.list_label)
        self.left_layout.addWidget(self.list_widget)

        # --- JOBB OLDAL: Megjelenítő ---
        self.setup_display_section()

        self.h_layout.addLayout(self.left_layout, 1)
        self.h_layout.addLayout(self.right_layout, 3)

    def setup_browser_section(self):
        """Létrehozza a mappa tallózó részt."""
        self.browser_card = CardWidget(self)
        browser_layout = QVBoxLayout(self.browser_card)

        # DICOM gomb
        self.dicom_btn = PushButton(FluentIcon.FOLDER, "DICOM mappa kiválasztása", self)
        self.dicom_btn.clicked.connect(self.select_dicom_folder)

        # XML gomb
        self.xml_btn = PushButton(FluentIcon.FOLDER, "XML mappa kiválasztása", self)
        self.xml_btn.clicked.connect(self.select_xml_folder)

        self.path_status_label = CaptionLabel("Nincs mappa kiválasztva")

        browser_layout.addWidget(self.dicom_btn)
        browser_layout.addWidget(self.xml_btn)
        browser_layout.addWidget(self.path_status_label)

        self.left_layout.addWidget(self.browser_card)

    def setup_display_section(self):
        self.image_card = CardWidget(self)
        card_layout = QVBoxLayout(self.image_card)

        self.image_display = ImageLabel(self)
        self.image_display.setFixedSize(512, 512)
        self.image_display.setBorderRadius(8, 8, 8, 8)

        self.info_label = CaptionLabel("Válassz mappákat az induláshoz...")
        self.process_btn = PrimaryPushButton("Analízis indítása")
        self.process_btn.setEnabled(False)

        card_layout.addWidget(self.image_display, 0, Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.info_label)
        card_layout.addWidget(self.process_btn)

        self.right_layout.addWidget(self.image_card)

    def select_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM mappa kiválasztása")
        if folder:
            self.dicom_path = folder
            self.update_init_status()

    def select_xml_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "XML mappa kiválasztása")
        if folder:
            self.xml_path = folder
            self.update_init_status()

    def update_init_status(self):
        """Ellenőrzi, hogy mindkét mappa megvan-e, és ha igen, indexel."""
        if self.dicom_path and self.xml_path:
            self.path_status_label.setText(f"Mappák rendben. Indexelés...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            try:
                # Most példányosítjuk a DataManagert
                self.data_manager = DataManager(self.dicom_path, self.xml_path)
                self.data_manager.index_files()

                # Lista frissítése
                self.list_widget.clear()
                for pair in self.data_manager.valid_pairs:
                    self.list_widget.addItem(pair[0].name)

                self.path_status_label.setText(f"Talált párok: {len(self.data_manager.valid_pairs)}")
                self.info_label.setText("Adatok betöltve. Válassz egyet a listából!")
            except Exception as e:
                log.error(f"Hiba az indexeléskor: {e}")
                self.path_status_label.setText("Hiba történt az indexelés során!")

            QApplication.restoreOverrideCursor()
        elif self.dicom_path:
            self.path_status_label.setText("Válaszd ki az XML mappát is!")
        elif self.xml_path:
            self.path_status_label.setText("Válaszd ki a DICOM mappát is!")

    def on_item_selected(self):
        if not hasattr(self, 'data_manager') or self.list_widget.currentRow() < 0:
            return

        selected_idx = self.list_widget.currentRow()
        dicom_path, xml_path = self.data_manager.valid_pairs[selected_idx]

        self.info_label.setText(f"Kiválasztva: {dicom_path.name}\nAnnotáció: {xml_path.name}")
        self.process_btn.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())