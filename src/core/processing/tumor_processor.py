# src/core/processing/tumor_processor.py
import numpy as np
import cv2
import os
from PyQt6.QtCore import QThread, pyqtSignal

# Importok a saját moduljaidból
from src.core.segmentation.lung_segmenter import LungSegmenter
import src.utils.project_utils as project_utils
from src.core.lsmc import LSMC


class TumorProcessor(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, patient_store, output_dir="processed_data"):
        super().__init__()
        self.patient_store = patient_store
        self.output_dir = output_dir
        self.lsmc = LSMC()  # Példányosítjuk a tüdőmaszkolót

        # Ezek az osztályok, amiket keresünk (sorrend fontos a One-Hot Encodinghoz)
        self.target_labels = ['A', 'B', 'G', 'D']

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def prepare_data_for_roi2rect(self, annotations):
        """
        Átalakítja az XML dict formátumot a roi2rect által várt listás formátumra.
        Bemenet: [{'bbox': (x1, y1, x2, y2), 'label': 'A'}, ...]
        Kimenet: [[x1, y1, x2, y2, 1, 0, 0, 0], ...]
        """
        img_data_list = []
        if not annotations:
            return None

        for ann in annotations:
            xmin, ymin, xmax, ymax = ann['bbox']
            label = ann['label']

            # One-Hot Encoding generálása
            one_hot = [0] * len(self.target_labels)
            if label in self.target_labels:
                idx = self.target_labels.index(label)
                one_hot[idx] = 1
            else:
                # Ha ismeretlen címke, akkor mind 0 vagy alapértelmezett?
                # Most hagyjuk csupa 0-n, vagy kezeljük úgy mint az elsőt
                pass

                # Összefűzés: [koordináták + one_hot vektor]
            row = [xmin, ymin, xmax, ymax] + one_hot
            img_data_list.append(row)

        return img_data_list

    def run(self):
        # Csak a daganatos képeket gyűjtjük ki feldolgozásra
        tasks = []
        for p_id, slices in self.patient_store.items():
            for s in slices:
                if s['has_tumor']:
                    tasks.append(s)

        total = len(tasks)
        self.log_signal.emit(f"⚙️ Feldolgozás indítása: {total} daganatos szelet...")

        for i, slice_data in enumerate(tasks):
            img_name = slice_data['img_name']

            try:
                # --- 1) Pixel array beolvasás ---
                ds, _, origin_img, _, _, _, _ = LungSegmenter.load_file(slice_data['path'])

                # float32 konverzió (ahogy az eredeti kódban volt)
                # Megjegyzés: A roi2rect float32-vel vagy uint8-cal dolgozik?
                # A gvf_snake normalizál, de a roi2rect copy-t csinál.
                origin_img = origin_img.astype('float32')

                # Adatok konvertálása a roi2rect számára
                img_data_formatted = self.prepare_data_for_roi2rect(slice_data['annotations'])

                # --- 2) ROI + mask generálás (TE KÓDOD HÍVÁSA) ---
                tumor_mask_ndarray, roi_rectangle_position, tumor_mask_label = project_utils.roi2rect(
                    img_name=img_name,
                    img_np=origin_img,
                    img_data=img_data_formatted,  # A konvertált lista
                    label_list=self.target_labels,
                    image=origin_img
                )

                if roi_rectangle_position is None:
                    # Ha üres lett a ROI (pl hiba miatt), lépjünk tovább
                    continue

                # --- 3) Maskok átalakítása OpenCV használatával ---
                # Figyelem: A float32 képet CV2 néha nem szereti konverzióknál, ha nincs 0-1 vagy 0-255 között
                # Biztosítjuk a konverziót uint8-ra a vizuális műveletekhez ha kell,
                # de a te kódod a tumor_mask_ndarray-t használja, ami elvileg binary (0, 255).

                tumor_mask_img = cv2.cvtColor(tumor_mask_ndarray, cv2.COLOR_GRAY2BGR)  # Ha 1 csatornás
                tumor_mask_img_gray = cv2.cvtColor(tumor_mask_img, cv2.COLOR_BGR2GRAY)

                # --- 4) GVF Snake (TE KÓDOD HÍVÁSA) ---
                tumor_img, snake_points, roi_points = project_utils.gvf_snake(
                    tumor_mask_img_gray,  # Itt a szürkeárnyalatos maszkot várja
                    roi_rectangle_position
                )

                # --- 5) Poligon maskok ---
                tumor_mask = np.zeros_like(tumor_mask_img_gray)
                cv2.fillPoly(tumor_mask, pts=[snake_points], color=(255,))

                # Maszkolás az EREDETI képen
                masked_tumor = tumor_mask * origin_img

                roi_mask = np.zeros_like(tumor_mask_img_gray)
                cv2.fillPoly(roi_mask, pts=[roi_points], color=(255,))
                masked_roi = roi_mask * origin_img

                inverted_mask = np.ones_like(masked_roi) * 255
                cv2.fillPoly(inverted_mask, pts=[snake_points], color=(0,))
                inverted_masked_roi = cv2.bitwise_and(masked_roi, inverted_mask)

                # --- 6) Parenchyma mask (TE KÓDOD HÍVÁSA) ---
                # A make_lungmask listát vár és listát ad vissza
                mask_list_400 = self.lsmc.make_lungmask([slice_data['path']], -400)
                segmented_parenchyma = mask_list_400[0] * origin_img

                # --- 7) Eredmény mentése .npz fájlba ---
                save_path = os.path.join(self.output_dir, f"{slice_data['patient_id']}_{img_name}.npz")

                np.savez_compressed(save_path,
                                    original=origin_img,
                                    parenchyma=segmented_parenchyma,
                                    masked_tumor=masked_tumor,
                                    inverted_roi=inverted_masked_roi,
                                    label=tumor_mask_label,
                                    snake_points=snake_points,  # Elmentjük a kontúrt is
                                    patient_id=slice_data['patient_id'])

                self.log_signal.emit(f"✅ Feldolgozva és mentve: {img_name}")

            except Exception as e:
                self.log_signal.emit(f"❌ HIBA ({img_name}): {str(e)}")
                import traceback
                print(traceback.format_exc())  # Konzolba is, részletesen

            self.progress_signal.emit(int(((i + 1) / total) * 100))

        self.log_signal.emit("🏁 Minden kijelölt szelet feldolgozva.")
        self.finished.emit()