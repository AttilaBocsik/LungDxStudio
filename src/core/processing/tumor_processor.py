# src/core/processing/tumor_processor.py
import numpy as np
import cv2
import os
import gc
import shutil  # Új import a törléshez
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

# Importok a saját moduljaidból
from src.core.segmentation.lung_segmenter import LungSegmenter
import src.utils.project_utils as project_utils
from src.core.lsmc import LSMC


class TumorProcessor(QThread):
    """
    Háttérszál (QThread) a daganatos CT szeletek kötegelt feldolgozására.
    Optimalizált memória- és CPU-kezeléssel az orvosi képekhez.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, patient_store, output_dir="processed_data"):
        super().__init__()
        self.patient_store = patient_store
        self.output_dir = output_dir
        self.lsmc = LSMC()
        self.target_labels = ['A', 'B', 'G', 'D']

        # --- Mappa ürítése/létrehozása inicializáláskor ---
        self._prepare_output_directory()

    def _prepare_output_directory(self):
        """
        Létrehozza a kimeneti mappát, vagy ha létezik, törli annak tartalmát.
        Mivel a run() metódus előtt fut, a log_signal-t itt még nem tudjuk megbízhatóan használni,
        ezért a konzolra és később a run elején a log-ba is jelezzük.
        """
        try:
            if os.path.exists(self.output_dir):
                # Töröljük a tartalmát (fájlok és almappák)
                for filename in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"🗑️ Cleaned: {self.output_dir}")
            else:
                os.makedirs(self.output_dir)
                print(f"📁 Created: {self.output_dir}")
        except Exception as e:
            print(f"❌ Error during folder cleanup: {e}")

    # ... (prepare_data_for_roi2rect változatlan) ...
    def prepare_data_for_roi2rect(self, annotations):
        """Átalakítja az annotációkat One-Hot kódolt listává."""
        img_data_list = []
        if not annotations:
            return None

        for ann in annotations:
            xmin, ymin, xmax, ymax = ann['bbox']
            label = ann['label']
            one_hot = [0] * len(self.target_labels)
            if label in self.target_labels:
                idx = self.target_labels.index(label)
                one_hot[idx] = 1

            row = [xmin, ymin, xmax, ymax] + one_hot
            img_data_list.append(row)
        return img_data_list

    def run(self):
        """
        Optimalizált feldolgozási folyamat.
        """
        # Jelzés a rendszer naplónak az ürítésről
        self.log_signal.emit(f"🧹 Kimeneti könyvtár ({self.output_dir}) kiürítve.")

        # Feladatok kigyűjtése
        tasks = []
        for p_id, slices in self.patient_store.items():
            for s in slices:
                if s.get('has_tumor', False):
                    tasks.append(s)

        total = len(tasks)
        self.log_signal.emit(f"⚙️ Feldolgozás indítása: {total} daganatos szelet (Optimalizált mód)...")

        for i, slice_data in enumerate(tasks):
            img_name = slice_data['img_name']
            p_id = slice_data['patient_id']

            try:
                # 1) Adat beolvasás
                _, _, origin_img, _, _, _, _ = LungSegmenter.load_file(slice_data['path'])
                origin_img = origin_img.astype('float32')
                # 2) ROI + Maszk generálás
                img_data_formatted = self.prepare_data_for_roi2rect(slice_data['annotations'])
                tumor_mask_ndarray, roi_pos, tumor_label = project_utils.roi2rect(
                    img_name=img_name,
                    img_np=origin_img,
                    img_data=img_data_formatted,
                    label_list=self.target_labels,
                    image=origin_img
                )

                if roi_pos is None or tumor_mask_ndarray is None:
                    self.log_signal.emit(f"⚠️ SKIPPED ({img_name}): Nincs érvényes ROI.")
                    continue

                # 3) Maszk normalizálása
                if len(tumor_mask_ndarray.shape) == 3:
                    tumor_mask_gray = cv2.cvtColor(tumor_mask_ndarray, cv2.COLOR_BGR2GRAY)
                else:
                    tumor_mask_gray = tumor_mask_ndarray

                # 4) GVF Snake
                _, snake_points, roi_points = project_utils.gvf_snake(tumor_mask_gray, roi_pos)

                # 5) Poligon maszkok
                final_tumor_mask = np.zeros(tumor_mask_gray.shape, dtype='uint8')
                cv2.fillPoly(final_tumor_mask, pts=[snake_points], color=255)
                masked_tumor = np.where(final_tumor_mask > 0, origin_img, 0).astype('float32')

                roi_mask = np.zeros(tumor_mask_gray.shape, dtype='uint8')
                cv2.fillPoly(roi_mask, pts=[roi_points], color=255)
                inverse_roi_mask = cv2.subtract(roi_mask, final_tumor_mask)
                inverted_masked_roi = np.where(inverse_roi_mask > 0, origin_img, 0).astype('float32')

                # 6) Parenchyma
                mask_list_400 = self.lsmc.make_lungmask([slice_data['path']], -400)
                segmented_parenchyma = (mask_list_400[0] * origin_img).astype(
                    'float32') if mask_list_400 else np.zeros_like(origin_img)

                # 7) Mentés
                save_path = os.path.join(self.output_dir, f"{p_id}_{img_name}.npz")
                np.savez_compressed(
                    save_path,
                    original=origin_img,
                    parenchyma=segmented_parenchyma,
                    masked_tumor=masked_tumor,
                    inverted_roi=inverted_masked_roi,
                    label=tumor_label,
                    patient_id=p_id
                )
                self.log_signal.emit(f"✅ Mentve: {p_id} -> {img_name}")
            except Exception as e:
                self.log_signal.emit(f"❌ Kihagyva {p_id} -> ({img_name}): {str(e)}")
                continue

            finally:
                # 8) Memória felszabadítás
                vars_to_del = ['origin_img', 'tumor_mask_ndarray', 'tumor_mask_gray', 'final_tumor_mask',
                               'masked_tumor', 'roi_mask', 'inverted_masked_roi', 'segmented_parenchyma']
                for v in vars_to_del:
                    if v in locals(): del locals()[v]
                if i % 5 == 0: gc.collect()

            self.progress_signal.emit(int(((i + 1) / total) * 100))

        self.log_signal.emit("🏁 Feldolgozás befejezve. A RAM felszabadítva.")
        self.finished.emit()