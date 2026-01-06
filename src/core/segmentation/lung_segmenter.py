# src/core/segmentation/lung_segmenter.py
import numpy as np
import cv2
import pydicom
import SimpleITK as sitk  # Ez új import!
from skimage import measure, morphology
from scipy import ndimage


class LungSegmenter:
    def __init__(self, threshold_hu=-320):
        self.threshold_hu = threshold_hu

    @staticmethod
    def load_file(path):
        """
        Teljesen biztonságos DICOM betöltés: pydicom + kézi spacing kezelés.
        Visszatérési értékek: ds, img (SimpleITK), img_array (numpy), frame_num, width, height, ch
        """
        # 1. DICOM beolvasás pydicom-mal
        path_str = str(path)  # Biztosítjuk, hogy string legyen (Pathlib objektum helyett)
        ds = pydicom.dcmread(path_str)

        # 2. Pixel adatokat numpy tömbbé alakítjuk
        img_array = ds.pixel_array.astype(np.float32)

        # 3. --- Spacing kinyerése és javítása ---
        spacing_x, spacing_y = 1.0, 1.0
        spacing_z = 1.0

        if "PixelSpacing" in ds:
            try:
                spacing_x, spacing_y = map(float, ds.PixelSpacing)
            except Exception:
                pass  # Hiba esetén marad az alapértelmezett 1.0

        if "SliceThickness" in ds and float(ds.SliceThickness) > 0:
            spacing_z = float(ds.SliceThickness)
        elif "SpacingBetweenSlices" in ds and float(ds.SpacingBetweenSlices) > 0:
            spacing_z = float(ds.SpacingBetweenSlices)

        # Kritikus javítás: Ha a Z spacing 0, kényszerítjük 1.0-ra
        if spacing_z == 0:
            spacing_z = 1.0

        spacing = (spacing_x, spacing_y, spacing_z)

        # 4. --- SimpleITK kép létrehozása ---
        try:
            img = sitk.GetImageFromArray(img_array)
            img.SetSpacing(spacing)
        except Exception:
            # Ha a SimpleITK valamiért elhasalna (ritka), létrehozunk egy üreset
            img = None

        # 5. --- Kép dimenziók kinyerése ---
        if len(img_array.shape) == 3:
            frame_num, width, height = img_array.shape
            ch = 1
        elif len(img_array.shape) == 2:
            frame_num, width, height, ch = 1, *img_array.shape, 1
        else:
            frame_num, width, height, ch = 1, 1, 1, 1

        return ds, img, img_array, frame_num, width, height, ch

    def segment_mask(self, img_array):
        """
        Bemenet: numpy array (HU értékekkel)
        Kimenet: bináris maszk
        """
        try:
            # 1. Küszöbölés
            binary_image = np.array(img_array < self.threshold_hu, dtype=np.int8)

            # 2. Zajszűrés (Closing) - JAVÍTVA
            footprint = morphology.disk(2)
            binary_image = morphology.closing(binary_image, footprint)

            # 3. Címkézés és legnagyobb összefüggő terület keresése
            label_image = measure.label(binary_image)
            regions = measure.regionprops(label_image)

            if not regions:
                return np.zeros_like(img_array, dtype=np.uint8)

            # Tüdő keresése (terület alapján)
            regions.sort(key=lambda x: x.area, reverse=True)

            # Maszk létrehozása
            mask = (label_image > 0).astype(np.uint8)

            # A szegélynél lévő lyukak kitöltése
            mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

            return mask

        except Exception as e:
            # Itt print helyett használhatsz loggert is, ha van importálva
            print(f"Segmenter Error: {e}")
            return np.zeros_like(img_array, dtype=np.uint8)