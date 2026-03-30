# src/core/segmentation/lung_segmenter.py
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import measure, morphology
from scipy import ndimage


class LungSegmenter:
    """
    CT felvételeken végzett tüdőszegmentálásért felelős osztály.
    Küszöbölést és morfológiai műveleteket használ a tüdőállomány elkülönítéséhez.
    """

    def __init__(self, threshold_hu=-320):
        """
        Args:
            threshold_hu (int): A szegmentáláshoz használt Hounsfield-egység (HU) küszöbérték.
                                Az alapértelmezett -320 körüli érték alkalmas a levegőtartalmú tüdőhöz.
        """
        self.threshold_hu = threshold_hu

    @staticmethod
    def load_file(path):
        """
        Beolvas egy DICOM fájlt, kezeli a hiányzó metaadatokat és az RGB konverziót.
        """
        ds = pydicom.dcmread(str(path))

        # Biztonsági mentés: Ha valahol később a SliceLocation-t keresné a kód,
        # de nincs a fájlban, adjunk neki egy alapértelmezett értéket, hogy ne szálljon el.
        if not hasattr(ds, 'SliceLocation'):
            # Megpróbáljuk az ImagePositionPatient-ből kiszedni a Z-t, ha az sincs, marad a 0.0
            ds.SliceLocation = ds.ImagePositionPatient[2] if 'ImagePositionPatient' in ds else 0.0

        img_array = ds.pixel_array.astype(np.float32)

        # 1. RGB kezelése
        if img_array.ndim == 3 and img_array.shape[-1] == 3:
            img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

        # 2. Dimenziók tisztítása
        img_array = np.squeeze(img_array)

        # 3. Spacing adatok biztonságos kezelése
        spacing_x, spacing_y = (1.0, 1.0)
        if "PixelSpacing" in ds:
            spacing_x, spacing_y = map(float, ds.PixelSpacing)

        # Több lépcsős fallback a Z spacingre
        spacing_z = float(getattr(ds, "SliceThickness",
                                  getattr(ds, "SpacingBetweenSlices", 1.0)))
        if spacing_z <= 0: spacing_z = 1.0

        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing((spacing_x, spacing_y, spacing_z))

        # Biztonságos alak kinyerése
        if img_array.ndim == 2:
            height, width = img_array.shape
        else:
            # Ha még mindig több dimenziós lenne, vegyük az utolsó kettőt
            height, width = img_array.shape[-2], img_array.shape[-1]

        frame_num = 1

        return ds, img_sitk, img_array, frame_num, width, height, 1
    '''
    def load_file(path):
        """
        Beolvas egy DICOM fájlt, kezeli a fizikai méretezést (spacing) és
        SimpleITK objektumot, valamint numpy tömböt hoz létre belőle.

        Args:
            path (str/Path): A DICOM fájl elérési útja.

        Returns:
            tuple: (pydicom_ds, sitk_image, img_array, slices, width, height, channel)
        """
        ds = pydicom.dcmread(str(path))
        img_array = ds.pixel_array.astype(np.float32)

        # Spacing (térköz) adatok kinyerése és javítása a fizikai pontosság érdekében
        spacing_x, spacing_y = 1.0, 1.0
        spacing_z = 1.0

        if "PixelSpacing" in ds:
            spacing_x, spacing_y = map(float, ds.PixelSpacing)

        spacing_z = float(getattr(ds, "SliceThickness", getattr(ds, "SpacingBetweenSlices", 1.0)))
        if spacing_z == 0: spacing_z = 1.0

        spacing = (spacing_x, spacing_y, spacing_z)
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing(spacing)

        frame_num, width, height = (img_array.shape if len(img_array.shape) == 3
                                    else (1, img_array.shape[0], img_array.shape[1]))

        return ds, img_sitk, img_array, frame_num, width, height, 1
    '''

    def segment_mask(self, img_array):
        """
        Létrehozza a tüdő bináris maszkját a bemeneti képtömb alapján.

        A folyamat lépései:
        1. Küszöbölés a megadott HU érték alapján.
        2. Morfológiai zárás (closing) a kisebb rések eltüntetéséhez.
        3. Régiók címkézése és a legnagyobb összefüggő terület (tüdő) kiválasztása.
        4. Lyukkitöltés (binary fill holes) a tüdőn belüli erek/daganatok befoglalásához.

        Args:
            img_array (numpy.ndarray): A CT szelet pixeladatai.

        Returns:
            numpy.ndarray: Bináris maszk (0 és 1 értékekkel).
        """
        # Küszöbölés
        binary_image = np.array(img_array < self.threshold_hu, dtype=np.int8)
        # Morfológiai műveletek
        footprint = morphology.disk(2)
        binary_image = morphology.closing(binary_image, footprint)
        # Legnagyobb összefüggő terület keresése
        label_image = measure.label(binary_image)
        regions = measure.regionprops(label_image)
        if not regions: return np.zeros_like(img_array, dtype=np.uint8)
        regions.sort(key=lambda x: x.area, reverse=True)
        mask = (label_image > 0).astype(np.uint8)
        # Lyukak kitöltése a maszkon belül
        return ndimage.binary_fill_holes(mask).astype(np.uint8)
