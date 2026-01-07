# src/core/segmentation/lung_segmenter.py
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import measure, morphology
from scipy import ndimage


class LungSegmenter:
    def __init__(self, threshold_hu=-320):
        self.threshold_hu = threshold_hu

    @staticmethod
    def load_file(path):
        """
        Biztonságos DICOM betöltés pixeladatokkal együtt.
        Ezt akkor hívd meg, amikor ténylegesen számolni akarsz a képpel!
        """
        ds = pydicom.dcmread(str(path))
        img_array = ds.pixel_array.astype(np.float32)

        # Spacing javítása
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

    def segment_mask(self, img_array):
        """A korábbi javított szegmentáló logika (skimage.morphology.closing)."""
        binary_image = np.array(img_array < self.threshold_hu, dtype=np.int8)
        footprint = morphology.disk(2)
        binary_image = morphology.closing(binary_image, footprint)

        label_image = measure.label(binary_image)
        regions = measure.regionprops(label_image)
        if not regions: return np.zeros_like(img_array, dtype=np.uint8)

        regions.sort(key=lambda x: x.area, reverse=True)
        mask = (label_image > 0).astype(np.uint8)
        return ndimage.binary_fill_holes(mask).astype(np.uint8)