# src/core/lsmc.py
import numpy as np
import pydicom
from scipy import ndimage
from skimage import measure, segmentation


class LSMC:
    """
    Lung Segmentation and Mask Correction (LSMC) modul.
    Felelős a CT szeletek betöltéséért, HU konverzióért és a tüdőmaszk generálásáért.
    """

    def load_scans(self, paths):
        """
        Loads scans from a folder and into a list.
        Parameters: path (Folder path)
        Returns: slices (List of slices)
        """
        slices = []
        for path in paths:
            # force=True szükséges lehet, ha a fájl nem teljesen szabványos
            slices.append(pydicom.dcmread(str(path), force=True))

        # Sorbarendezés InstanceNumber alapján
        slices.sort(key=lambda x: int(x.InstanceNumber))

        try:
            if len(paths) == 1:
                # Figyelem: Egy szelet esetén a pozícióból tippelünk vastagságot
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2])
            else:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            # Fallback, ha nincs ImagePositionPatient
            if len(paths) == 1:
                slice_thickness = np.abs(slices[0].SliceLocation)
            else:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices

    def get_pixels_hu(self, scans):
        """
        Converts raw images to Hounsfield Units (HU).
        Parameters: scans (Raw images)
        Returns: image (NumPy array)
        """
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)

        # Since the scanning equipment is cylindrical in nature and image output is square,
        # we set the out-of-scan pixels to 0
        image[image == -2000] = 0

        # HU = m*P + b
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def generate_markers(self, image, hu):
        """
        Jelölőket generál egy adott képhez (Watershed algoritmushoz).
        Parameters: image
        Returns: Internal Marker, External Marker, Watershed Marker
        """
        # Creation of the internal Marker
        marker_internal = image < hu
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        # Csak a legnagyobb területeket hagyjuk meg (tüdőlebenyek)
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        # Creation of the External Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        # Creation of the Watershed Marker
        marker_watershed = np.zeros((512, 512), dtype=np.int32)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed

    def make_lungmask(self, slice_dcm_path_list, hu=-400):
        """
        Szeletfájl-maszk létrehozása. Ez a fő belépési pont.
        :param slice_dcm_path_list: DICOM fájlok elérési útvonalának listája
        :param hu: Hounsfield küszöbérték (alapértelmezett: -400)
        :return: Szelet fájlok maszk tömb
        """
        train_patient_scans = self.load_scans(slice_dcm_path_list)
        train_patient_images = self.get_pixels_hu(train_patient_scans)

        test_patient_internal_list = []

        # Iterálás a képeken (jelen esetben 1 db kép van a listában a feldolgozásnál)
        for imgi in range(len(train_patient_images[:])):
            test_patient_internal, test_patient_external, test_patient_watershed = self.generate_markers(
                train_patient_images[imgi], hu)
            test_patient_internal_list.append(test_patient_internal)

        return test_patient_internal_list