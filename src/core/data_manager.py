# src/core/data_manager.py
import pydicom
from pathlib import Path
from src.utils.logger import setup_logger
from src.core.annotation_handler import AnnotationHandler

log = setup_logger("DataManager")


class DataManager:
    """
    A DICOM képek és XML annotációk indexelését és párosítását kezelő osztály.
    UID (SOPInstanceUID) alapú keresést használ a fájlnévfüggetlen párosításhoz.
    """

    def __init__(self, dicom_dir, annotation_dir):
        self.dicom_dir = Path(dicom_dir)
        self.annotation_dir = Path(annotation_dir)
        self.annot_handler = AnnotationHandler()

        # Hash Map-ek a gyors kereséshez (O(1) komplexitás)
        # Kulcs: SOPInstanceUID (a kép egyedi azonosítója)
        # Érték: Teljes fájl elérési út
        self.dicom_map = {}
        self.xml_map = {}

        self.valid_pairs = []  # Lista a (dicom_path, xml_path) párokról

    def index_files(self):
        """
        Végigpásztázza a forrásmappákat, felépíti az indexeket és párosítja a fájlokat.
        """
        log.info("Indexelés indítása...")
        self._index_dicoms()
        self._index_xmls()
        self._match_pairs()
        log.info(f"Indexelés kész. Valid párok száma: {len(self.valid_pairs)}")

    def _index_dicoms(self):
        """
        A DICOM fájlok gyors indexelése. Csak a metaadatokat olvassa be (pixeladatok nélkül),
        hogy kinyerje a SOPInstanceUID-t a gyors párosításhoz.
        """
        files = list(self.dicom_dir.rglob("*.dcm"))
        log.info(f"DICOM fájlok keresése: {len(files)} db talált fájl.")

        for f in files:
            try:
                # 1. Próbáljuk a modern, gyors módszerrel
                try:
                    ds = pydicom.dcmread(f, stop_before_pixel_data=True)
                except TypeError:
                    # 2. Ha régi a pydicom, olvassuk be hagyományosan (ez lassabb lesz!)
                    ds = pydicom.dcmread(f)

                # SOPInstanceUID: Ez a globálisan egyedi azonosítója a szeletnek
                uid = str(ds.SOPInstanceUID)
                self.dicom_map[uid] = f
            except Exception as e:
                log.warning(f"Hibás DICOM fájl kihagyva: {f.name} ({e})")

    def _index_xmls(self):
        """XML fájlok indexelése."""
        files = list(self.annotation_dir.rglob("*.xml"))
        log.info(f"XML fájlok keresése: {len(files)} db talált fájl.")

        for f in files:
            # Feltételezés: A fájlnév maga az UID (a logjaid alapján ez igaz)
            # pl. 1.3.6.1.4....xml -> UID: 1.3.6.1.4...
            uid = f.stem
            self.xml_map[uid] = f

    def _match_pairs(self):
        """
        Összeveti a DICOM és XML indexeket, és létrehozza a valid párok listáját,
        ahol mindkét fájl elérhető.
        """
        self.valid_pairs = []

        for uid, dicom_path in self.dicom_map.items():
            if uid in self.xml_map:
                xml_path = self.xml_map[uid]

                # Opcionális: Itt ellenőrizhetjük, hogy az XML tényleg tartalmaz-e bboxot
                # De teljesítmény okokból ezt jobb a tanítási ciklusra hagyni,
                # vagy külön validációs lépésben csinálni.
                self.valid_pairs.append((dicom_path, xml_path))
            else:
                # Olyan DICOM, aminek nincs annotációja (ez gyakori, nem hiba)
                pass

    def get_data_generator(self):
        """
        Python generátor, amely egyesével tölti be a memóriába a képeket és
        annotációkat a tanításhoz vagy feldolgozáshoz.

        Memóriatakarékos megoldás: csak az aktuálisan kért adatpárt tartja a memóriában.

        Yields:
            dict: Egy szótár, ami tartalmazza az UID-t, a képtömböt, a boxokat és az osztályokat.
        """
        for dicom_path, xml_path in self.valid_pairs:
            try:
                # Itt már betöltjük a teljes képet
                ds = pydicom.dcmread(dicom_path)
                image_data = ds.pixel_array

                # És a maszkokat
                bboxes, classes = self.annot_handler.parse_xml(xml_path)

                if bboxes is not None:
                    yield {
                        "uid": str(ds.SOPInstanceUID),
                        "image": image_data,
                        "boxes": bboxes,
                        "classes": classes,
                        "dicom_path": dicom_path
                    }
            except Exception as e:
                log.error(f"Hiba a pár feldolgozása közben: {dicom_path.name} -> {e}")
