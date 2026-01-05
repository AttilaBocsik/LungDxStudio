# src/core/annotation_handler.py
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


class AnnotationHandler:
    """
    Felelőssége: Egyetlen XML fájl beolvasása és az annotációk kinyerése.
    Nem tud a fájlrendszerről, csak azt az egy fájlt dolgozza fel, amit kap.
    Ez váltja ki az xml_preprocessor-t
    A xml_preprocesssor.py-t átalakítás egy tiszta osztállyá (AnnotationHandler).
    Kivéve belőle a fájlba írás és a sys.exit()
    hívásokat (ezek "megölik" a programot hiba esetén, ami tilos egy GUI alkalmazásnál).
    Helyette kivételeket dob, amit a hívó kezel le.
    """

    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        # Osztályok leképezése
        self.label_map = {'A': 0, 'B': 1, 'D': 2, 'G': 3}

    def parse_xml(self, xml_path: Path):
        """
        Visszaadja a bounding boxokat és a class-okat.
        Return: (bounding_boxes, one_hot_classes) vagy (None, None) ha üres/hibás.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Hibás XML formátum: {xml_path} -> {e}")

        bounding_boxes = []
        one_hot_classes = []

        # XML struktúra bejárása (LIDC-IDRI formátumhoz igazítva)
        for object_tag in root.findall('object'):
            name = object_tag.find('name').text
            if name not in self.label_map:
                continue  # Ismeretlen osztályt kihagyunk

            bndbox = object_tag.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Validálás: ne legyen negatív vagy 0 méretű
            if xmax <= xmin or ymax <= ymin:
                continue

            bounding_boxes.append([xmin, ymin, xmax, ymax])
            one_hot_classes.append(self._to_one_hot(name))

        if not bounding_boxes:
            return None, None

        return np.array(bounding_boxes, dtype=np.float32), np.array(one_hot_classes, dtype=np.float32)

    def _to_one_hot(self, name):
        """One-hot kódolást végez (pl. 'B' -> [0, 1, 0, 0])"""
        vec = [0] * self.num_classes
        if name in self.label_map:
            vec[self.label_map[name]] = 1
        return vec
