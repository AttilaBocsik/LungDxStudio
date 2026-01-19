# src/core/data_prep/annotation_parser.py
import xml.etree.ElementTree as ET
import os


class AnnotationParser:
    """
    Annotációs fájlok (pl. XML) beolvasását és értelmezését végző segédosztály.

    Ez az osztály statikus metódusokat biztosít a különböző annotációs formátumok
    kezeléséhez. Jelenleg a Pascal VOC szabványú XML fájlok feldolgozását támogatja,
    amelyek a daganatok vagy egyéb objektumok befoglaló téglalapjait (bbox) tartalmazzák.
    """

    @staticmethod
    def parse_voc_xml(xml_path):
        """
        Egy Pascal VOC formátumú XML annotációs fájl beolvasása és feldolgozása.

        A függvény megnyitja a megadott XML fájlt, végigiterál az összes benne található
        `object` elemen, és kinyeri a címkét (name), valamint a befoglaló téglalap
        (bndbox) koordinátáit. Ezen felül származtatott adatokat (középpont, terület)
        is számol.

        Args:
            xml_path (str): A feldolgozandó XML fájl teljes elérési útja.

        Returns:
            list of dict: Az azonosított objektumok (annotációk) listája.
                Ha a fájl nem létezik vagy hiba történik, üres listával tér vissza.

                Egy lista elem (szótár) felépítése:
                {
                    "label" (str): Az objektum osztálya/neve (pl. 'tumor', 'A').
                    "bbox" (tuple): (xmin, ymin, xmax, ymax) egész számként.
                    "center" (tuple): A téglalap középpontja ((x+x)/2, (y+y)/2) float-ként.
                    "area" (int): A befoglaló téglalap területe pixelben.
                }
        """
        objects = []
        if not os.path.exists(xml_path):
            return objects

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')

                # Koordináták kinyerése
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                objects.append({
                    "label": name,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center": ((xmin + xmax) / 2, (ymin + ymax) / 2),
                    "area": (xmax - xmin) * (ymax - ymin)
                })
        except Exception as e:
            print(f"Hiba az XML olvasásakor ({xml_path}): {e}")

        return objects
