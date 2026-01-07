# src/core/data_prep/annotation_parser.py
import xml.etree.ElementTree as ET
import os


class AnnotationParser:
    @staticmethod
    def parse_voc_xml(xml_path):
        """
        Pascal VOC XML feldolgozása.
        Visszaadja a daganatok (objektumok) listáját.
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