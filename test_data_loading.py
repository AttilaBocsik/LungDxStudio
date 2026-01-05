# test_data_loading.py
from src.core.data_manager import DataManager
import matplotlib.pyplot as plt
import cv2

# ÁLLÍTSD BE A SAJÁT ÚTVONALAIDAT!
DICOM_DIR = r"D:/GitProjects/LungDxStudio/Data/Train/DICOM"  # pl: C:/Data/Train/DICOM
XML_DIR = r"D:/GitProjects/LungDxStudio/Data/Train/ANNOTATION"  # pl: C:/Data/Train/XML


def main():
    # 1. Manager példányosítása
    manager = DataManager(DICOM_DIR, XML_DIR)

    # 2. Indexelés
    manager.index_files()

    # 3. Teszt: vegyünk ki 5 példát és jelenítsük meg
    print("\n--- Megjelenítés teszt ---")
    gen = manager.get_data_generator()

    for i in range(25):
        try:
            data = next(gen)
            print(f"[{i + 1}] Feldolgozva: {data['uid']}")
            print(f"    Dobozok száma: {len(data['boxes'])}")

            # Gyors vizualizáció
            img = data['image'].copy()
            # Normalizálás 0-255 közé a megjelenítéshez
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for box in data['boxes']:
                xmin, ymin, xmax, ymax = box.astype(int)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            plt.imshow(img)
            plt.title(f"Minta {i + 1}")
            plt.show()

        except StopIteration:
            print("Nincs több adat.")
            break


if __name__ == "__main__":
    main()