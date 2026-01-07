# debug_npz_viewer.py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def check_saved_files():
    # Megkeressük az összes .npz fájlt a processed_data mappában
    data_folder = "src/gui/processed_data"
    files = glob.glob(os.path.join(data_folder, "*.npz"))

    if not files:
        print(f"❌ Nem található feldolgozott fájl a '{data_folder}' mappában.")
        print("Futtasd le a GUI-ban a 'Feldolgozás' folyamatot előbb!")
        return

    print(f"📂 Talált fájlok száma: {len(files)}")

    # Kiválasztjuk az utolsó mentett fájlt (vagy módosíthatod az indexet)
    file_path = files[0]
    print(f"🔍 Megtekintés: {file_path}")

    try:
        with np.load(file_path) as data:
            # Kiírjuk a metaadatokat a konzolba
            print("-" * 30)
            print(f"Páciens ID: {data['patient_id']}")
            print(f"Diagnózis (Címke): {data['label']}")
            print(f"Elérhető kulcsok a fájlban: {list(data.keys())}")
            print("-" * 30)

            # Képek megjelenítése
            plt.style.use('dark_background')  # Hogy jobban nézzen ki
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
            fig.suptitle(
                f"Feldolgozott szelet: {os.path.basename(file_path)}\nPatient: {data['patient_id']} | Label: {data['label']}",
                fontsize=14)

            # 1. Eredeti
            axes[0].imshow(data['original'], cmap='gray')
            axes[0].set_title("1. Original (HU)")
            axes[0].axis('off')

            # 2. Parenchyma
            axes[1].imshow(data['parenchyma'], cmap='gray')
            axes[1].set_title("2. Lung Parenchyma")
            axes[1].axis('off')

            # 3. Tumor
            axes[2].imshow(data['masked_tumor'], cmap='gray')
            axes[2].set_title("3. Masked Tumor (ROI)")
            axes[2].axis('off')

            # 4. Inverted ROI
            axes[3].imshow(data['inverted_roi'], cmap='gray')
            axes[3].set_title("4. Inverted ROI Context")
            axes[3].axis('off')

            plt.tight_layout()
            print("📈 Megjelenítés folyamatban...")
            plt.show()

    except Exception as e:
        print(f"❌ Hiba a fájl beolvasása közben: {e}")


if __name__ == "__main__":
    check_saved_files()