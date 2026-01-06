import sys
import numpy as np
import warnings
from src.core.segmentation.lung_segmenter import LungSegmenter


def test_morphology_fix():
    print("--- MORPHOLOGY JAVÍTÁS TESZTELÉSE ---")

    # 1. Hozzunk létre egy teszt képet (512x512)
    # Ez egy egyszerű szimuláció: zajos háttér + egy sötét folt (tüdő)
    img = np.random.randint(0, 100, (512, 512), dtype=np.int16)  # Háttér (világosabb)

    # Csinálunk egy "tüdőt" a közepére (sötét terület, -500 HU alatt)
    y, x = np.ogrid[:512, :512]
    mask_circle = (x - 256) ** 2 + (y - 256) ** 2 <= 100 ** 2
    img[mask_circle] = -1000  # Levegő sűrűség

    print("1. Teszt kép létrehozva (512x512).")

    segmenter = LungSegmenter()

    # 2. Figyeljük a Warningokat futás közben
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Minden warningot elkapunk

        print("2. Szegmentálás futtatása...")
        try:
            mask = segmenter.segment_mask(img)
            pixel_count = np.sum(mask > 0)
            print(f"   -> Eredmény: {pixel_count} pixel a maszkon.")
        except Exception as e:
            print(f"❌ HIBA TÖRTÉNT: {e}")
            return

        # 3. Kiértékelés
        found_deprecated = False
        for warning in w:
            if "deprecated" in str(warning.message) or "binary_closing" in str(warning.message):
                found_deprecated = True
                print(f"⚠️ FIGYELEM: Még mindig kaptunk Warningot: {warning.message}")

    if not found_deprecated:
        print("\n✅ SIKER! A 'binary_closing' Warning eltűnt.")
        print("   A kód kompatibilis az új scikit-image verzióval.")
    else:
        print("\n❌ SIKERTELEN. A Warning még mindig jelen van.")


if __name__ == "__main__":
    test_morphology_fix()