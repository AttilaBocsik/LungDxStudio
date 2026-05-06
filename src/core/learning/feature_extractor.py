# src/core/learning/feature_extractor.py
import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage as nd
from skimage.filters import sobel


class FeatureExtractor:
    """
    Képjellemzők kinyeréséért és adathalmaz összeállításáért felelős osztály.

    Ez az osztály .npz fájlokból olvassa be a szegmentált CT képeket, különböző
    képfeldolgozó szűrőket (Gabor, Sobel, Gauss, stb.) alkalmaz rajtuk pixel-szinten,
    majd az eredményeket egy strukturált pandas DataFrame-be gyűjti össze a gépi tanuláshoz.

    Attributes:
        data_dir (str): A feldolgozott (.npz) fájlok forráskönyvtára.
        gabor_kernels (list): A generált Gabor-szűrő magok listája.
    """

    def __init__(self, data_dir="processed_data"):
        """
        Inicializálja a FeatureExtractor-t és legenerálja a szűrőmagokat.

        Args:
            data_dir (str, optional): A bemeneti adatok mappája. Alapértelmezett: "processed_data".
        """
        self.data_dir = data_dir
        self.gabor_kernels = self.create_gabor_kernels()
        print(f"✅ FeatureExtractor inicializálva. Gabor kernelek száma: {len(self.gabor_kernels)}")

    @staticmethod
    def create_gabor_kernels():
        """
        Gabor-szűrő magok (kernels) generálása különböző paraméterekkel.

        Returns:
            list: OpenCV Gabor kernel objektumok listája.
        """
        kernels = []
        ksize = 3
        thetas = [0, np.pi / 4]
        sigmas = [1, 3]
        lamdas = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        gammas = [0.05, 0.5]
        psi = 0
        for theta in thetas:
            for sigma in sigmas:
                for lamda in lamdas:
                    for gamma in gammas:
                        kernel = cv2.getGaborKernel(
                            (ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F
                        )
                        kernels.append(kernel)
        return kernels

    @staticmethod
    def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Eltávolítja a háttérnek minősülő (0 értékű) pixeleket a táblázatból.

        Args:
            df (pd.DataFrame): A bemeneti jellemzőtábla.

        Returns:
            pd.DataFrame: A megtisztított táblázat, amely csak releváns pixeleket tartalmaz.
        """
        if df.empty:
            return df

        # 1. Stratégia: Ha az EREDETI pixel (Image oszlop) 0, akkor az háttér.
        if 'Image' in df.columns:
            # Csak azokat tartjuk meg, ahol az eredeti kép abszolút értéke nagyobb mint 0
            # (1e-6 a lebegőpontos hibák miatt biztonságosabb mint a sima 0)
            df_cleaned = df[df['Image'].abs() > 1e-6]
            return df_cleaned

        # 2. Stratégia (Fallback): Ha valamiért nincs Image oszlop
        middle_cols = df.iloc[:, 1:-2]
        condition = ((middle_cols == 0) | (middle_cols.isna())).all(axis=1)
        df_cleaned = df[~condition]
        return df_cleaned

    @staticmethod
    def select_random_rows(df: pd.DataFrame, selected_values: list, n_limit: int = 2000) -> pd.DataFrame:
        """
        Véletlenszerű mintavételezés (downsampling) a kiegyensúlyozott adathalmazért.

        Címkénként korlátozza a pixelek számát, hogy a modell ne tanuljon el részrehajlást
        a túlreprezentált osztályok irányába.

        Args:
            df (pd.DataFrame): Forrás adatok.
            selected_values (list): A megtartandó Label értékek listája.
            n_limit (int): Maximális mintaszám címkénként. Alapértelmezett: 2000.

        Returns:
            pd.DataFrame: A mintavételezett adathalmaz.
        """
        if df.empty:
            return df

        filtered_dfs = []
        for value in selected_values:
            temp_df = df[df['Label'] == value]
            if not temp_df.empty:
                # Ha kevesebb pixel van, mint a limit, az összeset kéri,
                # ha több, akkor pontosan n_limit darabot.
                count = min(n_limit, len(temp_df))
                sampled_df = temp_df.sample(n=count, random_state=42)
                filtered_dfs.append(sampled_df)

        if not filtered_dfs:
            return pd.DataFrame(columns=df.columns)

        return pd.concat(filtered_dfs).sort_index()

    def multi_filter(self, patient_id, img, tumor_type, lung_state):
        """
        Különböző digitális képszűrők alkalmazása egy adott képre.

        Létrehozza a jellemzővektorokat: Gabor-válaszok, Sobel-élkeresés,
        Gauss-simítás, medián szűrő és variancia. Hozzárendeli a megfelelő
        címkét (Label) a tüdő állapota és a daganat típusa alapján.

        Args:
            patient_id (str): A páciens azonosítója.
            img (np.ndarray): A bemeneti kép (pixel array).
            tumor_type (str): A daganat típusa (A, B, G, D).
            lung_state (str): A szövet típusa (pl. 'diseased_lungs', 'healthy_soft_tissue').

        Returns:
            pd.DataFrame: Egy táblázat, ahol minden sor egy pixel, az oszlopok pedig a szűrt értékek.
        """
        # Másolat készítése
        img2 = img.copy()

        df = pd.DataFrame()

        # 1. Oszlop: Eredeti pixel értékek
        df["Image"] = img2.reshape(-1)

        # 2. Gabor szűrők
        num = 1
        for gabor in self.gabor_kernels:
            gabor_label = 'Gabor' + str(num)
            fimg = cv2.filter2D(img2.astype('float32'), cv2.CV_32F, gabor)
            df[gabor_label] = fimg.reshape(-1)
            num += 1

        # 3. Egyéb szűrők
        df['Sobel'] = sobel(img2).reshape(-1)
        df['Gaussian_s3'] = nd.gaussian_filter(img2, sigma=3).reshape(-1)
        df['Gaussian_s7'] = nd.gaussian_filter(img2, sigma=7).reshape(-1)
        df['Median_s3'] = nd.median_filter(img2, size=3).reshape(-1)
        df['Variance_s3'] = nd.generic_filter(img2, np.var, size=3).reshape(-1)

        # --- Címkézés ---
        label_value = 0
        if lung_state == "healthy_lungs":
            label_value = 1
        elif lung_state == "diseased_lungs":
            if tumor_type == 'A':
                label_value = 4
            elif tumor_type == 'B':
                label_value = 5
            elif tumor_type == 'D':
                label_value = 6
            elif tumor_type == 'G':
                label_value = 7
        elif lung_state == "healthy_soft_tissue":
            label_value = 1
        elif lung_state == "diseased_soft_tissue":
            if tumor_type == 'A':
                label_value = 8
            elif tumor_type == 'B':
                label_value = 10
            elif tumor_type == 'D':
                label_value = 12
            elif tumor_type == 'G':
                label_value = 14

        df["Label"] = label_value
        # JAVÍTÁS: Biztosítjuk, hogy a patient_id minden sorba bekerüljön
        df["patient_id"] = str(patient_id)

        return df

    def extract_features(self):
        """
        A teljes jellemzőkinyerési folyamat vezérlése.

        Végigmegy az összes .npz fájlon, végrehajtja a szűrést, a tisztítást és a
        mintavételezést, majd összevont statisztikát készít a páciensekről.

        Returns:
            pd.DataFrame: Az összesített, kevert (shuffled) tanító adathalmaz.
        """
        npz_files = glob.glob(os.path.join(self.data_dir, "*.npz"))

        if not npz_files:
            print("❌ Nincsenek .npz fájlok a processed_data mappában.")
            return None

        print(f"🔄 Jellemzők kinyerése {len(npz_files)} fájlból...")

        dfs_to_merge = []

        for file_path in tqdm(npz_files, desc="Feldolgozás"):
            try:
                with np.load(file_path) as data:
                    img_original = data['original']
                    img_parenchyma = data['parenchyma']
                    img_tumor = data['masked_tumor']
                    img_roi_context = data['inverted_roi']
                    label = str(data['label'])

                    # Páciens ID tisztítása
                    raw_id = data['patient_id']
                    p_id = str(raw_id).replace("['", "").replace("']", "")

                # --- Mintavételezés (Szeletenként és képtípusonként 2000 minta) ---

                # A) Beteg tüdő
                df_orig = self.multi_filter(p_id, img_original, label, lung_state="diseased_lungs")
                df_orig = self.remove_null_rows(df_orig)
                df_orig = self.select_random_rows(df_orig, [0, 4, 5, 6, 7], n_limit=2000)

                # B) Egészséges tüdő
                df_par = self.multi_filter(p_id, img_parenchyma, label, lung_state="healthy_lungs")
                df_par = self.remove_null_rows(df_par)
                df_par = self.select_random_rows(df_par, [0, 1], n_limit=2000)

                # C) Daganat
                df_tum = self.multi_filter(p_id, img_tumor, label, lung_state="diseased_soft_tissue")
                df_tum = self.remove_null_rows(df_tum)
                df_tum = self.select_random_rows(df_tum, [0, 8, 10, 12, 14], n_limit=2000)

                # D) ROI Context
                df_roi = self.multi_filter(p_id, img_roi_context, label, lung_state="healthy_soft_tissue")
                df_roi = self.remove_null_rows(df_roi)
                df_roi = self.select_random_rows(df_roi, [0, 1], n_limit=2000)

                dfs_to_merge.extend([df_orig, df_par, df_tum, df_roi])

            except Exception as e:
                print(f"⚠️ Hiba a fájlnál ({os.path.basename(file_path)}): {e}")

        if dfs_to_merge:
            print("\n📊 Adatok egyesítése és végső simítások...")
            df_all = pd.concat(dfs_to_merge, ignore_index=True)

            # Tisztítás
            df_all.loc[df_all['Image'] == 0.0, 'Label'] = 0

            # --- STATISZTIKA KÉSZÍTÉSE ---
            print("\n" + "=" * 50)
            print("        📊 PÁCIENS SZINTŰ STATISZTIKA")
            print("=" * 50)

            # Megszámoljuk, melyik páciensből hány sor (pixel) került be
            stats = df_all['patient_id'].value_counts()

            for p_name, count in stats.items():
                # Kiszámoljuk a daganatos pixelek arányát is az adott páciensnél
                p_data = df_all[df_all['patient_id'] == p_name]
                tumor_pixels = len(p_data[p_data['Label'] > 1])
                print(f"👤 Páciens: {p_name:<20} | Összes pixel: {count:>6} | Daganatos: {tumor_pixels:>6}")

            print("-" * 50)
            print(f"📈 ÖSSZESEN: {len(df_all)} sor a CSV fájlban.")
            print("=" * 50 + "\n")

            # Keverés (Shuffle)
            print("🔀 Adatok összekeverése...")
            df_all = df_all.sample(frac=1).reset_index(drop=True)

            return df_all
        else:
            return pd.DataFrame()

    def save_to_csv(self, df, output_path="training_data_pixelwise.csv"):
        """Mentés CSV-be."""
        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            print(f"💾 Mentve: {output_path}")
        else:
            print("⚠️ Nincs mit menteni (üres DataFrame).")

    def save_to_parquet(self, df, parquet_path):
        """Mentés Parquet-be."""
        if df is not None and not df.empty:
            df.to_parquet(parquet_path, index=False)
            print(f"💾 Mentve: {parquet_path}")
        else:
            print("⚠️ Nincs mit menteni (üres DataFrame).")
