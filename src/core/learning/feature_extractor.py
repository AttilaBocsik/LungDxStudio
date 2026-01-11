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
    def __init__(self, data_dir="processed_data"):
        self.data_dir = data_dir
        # Gabor kernelek inicializálása (ugyanaz, mint a régi project_utils-ban)
        self.gabor_kernels = self.create_gabor_kernels()
        print(f"✅ FeatureExtractor inicializálva. Gabor kernelek száma: {len(self.gabor_kernels)}")

    @staticmethod
    def create_gabor_kernels():
        """
        Gabor kernelek generálása (A régi kód alapján).
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
        Üres vagy nullás sorok törlése (A régi kód alapján).
        """
        if df.empty:
            return df

        # Köztes oszlopok: Image (első) és Label/PatientID (utolsó kettő) kivételével
        # Feltételezzük, hogy az utolsó két oszlop a Label és a patient_id
        middle_cols = df.iloc[:, 1:-2]

        # Feltétel: minden köztes cella 0 vagy NaN
        condition = ((middle_cols == 0) | (middle_cols.isna())).all(axis=1)

        # Csak azok maradnak, ahol NEM mind nulla
        df_cleaned = df[~condition]
        return df_cleaned

    @staticmethod
    def select_random_rows(df: pd.DataFrame, selected_values: list) -> pd.DataFrame:
        """
        Downsampling: Csak max 1000 sort tart meg címkénként, hogy ne fogyjon el a RAM.
        (A régi kód alapján).
        """
        if df.empty:
            return df

        filtered_dfs = []
        for value in selected_values:
            temp_df = df[df['Label'] == value]
            if not temp_df.empty:
                # Véletlenszerű mintavételezés (max 1000 sor)
                n_samples = min(1000, len(temp_df))
                sampled_df = temp_df.sample(n=n_samples, random_state=99)
                filtered_dfs.append(sampled_df)

        if not filtered_dfs:
            return pd.DataFrame(columns=df.columns)

        new_df = pd.concat(filtered_dfs).sort_index()
        return new_df

    def multi_filter(self, patient_id, img, tumor_type, lung_state):
        """
        Pixel-szintű szűrők alkalmazása (Gabor, Sobel, Gaussian, stb.).
        Visszatér egy DataFrame-mel, ahol minden sor egy pixel.
        """
        # --- Kép előkészítése ---
        # A régi kód RGB konverziót csinált, de az .npz-ben már szürkeárnyalatos (2D) képek vannak.
        # Ha float32, konvertáljuk uint8-ra vagy normalizáljuk, ha a szűrők azt igénylik.
        # Itt feltételezzük, hogy a bemenet 2D numpy array.

        # Másolat készítése, hogy ne írjuk felül az eredetit
        img2 = img.copy()

        # Ha nem uint8, konvertálhatjuk (cv2 szűrők néha igénylik, de float32-vel is mennek)
        # A régi kódban: img2 = img.reshape(-1) -> Ez az oszlopvektor

        df = pd.DataFrame()

        # 1. Oszlop: Eredeti pixel értékek
        df["Image"] = img2.reshape(-1)

        # 2. Gabor szűrők
        num = 1
        for gabor in self.gabor_kernels:
            gabor_label = 'Gabor' + str(num)
            # filter2D elfogad float32-t is
            fimg = cv2.filter2D(img2.astype('float32'), cv2.CV_32F, gabor)
            df[gabor_label] = fimg.reshape(-1)
            num += 1

        # 3. Sobel
        edge_sobel = sobel(img2)
        df['Sobel'] = edge_sobel.reshape(-1)

        # 4. Gaussian (sigma=3)
        gaussian_img = nd.gaussian_filter(img2, sigma=3)
        df['Gaussian_s3'] = gaussian_img.reshape(-1)

        # 5. Gaussian (sigma=7)
        gaussian_img2 = nd.gaussian_filter(img2, sigma=7)
        df['Gaussian_s7'] = gaussian_img2.reshape(-1)

        # 6. Median (size=3)
        median_img = nd.median_filter(img2, size=3)
        df['Median_s3'] = median_img.reshape(-1)

        # 7. Variance (size=3)
        variance_img = nd.generic_filter(img2, np.var, size=3)
        df['Variance_s3'] = variance_img.reshape(-1)

        # --- Címkézés (Labeling) ---
        # A régi logika alapján számkódokat rendelünk a pixelekhez
        label_value = 0

        if lung_state == "healthy_lungs":  # 1. Egészséges tüdő (Parenchyma)
            label_value = 1
        elif lung_state == "diseased_lungs":  # 2. Beteg tüdő (Teljes kép)
            if tumor_type == 'A':
                label_value = 4
            elif tumor_type == 'B':
                label_value = 5
            elif tumor_type == 'D':
                label_value = 6
            elif tumor_type == 'G':
                label_value = 7
            else:
                label_value = 0
        elif lung_state == "healthy_soft_tissue":  # 4. Egészséges szövet (ROI Context)
            label_value = 1
        elif lung_state == "diseased_soft_tissue":  # 3. Beteg szövet (Masked Tumor)
            if tumor_type == 'A':
                label_value = 8
            elif tumor_type == 'B':
                label_value = 10
            elif tumor_type == 'D':
                label_value = 12
            elif tumor_type == 'G':
                label_value = 14
            else:
                label_value = 0

        df["Label"] = label_value
        df["patient_id"] = patient_id

        return df

    def extract_features(self):
        """
        Ez a metódus helyettesíti a régi 'preprocessing_images'-t.
        Végigmegy az összes .npz fájlon, és létrehozza a nagy tanító táblázatot.
        """
        npz_files = glob.glob(os.path.join(self.data_dir, "*.npz"))

        if not npz_files:
            print("❌ Nincsenek .npz fájlok a processed_data mappában.")
            return None

        print(f"🔄 Pixel-szintű jellemzők kinyerése {len(npz_files)} fájlból...")

        dfs_to_merge = []  # Ide gyűjtjük a kisebb DataFrame-eket

        for file_path in tqdm(npz_files, desc="Feldolgozás"):
            try:
                # 1. Betöltjük az .npz fájlt (Lazy Loading helyett itt memóriába vesszük)
                with np.load(file_path) as data:
                    # Kinyerjük a képeket és metaadatokat
                    # [0] Eredeti -> data['original']
                    # [1] Parenchyma -> data['parenchyma']
                    # [2] Masked Tumor -> data['masked_tumor']
                    # [3] Inverted ROI -> data['inverted_roi']
                    # [4] Label -> data['label']
                    # [5] Patient ID -> data['patient_id']

                    img_original = data['original']
                    img_parenchyma = data['parenchyma']
                    img_tumor = data['masked_tumor']
                    img_roi_context = data['inverted_roi']

                    label = str(data['label'])
                    p_id = str(data['patient_id'])

                # 2. Szűrési lépések (ugyanaz a sorrend, mint a régiben)

                # --- A) Beteg tüdő (Teljes kép - Original) ---
                df_orig = self.multi_filter(p_id, img_original, label, lung_state="diseased_lungs")
                df_orig = self.remove_null_rows(df_orig)
                df_orig = self.select_random_rows(df_orig, [0, 4, 5, 6, 7])

                # --- B) Egészséges tüdő (Parenchyma) ---
                df_par = self.multi_filter(p_id, img_parenchyma, label, lung_state="healthy_lungs")
                df_par = self.remove_null_rows(df_par)
                df_par = self.select_random_rows(df_par, [0, 1])

                # --- C) Beteg lágyszövet (Tumor) ---
                df_tum = self.multi_filter(p_id, img_tumor, label, lung_state="diseased_soft_tissue")
                df_tum = self.remove_null_rows(df_tum)
                df_tum = self.select_random_rows(df_tum, [0, 8, 10, 12, 14])

                # --- D) Egészséges lágyszövet (ROI Context) ---
                df_roi = self.multi_filter(p_id, img_roi_context, label, lung_state="healthy_soft_tissue")
                df_roi = self.remove_null_rows(df_roi)
                df_roi = self.select_random_rows(df_roi, [0, 1])

                # Hozzáadjuk a listához
                dfs_to_merge.extend([df_orig, df_par, df_tum, df_roi])

            except Exception as e:
                print(f"⚠️ Hiba a fájlnál ({os.path.basename(file_path)}): {e}")

        # 3. Összefűzés (Final Merge)
        if dfs_to_merge:
            print("📊 Adatok egyesítése egyetlen DataFrame-be...")
            df_all = pd.concat(dfs_to_merge, ignore_index=True)

            # Utólagos tisztítás (ahogy a régi kódban volt)
            df_all.loc[df_all['Image'] == 0.0, 'Label'] = 0

            print(f"✅ Kész! Eredmény mérete: {df_all.shape}")
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