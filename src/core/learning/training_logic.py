import os
import joblib
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from xgboost import dask as dxgb


class XGBoostTrainer:
    def __init__(self, csv_file_path, resource_folder, config, client):
        self.csv_file_path = csv_file_path
        self.resource_folder = resource_folder
        self.config = config
        self.client = client
        self.stop_requested = False  # Lehetőség a megállításra

    def train(self, log_callback):
        """
        A log_callback egy függvény, amit meghívunk, ha üzenni akarunk a GUI-nak.
        """
        if not os.path.exists(self.csv_file_path):
            log_callback(f"⚠️ Nem található {self.csv_file_path} fájl.")
            return False  # Hibával térünk vissza

        try:
            log_callback("⏳ Adatok betöltése és előkészítése Dask segítségével...")

            origin_ddf = dd.read_csv(self.csv_file_path)
            # Újraparticionálás, ha szükséges (CPU magok számától függően)
            origin_ddf = origin_ddf.repartition(npartitions=2)

            # Felesleges oszlopok tisztítása
            for col in ['Unnamed: 0.1', 'Unnamed: 0']:
                if col in origin_ddf.columns:
                    origin_ddf = origin_ddf.drop(columns=[col])

            # Label és Feature szétválasztás
            y = origin_ddf['Label'].astype('int')
            # pi = origin_ddf['patient_id'] # Ha nem használjuk a tréninghez, itt nem kell tárolni
            X = origin_ddf.drop(['Label', 'patient_id'], axis=1)

            # Split
            log_callback("✂️ Adatok felosztása (80% Train - 20% Test)...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Memória tisztítás (Dask-nál a lazy eval miatt ez kevésbé kritikus, de nem árt)
            del origin_ddf

            # Paraméterek
            params = {
                'objective': 'multi:softprob',
                'num_class': 15,
                'eval_metric': 'mlogloss',
                'max_depth': 4,
                'eta': 0.02,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'tree_method': 'hist',  # CPU-n kritikus
                'max_bin': 256,
            }

            log_callback("⚙️ Dask DMatrix előkészítése...")
            dtrain = dxgb.DaskDMatrix(self.client, X_train, y_train)
            # dtest = dxgb.DaskDMatrix(self.client, X_test, y_test) # Opcionális validációhoz

            log_callback("🚀 Modell tanításának indítása (ez eltarthat egy ideig)...")

            num_round = 1000
            model = dxgb.train(
                self.client,
                params,
                dtrain,
                num_boost_round=num_round,
                evals=[(dtrain, "train")]
            )

            booster = model["booster"]

            # Mentés
            pkl_path = f"{self.resource_folder}/{self.config['model-name']}"

            if os.path.exists(pkl_path):
                os.remove(pkl_path)
                log_callback(f"♻️ Régi modell törölve: {pkl_path}")

            log_callback(f"💾 Új modell mentése ide: {pkl_path} ...")
            joblib.dump(booster, pkl_path)

            log_callback("✅ Modell sikeresen létrehozva és elmentve.")
            return True

        except Exception as e:
            log_callback(f"❌ Hiba történt a tanítás során: {str(e)}")
            return False