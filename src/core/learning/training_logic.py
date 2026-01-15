import os
import joblib
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from xgboost import dask as dxgb

# Kiértékelési metrikák
from sklearn.metrics import (
    root_mean_squared_error, recall_score, f1_score,
    jaccard_score, confusion_matrix, accuracy_score
)


class XGBoostTrainer:
    def __init__(self, csv_file_path, resource_folder, config, client):
        self.csv_file_path = csv_file_path
        self.resource_folder = resource_folder
        self.config = config
        self.client = client

    def train(self, log_callback, do_split=True):
        """
        do_split: Ha True, 80/20 arányban tesztel. Ha False, 100% adaton tanít.
        """
        if not os.path.exists(self.csv_file_path):
            log_callback(f"⚠️ Nem található {self.csv_file_path} fájl.")
            return False

        try:
            log_callback(f"⏳ Adatok betöltése a {'Teszt' if do_split else 'Végleges'} módhoz...")
            origin_ddf = dd.read_csv(self.csv_file_path)

            # Tisztítás
            for col in ['Unnamed: 0.1', 'Unnamed: 0']:
                if col in origin_ddf.columns:
                    origin_ddf = origin_ddf.drop(columns=[col])

            y = origin_ddf['Label'].astype('int')
            X = origin_ddf.drop(['Label', 'patient_id'], axis=1)

            # --- Mód választás: Split vagy Full ---
            if do_split:
                log_callback("✂️ Adatok felosztása: 80% Tanító, 20% Teszt.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                dtrain = dxgb.DaskDMatrix(self.client, X_train, y_train)
                dtest = dxgb.DaskDMatrix(self.client, X_test, y_test)
            else:
                log_callback("🚀 Végleges mód: Az összes adat (100%) felhasználása tanításhoz.")
                dtrain = dxgb.DaskDMatrix(self.client, X, y)
                dtest = None

            # Paraméterek
            params = {
                'objective': 'multi:softprob',
                'num_class': 15,
                'eval_metric': 'mlogloss',
                'max_depth': 4,
                'eta': 0.02,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'tree_method': 'hist',
                'max_bin': 256,
            }

            log_callback("🚀 XGBoost tanítás indítása...")
            model = dxgb.train(
                self.client,
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, "train")]
            )

            booster = model["booster"]

            # Mentés
            # Ha végleges, adjunk egy '_final' jelzést a fájlnévhez (opcionális, de hasznos)
            base_name = self.config['model-name']
            if not do_split:
                base_name = base_name.replace(".pkl", "_final.pkl")

            pkl_path = f"{self.resource_folder}/{base_name}"

            if os.path.exists(pkl_path):
                os.remove(pkl_path)

            joblib.dump(booster, pkl_path)
            log_callback(f"💾 Modell elmentve: {pkl_path}")

            # --- Kiértékelés (csak ha kértünk tesztelést) ---
            if do_split and dtest is not None:
                log_callback("📊 Kiértékelés a tesztadatokon...")
                y_dask_pred = dxgb.predict(self.client, booster, dtest)

                # Számítások
                y_pred_prob = y_dask_pred.compute()
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_true = y_test.compute().to_numpy()

                # Metrikák
                accuracy = accuracy_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro')
                rmse = root_mean_squared_error(y_true, y_pred)
                conf_matrix = confusion_matrix(y_true, y_pred)

                log_callback('-' * 45)
                log_callback(f"🏆 EREDMÉNYEK (80/20 SPLIT):")
                log_callback(f"   Pontosság (Accuracy): {accuracy * 100:.2f}%")
                log_callback(f"   Recall (Weighted):    {recall:.4f}")
                log_callback(f"   F1 Score (Macro):     {f1:.4f}")
                log_callback(f"   RMSE:                 {rmse:.4f}")
                log_callback(f"   Confusion Matrix:\n{conf_matrix}")
                log_callback('-' * 45)
            else:
                log_callback("ℹ️ Végleges tanítás befejezve. Nincs tesztelési fázis.")

            return True

        except Exception as e:
            log_callback(f"❌ Hiba a tanítás során: {str(e)}")
            import traceback
            log_callback(traceback.format_exc())
            return False