import os
from datetime import datetime
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from xgboost import dask as dxgb

# ÚJ: MLflow importok a modern mentéshez
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

# Kiértékelési metrikák
from sklearn.metrics import (
    root_mean_squared_error, recall_score, f1_score,
    jaccard_score, confusion_matrix, accuracy_score
)


class XGBoostTrainer:
    """
    XGBoost modell tanítását és kiértékelését végző osztály Dask környezetben,
    kiegészítve lokális MLflow Tracking és Registry támogatással.
    """

    def __init__(self, csv_file_path, resource_folder, config, client):
        """
        Args:
            csv_file_path (str): A bemeneti jellemzőtábla (CSV) elérési útja.
            resource_folder (str): A modell mentési helye (Legacy).
            config (dict): Konfigurációs beállítások (pl. modell neve).
            client (dask.distributed.Client): A Dask cluster kliense a párhuzamosításhoz.
        """
        self.csv_file_path = csv_file_path
        self.resource_folder = resource_folder
        self.config = config
        self.client = client

        # ÚJ: Beállítjuk az asztali PC-n futó MLflow szerver elérhetőségét
        # Ha a config-ban nincs megadva, alapértelmezetten a http://localhost:5000-et használja
        mlflow_uri = self.config.get("mlflow_uri", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("pulmoflow-desktop-training")

    def train(self, log_callback, do_split=True):
        if not os.path.exists(self.csv_file_path):
            log_callback(f"⚠️ Nem található {self.csv_file_path} fájl.")
            return False

        try:
            log_callback(f"⏳ Adatok betöltése a {'Teszt' if do_split else 'Végleges'} módhoz...")
            origin_ddf = dd.read_parquet(self.csv_file_path)

            for col in ['Unnamed: 0.1', 'Unnamed: 0']:
                if col in origin_ddf.columns:
                    origin_ddf = origin_ddf.drop(columns=[col])

            origin_ddf = origin_ddf.persist()
            y = origin_ddf['Label'].astype('int')
            X = origin_ddf.drop(['Label', 'patient_id'], axis=1)

            if X.npartitions != y.npartitions:
                log_callback(f"🔧 Partíciók javítása: {X.npartitions} vs {y.npartitions}")
                y = y.repartition(npartitions=X.npartitions)

            if do_split:
                log_callback("✂️ Adatok felosztása: 80% Tanító, 20% Teszt.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                dtrain = dxgb.DaskDMatrix(self.client, X_train, y_train)
                dtest = dxgb.DaskDMatrix(self.client, X_test, y_test)
            else:
                log_callback("🚀 Végleges mód: Az összes adat (100%) felhasználása tanításhoz.")
                dtrain = dxgb.DaskDMatrix(self.client, X, y)
                dtest = None

            # Multi-class Paraméterek
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

            log_callback("🚀 XGBoost tanítás indítása Dask-on keresztül...")
            model = dxgb.train(
                self.client,
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, "train")]
            )

            # Kinyerjük a tiszta booster objektumot a Dask wrapperből
            booster = model["booster"]

            # -------------------------------------------------------------------------
            # MÓDOSÍTVA: NATÍV MLFLOW MENTÉS LOKÁLISAN AZ ASZTALI PC-N
            # -------------------------------------------------------------------------
            model_name = "PulmoFlow_Model"
            log_callback(f"📦 Modell naplózása az asztali MLflow-ba '{model_name}' néven...")

            # Adatbemeneti séma (signature) automatikus kinyerése a Dask DataFrame-ből
            input_sample = X.head(5)
            signature = mlflow.models.infer_signature(input_sample, np.zeros(5))

            with mlflow.start_run() as run:
                # Elmentjük az XGBoost boostert az MLflow-ba, és regisztráljuk a lokális nyilvántartásba
                mlflow.xgboost.log_model(
                    xgb_model=booster,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=model_name
                )
                log_callback(f"✅ Modell sikeresen elmentve az MLflow-ba! Run ID: {run.info.run_id}")

                # Ha végleges (100%-os) módban tanítottunk, automatikusan jelöljük meg Production fázisként lokálisan
                if not do_split:
                    client = MlflowClient()
                    latest_versions = client.get_latest_versions(model_name, stages=["None"])
                    if latest_versions:
                        current_version = latest_versions[0].version
                        client.transition_model_version_stage(
                            name=model_name,
                            version=current_version,
                            stage="Production"
                        )
                        log_callback(f"🚀 Lokális fázis átállítva: Version {current_version} -> Production")

            # -------------------------------------------------------------------------
            # Kiértékelés (csak ha teszt módban futott)
            # -------------------------------------------------------------------------
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
                log_callback("ℹ️ Végleges tanítás sikeresen befejezve. Letöltheted a ZIP fájlt az MLflow UI-ról.")

            return True

        except Exception as e:
            log_callback(f"❌ Hiba a tanítás során: {str(e)}")
            import traceback
            log_callback(traceback.format_exc())
            return False