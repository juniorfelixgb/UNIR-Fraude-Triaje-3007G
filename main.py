import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suprimir INFO y WARNING de TF

from pathlib import Path

import mlflow
import mlflow.data
import mlflow.keras
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
EXPERIMENT_NAME = "fraude-triaje-unir"
MODEL_NAME = "fraud-detection-hybrid"

PARAMS = {
    "seed": 42,
    "xgb_n_estimators": 100,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 6,
    "nn_epochs": 50,
    "nn_batch_size": 32,
    "nn_patience": 6,
    "meta_epochs": 20,
    "distilbert_model": "distilbert-base-multilingual-cased",
    "distilbert_max_length": 120,
    "test_size": 0.20,
    "meta_size": 0.25,
    "val_size": 0.20,
}


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_models=False)
    mlflow.tensorflow.autolog(log_models=False)
    print(f"[MLflow] Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"[MLflow] Experiment   : {EXPERIMENT_NAME}")


def run_training():
    from ml.train import run_pipeline

    data_path = PROJECT_ROOT / "ml" / "data" / "dataset_reclamos_ia_ruidoso_extremo.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {data_path}. "
            "Ejecuta primero el notebook 01_GenerarDataset.ipynb."
        )

    with mlflow.start_run(run_name="hybrid_stacking") as run:
        mlflow.log_params(PARAMS)

        # Registrar el dataset en MLflow
        df = pd.read_excel(data_path)
        dataset = mlflow.data.from_pandas(
            df,
            source=str(data_path),
            name="dataset_reclamos_ia_ruidoso_extremo",
            targets="Prediccion_Fraude",
        )
        mlflow.log_input(dataset, context="training")
        print(f"[MLflow] Dataset registrado: {len(df):,} registros, {df.columns.size} columnas")
        del df  # liberar memoria antes del pipeline

        metrics, artifacts = run_pipeline(PROJECT_ROOT, PARAMS)

        mlflow.log_metrics(metrics)
        print("\n[MLflow] Métricas:")
        for k, v in metrics.items():
            print(f"         {k}: {v:.4f}")

        # Artefactos de preprocesamiento y modelos serializados
        artifacts_dir = artifacts["artifacts_dir"]
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")

        # Visualizaciones generadas por los notebooks
        viz_dir = PROJECT_ROOT / "ml" / "visualizations"
        if viz_dir.exists() and any(viz_dir.iterdir()):
            mlflow.log_artifacts(str(viz_dir), artifact_path="visualizations")

        # Registrar modelos con sus flavors nativos de MLflow (MLflow 3.x: usar name sin /)
        mlflow.xgboost.log_model(artifacts["xgb_model"], name="xgboost_tabular")
        mlflow.keras.log_model(artifacts["nn_model"], name="nn_nlp_distilbert")
        mlflow.keras.log_model(artifacts["meta_model"], name="meta_learner_stacking")

        # Registrar el modelo híbrido final en el Model Registry
        model_uri = f"runs:/{run.info.run_id}/meta_learner_stacking"
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f"\n[MLflow] Run ID   : {run.info.run_id}")
        print(f"[MLflow] UI       : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return run.info.run_id


def main():
    print("=" * 60)
    print("  Sistema de Triaje Inteligente — Detección de Fraude")
    print("=" * 60)
    setup_mlflow()
    run_id = run_training()
    print(f"\n✅ Pipeline completado. Run ID: {run_id}")


if __name__ == "__main__":
    main()

