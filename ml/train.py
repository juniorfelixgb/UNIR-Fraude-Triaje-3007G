import random
import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoModel, AutoTokenizer

DISTILBERT_MODEL = "distilbert-base-multilingual-cased"


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_and_engineer_features(data_path: Path):
    df = pd.read_excel(data_path)

    date_cols = ["Incident_Date", "Date_Reported", "Policy_Start_Date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Report_Delay"] = (df["Date_Reported"] - df["Incident_Date"]).dt.days.fillna(0)
    df["Days_Since_Policy"] = (df["Incident_Date"] - df["Policy_Start_Date"]).dt.days.fillna(0)
    df["Car_Age"] = df["Incident_Date"].dt.year - df["Model_Year"]

    df = df.select_dtypes(exclude=["datetime64"])
    df = df.drop(
        columns=["Claim_ID", "Policy_Number", "Last_Purchase_History_Date",
                 "Policy_Renewal_Date", "Insured_Inception_Date", "Model_Year"],
        errors="ignore",
    )

    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "Claim_Description":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    X_tab_df = df.drop(columns=["Claim_Description", "Prediccion_Fraude"])
    return (
        X_tab_df.values,
        df["Prediccion_Fraude"].values,
        df["Claim_Description"].astype(str).tolist(),
        X_tab_df.columns.tolist(),
        label_encoders,
    )


def extract_embeddings(text_list, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=120, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)


def _split(X_tab, X_text, y, seed=42):
    idx = np.arange(len(y))
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        idx, y, test_size=0.20, random_state=seed, stratify=y
    )
    idx_base, idx_meta, y_base, y_meta = train_test_split(
        idx_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_base, y_base, test_size=0.20, random_state=seed, stratify=y_base
    )
    return (
        X_tab[idx_train], X_tab[idx_val], X_tab[idx_meta], X_tab[idx_test],
        X_text[idx_train], X_text[idx_val], X_text[idx_meta], X_text[idx_test],
        y_train, y_val, y_meta, y_test,
    )


def _train_xgboost(X_train, X_val, y_train, y_val, params: dict):
    model = xgb.XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        learning_rate=params["xgb_learning_rate"],
        max_depth=params["xgb_max_depth"],
        scale_pos_weight=float(np.sum(y_train == 0) / np.sum(y_train == 1)),
        eval_metric=["logloss", "error"],
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def _train_nn_nlp(X_train, X_val, y_train, y_val, params: dict):
    inp = Input(shape=(768,))
    x = Dense(128, activation="relu")(inp)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    cw = dict(enumerate(
        class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    ))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["nn_epochs"],
        batch_size=params["nn_batch_size"],
        class_weight=cw,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=params["nn_patience"],
            restore_best_weights=True,
        )],
        verbose=0,
    )
    return model


def _train_meta(xgb_model, nn_model, X_tab_meta, X_text_meta, y_meta, params: dict):
    p1 = xgb_model.predict_proba(X_tab_meta)[:, 1]
    p2 = nn_model.predict(X_text_meta, verbose=0).flatten()

    inp = Input(shape=(2,))
    x = Dense(8, activation="relu")(inp)
    out = Dense(1, activation="sigmoid")(x)
    meta = Model(inputs=inp, outputs=out)
    meta.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    meta.fit(
        np.column_stack((p1, p2)), y_meta,
        validation_split=0.2,
        epochs=params["meta_epochs"],
        verbose=0,
    )
    return meta


def run_pipeline(project_root: Path, params: dict) -> tuple[dict, dict]:
    data_path = project_root / "ml" / "data" / "dataset_reclamos_ia_ruidoso_extremo.xlsx"
    artifacts_dir = project_root / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(params["seed"])

    print("[1/6] Cargando datos y feature engineering...")
    X_tab, y, text_data, X_tab_features, label_encoders = load_and_engineer_features(data_path)

    print("[2/6] Cargando DistilBERT y extrayendo embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    distilbert = AutoModel.from_pretrained(DISTILBERT_MODEL)
    distilbert.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distilbert.to(device)
    X_text = extract_embeddings(text_data, tokenizer, distilbert, device)

    print("[3/6] Dividiendo datos (train/val/meta/test)...")
    (X_tab_train, X_tab_val, X_tab_meta, X_tab_test,
     X_text_train, X_text_val, X_text_meta, X_text_test,
     y_train, y_val, y_meta, y_test) = _split(X_tab, X_text, y, params["seed"])

    print("[4/6] Entrenando XGBoost...")
    xgb_model = _train_xgboost(X_tab_train, X_tab_val, y_train, y_val, params)

    print("[5/6] Entrenando NN NLP (DistilBERT embeddings)...")
    nn_model = _train_nn_nlp(X_text_train, X_text_val, y_train, y_val, params)

    print("[6/6] Entrenando Meta-Learner (Stacking)...")
    meta_model = _train_meta(xgb_model, nn_model, X_tab_meta, X_text_meta, y_meta, params)

    # Métricas
    prob_xgb = xgb_model.predict_proba(X_tab_test)[:, 1]
    prob_nn = nn_model.predict(X_text_test, verbose=0).flatten()
    prob_stack = meta_model.predict(np.column_stack((prob_xgb, prob_nn)), verbose=0).flatten()

    def _metrics(name, y_true, y_prob):
        y_pred = (y_prob > 0.5).astype(int)
        return {
            f"{name}_f1": float(f1_score(y_true, y_pred)),
            f"{name}_auc": float(roc_auc_score(y_true, y_prob)),
        }

    metrics = {}
    metrics.update(_metrics("xgboost", y_test, prob_xgb))
    metrics.update(_metrics("nn_nlp", y_test, prob_nn))
    metrics.update(_metrics("hybrid_stacking", y_test, prob_stack))

    # Guardar artefactos locales
    xgb_model.save_model(str(artifacts_dir / "xgb_model.json"))
    nn_model.save(str(artifacts_dir / "nn_text_model.keras"))
    meta_model.save(str(artifacts_dir / "meta_nn.keras"))
    joblib.dump(label_encoders, artifacts_dir / "label_encoders.pkl")
    joblib.dump(X_tab_features, artifacts_dir / "feature_names.pkl")

    return metrics, {
        "xgb_model": xgb_model,
        "nn_model": nn_model,
        "meta_model": meta_model,
        "artifacts_dir": artifacts_dir,
    }
