import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from operations.data.loader import load_data
from config.config import TOP_PREDICTIONS_COUNT
import joblib
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore

# Paths
DATA_FILE = "data/athletes_data.db"
TABLE_NAME = "cleaned_data"

# No persistence directory used: always train fresh

def get_past_events():
    """Gražina įvykusių etapų sąrašą chronologiškai."""
    try:
        df = load_data(DATA_FILE)
        race_cols = [col for col in df.columns if '(' in col and ')' in col]
        return sorted(race_cols, key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    except Exception as e:
        print(f"Klaida gaunant praėjusius etapus: {e}")
        return []

def _train_rf(event_type, df, races, static_feats):
    """Train RF on pairs of historic same-type event results and return the model."""
    X_list, y_list = [], []
    for i in range(len(races) - 1):
        prev_col, next_col = races[i], races[i+1]
        X = df[static_feats + [prev_col]].fillna(0).rename(columns={prev_col: 'last_time'})
        y = pd.to_numeric(df[next_col], errors='coerce')
        mask = y.notna()
        X_list.append(X[mask])
        y_list.append(y[mask])

    if not X_list:
        return None

    X_train = pd.concat(X_list, ignore_index=True)
    y_train = pd.concat(y_list, ignore_index=True)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def load_existing_model(model_path):
    if os.path.exists(model_path):
        model, columns = joblib.load(model_path)
        return model, columns
    return None, None

def predict_next_event(event_type, model_name):
    from models.RandomForest.predict_participation import adjust_predictions_by_format
    try:
        df = load_data(DATA_FILE)

        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].dropna().astype(str)
                if not s.empty and s.str.endswith('%').all():
                    df[col] = s.str.rstrip('%').astype(float)

        nation_cols = [c for c in df.columns if c.startswith('Nation_')]
        event_cols = [c for c in df.columns if c.startswith("202") and event_type.lower() in c.lower()]
        races = sorted(event_cols, key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
        if not races:
            return []

        part_model_map = {
            "random_forest": "next_event_Participation_RandomForest.pkl",
            "xgboost": "next_event_Participation_XGBoost.pkl",
            "lstm": "next_event_Participation_LSTM_Next.keras"
        }
        part_model_path = os.path.join("data", part_model_map[model_name])

        if model_name == "lstm":
            if not os.path.exists(part_model_path):
                print(f"❌ Dalyvavimo LSTM modelio failas nerastas: {part_model_path}")
                return []
            model = load_model(part_model_path, compile=False)
            static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]
            X_static = df[static_feats].fillna(0).values
            X_seq = df[races].fillna(0).values
            X_static = StandardScaler().fit_transform(X_static)
            X_seq = StandardScaler().fit_transform(X_seq)
            X_combined = np.hstack((X_static, X_seq)).reshape((len(df), -1, 1))
            raw = model.predict(X_combined).flatten()
        else:
            if not os.path.exists(part_model_path):
                print(f"❌ Dalyvavimo modelio failas nerastas: {part_model_path}")
                return []
            part_model, part_columns = joblib.load(part_model_path)
            X_part = df[part_columns].fillna(0)
            raw = part_model.predict(X_part)

        binary_mask = adjust_predictions_by_format(raw, event_type)
        df_part = df[binary_mask == 1]

        if df_part.empty:
            print("⚠️ Nėra numatomų dalyvių.")
            return []

        place_model_map = {
            "random_forest": "next_event_Place_RandomForest.pkl",
            "xgboost": "next_event_Place_XGBoost.pkl",
            "lstm": "next_event_Place_LSTM.keras"
        }
        place_model_path = os.path.join("data", place_model_map[model_name])

        if model_name == "lstm":
            if not os.path.exists(place_model_path):
                print(f"❌ Vietos LSTM modelis nerastas: {place_model_path}")
                return []
            model = load_model(place_model_path, compile=False)
            static_feats = [c for c in df_part.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]
            static_data = df_part[static_feats].fillna(0).values
            static_scaled = StandardScaler().fit_transform(static_data)
            seq_length = min(10, len(races))
            seq_cols = races[-seq_length:]
            seq_data = df_part[seq_cols].fillna(0).values
            seq_scaled = StandardScaler().fit_transform(seq_data)
            n_samples = len(df_part)
            n_features = static_scaled.shape[1] + 1
            X_combined = np.zeros((n_samples, seq_length, n_features))
            for t in range(seq_length):
                X_combined[:, t, :-1] = static_scaled
                X_combined[:, t, -1] = seq_scaled[:, t]
            preds = model.predict(X_combined).flatten()
        else:
            if not os.path.exists(place_model_path):
                print(f"❌ Vietos modelis nerastas: {place_model_path}")
                return []
            model, columns = joblib.load(place_model_path)
            X_place = df_part[columns].fillna(0)
            preds = model.predict(X_place)

        df_part = df_part.copy()
        df_part['predicted_time'] = preds
        df_sorted = df_part.sort_values('predicted_time').head(10)

        results = []
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            nation = next((col.split('_', 1)[1] for col in nation_cols if row.get(col, 0) == 1), None)
            results.append({
                'rank': idx,
                'name': row['FullName'],
                'nation': nation,
                'predicted': f"{idx} vieta"
            })

        return results

    except Exception as e:
        print(f"❌ Klaida prognozuojant etapą: {e}")
        return []