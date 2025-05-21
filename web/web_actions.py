import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
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
    try:
        df = load_data(DATA_FILE)
        race_cols = [col for col in df.columns if '(' in col and ')' in col]
        return sorted(race_cols, key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    except Exception as e:
        print(f"Klaida gaunant praėjusius etapus: {e}")
        return []


# def _train_rf(event_type, df, races, static_feats):

#     X_list, y_list = [], []
#     for i in range(len(races) - 1):
#         prev_col, next_col = races[i], races[i+1]
#         X = df[static_feats + [prev_col]].fillna(0).rename(columns={prev_col: 'last_time'})
#         y = pd.to_numeric(df[next_col], errors='coerce')
#         mask = y.notna()
#         X_list.append(X[mask])
#         y_list.append(y[mask])

#     if not X_list:
#         return None

    # X_train = pd.concat(X_list, ignore_index=True)
    # y_train = pd.concat(y_list, ignore_index=True)

    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    # return rf

def load_existing_model(model_path):
    if os.path.exists(model_path):
        model, columns = joblib.load(model_path)
        return model, columns
    return None, None


def predict_next_event(event_type, model_name):

    try:
        df = load_data(DATA_FILE)

        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].dropna().astype(str)
                if not s.empty and s.str.endswith('%').all():
                    df[col] = s.str.rstrip('%').astype(float)

        nation_cols = [c for c in df.columns if c.startswith('Nation_')]

        # Atsirenkame tinkamo tipo etapus
        event_cols = [
            c for c in df.columns
            if c.startswith('202') and event_type.lower() in c.lower()
        ]
        races = sorted(
            event_cols,
            key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d")
        )
        if not races:
            return []

        # Dalyvavimo modelio kelias
        part_model_path = os.path.join("data", "next_event_Participation_RandomForest.pkl")
        if not os.path.exists(part_model_path):
            print(f"❌ Dalyvavimo modelio failas nerastas: {part_model_path}")
            return []

        # Įkeliame dalyvavimo modelį
        part_model, part_columns = joblib.load(part_model_path)
        X_part = df[part_columns].fillna(0)
        part_raw = part_model.predict(X_part)

        # Konvertuojame prognozes į 0/1 (naudojame esamą logiką)
        from models.RandomForest.predict_participation import adjust_predictions_by_format
        binary_preds = adjust_predictions_by_format(part_raw, event_type)
        df = df[binary_preds == 1]  # Paliekame tik tuos, kurie dalyvauja

        # Paruošiame duomenis pagal pasirinkto modelio tipą
        event_map = {
            "Sprint": "Sprint",
            "Pursuit": "Pursuit",
            "Individual": "Individual",
            "Mass Start": "MassStart"
        }
        event_key = event_map.get(event_type, "Unknown")

        # ________ RANDOM FOREST ________
        if model_name == 'random_forest':
            model_filename = f"{event_key}_RandomForest_Next.pkl"
            model_path = os.path.join("data", model_filename)
            model, columns = load_existing_model(model_path)
            if model is None:
                return []
            X_pred = df[columns].fillna(0)
            preds = model.predict(X_pred)

        # ________ XGBOOST ________
        elif model_name == 'xgboost':
            model_filename = f"{event_key}_XGBoost_Next.pkl"
            model_path = os.path.join("data", model_filename)
            model, columns = load_existing_model(model_path)
            if model is None:
                return []
            X_pred = df[columns].fillna(0)
            preds = model.predict(X_pred)

        # ________ LSTM ________
        elif model_name == 'lstm':
            fn = f"{event_key}_LSTM_Next.keras"
            mpath = os.path.join("data", fn)
            if not os.path.exists(mpath):
                print(f"❌ LSTM modelio failas nerastas: {mpath}")
                return []
            model = load_model(mpath, compile=False)

            from sklearn.preprocessing import StandardScaler

            static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]
            X_static = df[static_feats].fillna(0).values

            train_date = "2024-12-22"
            train_cols = [c for c in races if datetime.strptime(c.split()[0], "%Y-%m-%d") <= datetime.strptime(train_date, "%Y-%m-%d")]
            X_seq = df[train_cols].fillna(0).values

            scaler_static = StandardScaler()
            scaler_seq = StandardScaler()
            X_static_norm = scaler_static.fit_transform(X_static)
            X_seq_norm = scaler_seq.fit_transform(X_seq)
        

            X_combined = np.hstack((X_static_norm, X_seq_norm)).reshape((len(df), -1, 1))

            preds = model.predict(X_combined).flatten()

        else:
            return []


        df['predicted_time'] = preds
        df_sorted = df.sort_values('predicted_time')
        top = df_sorted.head(10)

        results = []
        for idx, (_, row) in enumerate(top.iterrows()):
            nation = next(
                (col.split('_', 1)[1] for col in nation_cols if row.get(col, 0) == 1),
                None
            )
            results.append({
                'rank': idx + 1,
                'name': row['FullName'],
                'nation': nation,
                'predicted': f"{idx + 1} vieta"
            })
        return results

    except Exception as e:
        print(f"Klaida prognozuojant etapą: {e}")
        return []