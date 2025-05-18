import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
# Remove caching/persistence to ensure fresh training
from operations.data.loader import load_data
from config.config import TOP_PREDICTIONS_COUNT
import joblib
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore

# Paths
DATA_FILE = os.path.join('data', 'female_athletes_cleaned_final.csv')

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
    """
    Predicts the next event ranking using RF, XGBoost or LSTM,
    but only for athletes predicted to participate.
    """
    try:
        df = load_data(DATA_FILE)

        # Konvertuojame „%“ į skaičius (jei reikia)
        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].dropna().astype(str)
                if not s.empty and s.str.endswith('%').all():
                    df[col] = s.str.rstrip('%').astype(float)

        # Viena karšta kodavimas šaliai
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
            fn = f"{event_key}_LSTM_Next.h5"
            mpath = os.path.join("data", fn)
            if not os.path.exists(mpath):
                print(f"❌ LSTM modelio failas nerastas: {mpath}")
                return []
            model = load_model(mpath, compile=False)

            train_date = "2024-12-22"
            train_cols = [
                c for c in races
                if datetime.strptime(c.split()[0], "%Y-%m-%d")
                   <= datetime.strptime(train_date, "%Y-%m-%d")
            ]
            static_feats = [
                c for c in df.columns
                if not c.startswith("202") and c not in ["IBUId", "FullName"]
            ]
            X_seq = df[train_cols].fillna(0).to_numpy()
            X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
            X_static = df[static_feats].fillna(0).to_numpy()
            preds = model.predict([X_seq, X_static]).flatten()

        else:
            return []

        # Sudedame prognozes ir rūšiuojame
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