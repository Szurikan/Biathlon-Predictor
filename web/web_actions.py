import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
# Remove caching/persistence to ensure fresh training
from operations.data.loader import load_data
from config.config import TOP_PREDICTIONS_COUNT
import joblib

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
    Prognozuoja ateities rezultatus pagal modelį ir etapo tipą.

    :param event_type: "Sprint", "Pursuit", "Individual", arba "Mass Start"
    :param model_name: "random_forest" | "xgboost" | "lstm"
    :returns: list of dict su laukais rank, name, nation, predicted
    """
    try:
        df = load_data(DATA_FILE)

        # Convert percent strings to numeric
        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].dropna().astype(str)
                if not s.empty and s.str.endswith('%').all():
                    df[col] = s.str.rstrip('%').astype(float)

        # Identify nation one-hot columns
        nation_cols = [c for c in df.columns if c.startswith('Nation_')]

        # Identify same-type historic events
        event_cols = [c for c in df.columns if c.startswith('202') and event_type.lower() in c.lower()]
        races = sorted(event_cols, key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))

        if model_name == 'random_forest':
            # Pasiruošiam duomenis
            if len(races) < 1:
                return []
            event_map = {
                "Sprint": "Sprint",
                "Pursuit": "Pursuit",
                "Individual": "Individual",
                "Mass Start": "MassStart"
            }
            event_key = event_map.get(event_type, "Unknown")
            model_filename = f"{event_key}_RandomForest_Next.pkl"

            model_path = os.path.join("data", model_filename)

            model, columns = load_existing_model(model_path)
            if model is None:
                print(f"❌ Modelio failas nerastas: {model_path}")
                return []

            # Paimti X_pred pagal įkeltus stulpelius
            X_pred = df[columns].fillna(0)

            preds = model.predict(X_pred)
            df['predicted_time'] = preds

            # Sort and pick top
            df_sorted = df.sort_values('predicted_time')
            top = df_sorted.head(TOP_PREDICTIONS_COUNT)

            # Build results with nation detection
            results = []
            for idx, (_, row) in enumerate(top.iterrows()):
                nation = None
                for col in nation_cols:
                    if row.get(col, 0) == 1:
                        nation = col.split('_', 1)[1]
                        break
                results.append({
                    'rank': idx + 1,
                    'name': row['FullName'],
                    'nation': nation,
                    'predicted': f"{idx + 1} vieta"
                })
            return results

        elif model_name in ('xgboost', 'lstm'):
            # Simple stub: sort by last event time
            if not races:
                return []
            last_col = races[-1]
            clean = df.dropna(subset=[last_col]).sort_values(last_col)
            results = []
            for idx, (_, row) in enumerate(clean.head(TOP_PREDICTIONS_COUNT).iterrows()):
                nation = None
                for col in nation_cols:
                    if row.get(col, 0) == 1:
                        nation = col.split('_', 1)[1]
                        break
                results.append({
                    'rank': idx + 1,
                    'name': row['FullName'],
                    'nation': nation,
                    'predicted': f"{idx + 1} vieta"
                })
            return results

        else:
            # Unsupported model
            return []

    except Exception as e:
        print(f"Klaida prognozuojant etapą: {e}")
        return []
