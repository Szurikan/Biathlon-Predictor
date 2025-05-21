import pandas as pd
import sqlite3

# Random Forest
from models.RandomForest.predict_participation import predict_participation as rf_participation
from models.RandomForest.predict_place import predict_place_with_participation as rf_place

# XGBoost
from models.XGBoostClassifier.predict_participation import predict_participation as xgb_participation
from models.XGBoostClassifier.predict_place import predict_place_with_participation as xgb_place

# LSTM
from models.NNM.predict_participation import predict_participation_lstm as lstm_participation
from models.NNM.predict_place import predict_place_with_participation as lstm_place

# Keliai
BINARY_DB = "data/athletes_data.db"
CLEANED_DB = "data/athletes_data.db"
EVENT_TYPES = ["Mass Start"] 
 #              "Individual", 
  #             "Pursuit", 
   #            "Sprint"]

def train_all_events():
    # df = pd.read_csv(BINARY_DB)

    conn = sqlite3.connect(BINARY_DB)
    df = pd.read_sql_query("SELECT * FROM binary_data", conn)
    conn.close()
    competition_columns = [col for col in df.columns if col.startswith("202")]

    for event_type in EVENT_TYPES:
        print(f"\n==================== {event_type.upper()} ====================")
        matching_cols = [col for col in competition_columns if event_type in col]

        if not matching_cols:
            print(f"⚠️ Nerasta jokių {event_type} etapų duomenų.")
            continue

        latest_col = sorted(matching_cols, key=lambda x: x.split()[0])[-1]
        print(f"📌 Pasirinktas etapas: {latest_col}")

        # -------------------------
        # RANDOM FOREST
        # -------------------------
        print("Random Forest:")
        print("Dalyvavimo prognozė...")
        rf_participation(
            data_path=BINARY_DB,
            target_column=latest_col,
            output_dir="data"
        )
        print("Vietos prognozė...")
        rf_place(
            data_path=CLEANED_DB,
            target_column=latest_col,
            output_dir="data"
        )

        # -------------------------
        # XGBOOST
        # -------------------------
        print("XGBoost:")
        print("Dalyvavimo prognozė...")
        xgb_participation(
            data_path=BINARY_DB,
            target_column=latest_col,
            output_dir="data"
        )
        print("Vietos prognozė...")
        xgb_place(
            data_path=CLEANED_DB,
            target_column=latest_col,
            output_dir="data"
        )

        # -------------------------
        # LSTM
        # -------------------------
        print("LSTM:")
        print("Dalyvavimo prognozė...")
        lstm_participation(
            data_path=BINARY_DB,
            target_column=latest_col,
            output_dir="data"
        )
        print("Vietos prognozė...")
        lstm_place(
            data_path=CLEANED_DB,
            target_column=latest_col,
            output_dir="data"
        )

def train_model(model_name, task_type):
    conn = sqlite3.connect(BINARY_DB)
    df = pd.read_sql_query("SELECT * FROM binary_data", conn)
    conn.close()

    competition_columns = [col for col in df.columns if col.startswith("202")]

    for event_type in EVENT_TYPES:
        matching_cols = [col for col in competition_columns if event_type in col]
        if not matching_cols:
            print(f"⚠️ Nerasta {event_type} etapų.")
            continue

        latest_col = sorted(matching_cols, key=lambda x: x.split()[0])[-1]
        print(f"\n📌 Etapas: {latest_col} ({event_type})")

        if model_name == "RandomForest":
            if task_type in ["participation", "both"]:
                rf_participation(BINARY_DB, latest_col, output_dir="data")
            if task_type in ["place", "both"]:
                rf_place(CLEANED_DB, latest_col, output_dir="data")

        elif model_name == "XGBoost":
            if task_type in ["participation", "both"]:
                xgb_participation(BINARY_DB, latest_col, output_dir="data")
            if task_type in ["place", "both"]:
                xgb_place(CLEANED_DB, latest_col, output_dir="data")

        elif model_name == "LSTM":
            if task_type in ["participation", "both"]:
                lstm_participation(BINARY_DB, latest_col, output_dir="data")
            if task_type in ["place", "both"]:
                lstm_place(CLEANED_DB, latest_col, output_dir="data")

if __name__ == "__main__":
    train_all_events()
