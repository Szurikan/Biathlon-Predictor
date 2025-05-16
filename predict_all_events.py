import pandas as pd

# Random Forest
from models.RandomForest.predict_participation import predict_participation as rf_participation
from models.RandomForest.predict_place import predict_place_with_participation as rf_place

# XGBoost
from models.XGBoostClassifier.predict_participation import predict_participation_xgb as xgb_participation
from models.XGBoostClassifier.predict_place import predict_place_with_participation as xgb_place

# LSTM
from models.NNM.predict_participation import predict_participation_lstm as lstm_participation
from models.NNM.predict_place import predict_place_with_participation as lstm_place

# Keliai
BINARY_CSV = "data/female_athletes_binary_competitions.csv"
CLEANED_CSV = "data/female_athletes_cleaned_final.csv"
EVENT_TYPES = ["Mass Start", "Individual", "Pursuit", "Sprint"]

def train_all_events():
    df = pd.read_csv(BINARY_CSV)
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
        print("🌲 Random Forest:")
        print("🔄 Dalyvavimo prognozė...")
        rf_participation(
            data_path=BINARY_CSV,
            target_column=latest_col,
            output_dir="data"
        )
        print("🎯 Vietos prognozė...")
        rf_place(
            data_path=CLEANED_CSV,
            target_column=latest_col,
            output_dir="data"
        )

        # -------------------------
        # XGBOOST
        # -------------------------
        print("⚡ XGBoost:")
        print("🔄 Dalyvavimo prognozė...")
        xgb_participation(
            data_path=BINARY_CSV,
            target_column=latest_col,
            output_dir="data"
        )
        print("🎯 Vietos prognozė...")
        xgb_place(
            data_path=CLEANED_CSV,
            target_column=latest_col,
            output_dir="data"
        )

        # -------------------------
        # LSTM
        # -------------------------
        print("🧠 LSTM:")
        print("🔄 Dalyvavimo prognozė...")
        lstm_participation(
            data_path=BINARY_CSV,
            target_column=latest_col,
            output_dir="data"
        )
        print("🎯 Vietos prognozė...")
        lstm_place(
            data_path=CLEANED_CSV,
            target_column=latest_col,
            output_dir="data"
        )

if __name__ == "__main__":
    train_all_events()
