from models.RandomForest.predict_participation import predict_participation
from models.RandomForest.predict_place import predict_place_with_participation
import pandas as pd

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

        # Train participation
        predict_participation(
            data_path=BINARY_CSV,
            target_column=latest_col,
            output_dir="data"
        )

        # Train placement
        predict_place_with_participation(
            data_path=CLEANED_CSV,
            target_column=latest_col,
            output_dir="data"
        )

if __name__ == "__main__":
    train_all_events()