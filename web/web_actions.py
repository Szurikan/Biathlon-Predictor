import os
from operations.data.loader import load_data
from config.config import TOP_PREDICTIONS_COUNT

DATA_FILE = os.path.join('data', 'female_athletes_2425_full_stats_with_ranks.csv')

def get_past_events():
    """Gražina įvykusių etapų sąrašą iš duomenų failo."""
    try:
        # Naudojame load_data vietoj tiesioginio skaitymo
        df = load_data(DATA_FILE)
        
        # Filtruojame stulpelius, kurie yra varžybų rezultatai (formato "YYYY MM (Discipline) W")
        race_cols = [col for col in df.columns if "(" in col and ")" in col]
        # Rūšiuojame chronologiškai
        sorted_race_cols = sorted(race_cols, key=lambda x: (int(x[:4]), int(x[5:7])))
        return sorted_race_cols
    except Exception as e:
        print(f"Klaida gaunant praėjusius etapus: {str(e)}")
        return []
    
def predict_next_event(event, model_name):
    """
    Prognozuoja konkretaus etapo rezultatus pagal modelį.
    """
    try:
        df = load_data(DATA_FILE)

        names = df["FullName"]
        nations = df["Nation"]

        # Tiesiog imitacija – čia turėtų būti realus modelio naudojimas
        sorted_df = df.sort_values(by=event).dropna(subset=[event])
        top_athletes = []
        for i, (_, row) in enumerate(sorted_df.head(TOP_PREDICTIONS_COUNT).iterrows()):
            top_athletes.append({
                "rank": i + 1,
                "name": row["FullName"],
                "nation": row["Nation"],
                "predicted": f"{i+1} vieta"
            })

        return top_athletes
    except Exception as e:
        print(f"Klaida prognozuojant etapą: {str(e)}")
        return []