import os
from operations.loader import load_data

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