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
    
def predict_next_event(race_type):
    """
    Prognozuoja būsimo etapo rezultatus pagal nurodytą rungties tipą.
    
    Args:
        race_type (str): Rungties tipas (Sprint, Pursuit, Mass Start, Individual)
        
    Returns:
        list: TOP sportininkių prognozės
    """
    try:
        # Įkeliame duomenis
        df = load_data(DATA_FILE)
        
        # Gauname sportininkių indeksus, kurios greičiausiai dalyvaus
        # Paprastam demonstravimui, imame visas sportininkes
        likely_participants = list(range(len(df)))
        
        # Sukuriame sintetinius taikinius apmokyti modelį
        # Čia reikėtų pritaikyti pagal jūsų konkrečius poreikius
        race_cols = [col for col in df.columns if "(" in col and ")" in col]
        race_cols = [col for col in race_cols if race_type in col]
        
        if not race_cols:
            return []
        
        # Naudojame paskutiniuosius 3 tos rungties etapus
        last_races = sorted(race_cols)[-3:]
        
        # Paimame vardus ir šalis
        names = df["FullName"]
        nations = df["Nation"]
        
        # Prognozuojam rezultatus
        # Čia reikėtų pritaikyti pagal jūsų konkrečius poreikius
        # Tai yra supaprastinta versija
        top_athletes = []
        for i in range(min(TOP_PREDICTIONS_COUNT, len(likely_participants))):
            # Imituojame prognozavimą (realiame projekte čia būtų panaudotas modelis)
            top_athletes.append({
                "rank": i + 1,
                "name": names.iloc[likely_participants[i]],
                "nation": nations.iloc[likely_participants[i]],
                "predicted": f"{i+1} vieta"
            })
        
        return top_athletes
    except Exception as e:
        print(f"Klaida prognozuojant būsimą etapą: {str(e)}")
        return []