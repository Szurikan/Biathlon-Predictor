
import biathlonresults
import pandas as pd
import time
import os
from datetime import datetime

def update_data(output_path=None):

    if output_path is None:
        # issaugojimo kelias
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_dir, "data", "female_athletes_2425_full_stats_with_ranks.csv")
    
    print(f"[1] Pradedami rinkti duomenys iš IBU API...")
    
    # Sezonu ID
    season_ids = ["2122", "2223", "2324", "2425"]
    events = []
    
    print(f"[2] Gaunami renginiai iš {len(season_ids)} sezonų...")
    for season in season_ids:
        events += biathlonresults.events(season, level=biathlonresults.consts.LevelType.BMW_IBU_WC)

    race_ids = []
    race_columns = []
    race_participants = {}

    # Etapu skaicius sezone
    season_stage_counters = {season: 1 for season in season_ids}

    print(f"[3] Apdorojami {len(events)} renginiai...")
    for event in events:
        competitions = biathlonresults.competitions(event["EventId"])
        season = event["SeasonId"]
        stage_num = season_stage_counters[season]
        
        for comp in competitions:
            race_id = comp["RaceId"]
            try:
                results = biathlonresults.results(race_id)
                comp_info = results["Competition"]
                location = comp_info.get("Location", "Unknown")
                date_str = comp_info.get("StartTime", "")
                discipline_raw = comp_info.get("Description", "").strip()

                if any(keyword in discipline_raw.lower() for keyword in ["relay", "team", "mixed"]):
                    continue  # praleidžiam komandines

                # Nustatoma lytis
                if "Women" in discipline_raw:
                    gender_suffix = "W"
                elif "Men" in discipline_raw:
                    gender_suffix = "M"
                else:
                    gender_suffix = ""

                # Pavadinimo koregavimas
                discipline = (
                    discipline_raw.replace("km", "")
                    .replace("Men", "")
                    .replace("Women", "")
                    .strip()
                    .title()
                )

                # Gaunam data 
                race_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                year = race_date.year

                # Formatuojame pavadinimą: "2025 03 (Sprint) W"
                date_prefix = race_date.strftime("%Y-%m-%d")
                formatted_name = f"{date_prefix} {stage_num:02d} ({discipline}) {gender_suffix}"

                race_columns.append((race_id, formatted_name))
                race_ids.append(race_id)

                race_participants[race_id] = {
                    r["IBUId"]: r["Rank"] for r in results["Results"]
                }

            except Exception as e:
                print(f"Klaida su {race_id}: {e}")
            time.sleep(0.1)
        
        season_stage_counters[season] += 1  # Kitam etapui tame sezone

    athlete_ids = set()
    for race_data in race_participants.values():
        athlete_ids.update(race_data.keys())

    print(f"[4] Renkama informacija apie {len(athlete_ids)} sportininkes...")
    female_athletes = []

    def extract_stats(lst):
        return [lst[i] if len(lst) > i else "" for i in range(4)]

    for ibu_id in sorted(athlete_ids):
        try:
            bio = biathlonresults.cisbios(ibu_id)
            if bio.get("GenderId") == "W":
                full_name = bio.get("FullName", "")
                nat = bio.get("NAT", "")
                birth_year = bio.get("BirthYear", "")
                shooting = extract_stats(bio.get("StatShooting", []))
                prone = extract_stats(bio.get("StatShootingProne", []))
                standing = extract_stats(bio.get("StatShootingStanding", []))
                skiing = extract_stats(bio.get("StatSkiing", []))
                ski_kmb = extract_stats(bio.get("StatSkiKMB", []))
                starts = extract_stats(bio.get("StatStarts", []) or [])
                ranks = []
                for race_id, _ in race_columns:
                    rank = race_participants.get(race_id, {}).get(ibu_id, None)
                    ranks.append(rank)
                female_athletes.append((
                    ibu_id, full_name, nat, birth_year,
                    *shooting, *prone, *standing, *skiing, *ski_kmb, *starts,
                    *ranks
                ))
        except Exception as e:
            print(f"Klaida su {ibu_id}: {e}")
        time.sleep(0.2)

    columns = (
        ["IBUId", "FullName", "Nation", "BirthYear"] +
        [f"StatShooting_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [f"Prone_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [f"Standing_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [f"Skiing_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [f"SkiKMB_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [f"Starts_{y}" for y in ["24_25", "23_24", "22_23", "21_22"]] +
        [col_name for _, col_name in race_columns]
    )

    df = pd.DataFrame(female_athletes, columns=columns)
    df = df.sort_values("FullName")
    
    # Katalogu kurimas po klaidos
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[5] Išsaugome {len(df)} sportininkių duomenis į {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Duomenys sėkmingai atnaujinti!")
    
    return df

if __name__ == "__main__":
    update_data()