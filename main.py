# main.py

import pandas as pd
from operations.data.loader import load_data, clean_team_events, extract_identity_columns
from operations.data.preprocessing import (
    categorize_columns,
    fill_missing_values,
    convert_percentage_columns
)

def main():
    # 1. Įkeliame duomenis
    df = load_data()

    # 2. Šaliname komandines varžybas ir visiškai tuščius stulpelius
    df = clean_team_events(df)

    # 3. Ištraukiame identifikacinius stulpelius
    df, id_df = extract_identity_columns(df)

    # 4. Konvertuojame procentus į skaitines reikšmes
    df = convert_percentage_columns(df)

    # 5. Užpildome trūkstamas reikšmes
    df = fill_missing_values(df)

    # 6. Šaliname visiškai tuščias eilutes
    df.dropna(how='all', inplace=True)

    # 7. Suskirstome stulpelius (jei reikia tolesniam naudojimui)
    categorize_columns(df)

    # 8. Sujungiame su ID informacija
    df_final = pd.concat([id_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # 9. Išsaugome į CSV
    df_final.to_csv("cleaned_data.csv", index=False)
    print("[OK] Duomenys išsaugoti į 'cleaned_data.csv'")

if __name__ == "__main__":
    main()
