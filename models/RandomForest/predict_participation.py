import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
from datetime import datetime

def predict_participation(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)

    # Išrenkame varžybų stulpelius, pradedančius su data (YYYY-MM-DD)
    competition_columns = [col for col in df.columns if col.startswith("202")]
    
    if target_column not in df.columns:
        raise ValueError(f"Nurodytas stulpelis '{target_column}' nerastas faile.")

    # Surikiuojame stulpelius chronologiškai
    competition_columns_sorted = sorted(
        competition_columns, 
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    
    # Tikslo stulpelio indeksas sąraše
    target_index = competition_columns_sorted.index(target_column)
    
    # Tik ankstesnių varžybų stulpeliai
    past_columns = competition_columns_sorted[:target_index]
    
    # Saugiklis: tikriname, ar tikslo stulpelis tikrai nepatenka į požymius
    assert target_column not in past_columns, "Klaida: tikslo stulpelis pateko į požymius!"
    
    # Statiniai požymiai (visi ne datos stulpeliai, išskyrus ID ir vardą)
    static_features = [col for col in df.columns if not col.startswith("202") and col not in ["IBUId", "FullName"]]

    # Atrenkame tik eilutes su žinomais tikslo duomenimis
    df_model = df.dropna(subset=[target_column])
    
    # Požymiai ir tikslas
    X = df_model[static_features + past_columns]
    y = df_model[target_column].astype(int)
    
    # Dar vienas saugiklis: tikslo stulpelis neturi būti tarp požymių
    assert target_column not in X.columns, "Klaida: tikslo stulpelis yra tarp požymių!"

    # Duomenų padalijimas mokymuisi ir testavimui
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelio kūrimas ir mokymas
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Modelio išsaugojimas (su požymių sąrašu)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    # Išsaugome modelį kartu su požymių sąrašu
    joblib.dump((model, list(X.columns)), model_path)
    print(f"Modelis išsaugotas: {model_path}")

    # Modelio įvertinimas
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    print(f"\nModelio rezultatai prognozuojant '{target_column}':\n")
    print(report)

    # Prognozė visoms sportininkėms
    df_all = df.copy()
    df_all_features = df_all[static_features + past_columns].copy()
    df_all_features = df_all_features.fillna(0)
    
    # Dar vienas saugiklis prieš prognozavimą
    assert target_column not in df_all_features.columns, "Klaida: tikslo stulpelis yra prognozavimo požymiuose!"
    
    df_all["PredictedParticipation"] = model.predict(df_all_features)

    # Išsaugoti prognozių failą
    output_csv = os.path.join(output_dir, f"predictions_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    df_all[["FullName", "PredictedParticipation"]].to_csv(output_csv, index=False)
    print(f"\nPrognozių failas sukurtas: {output_csv}")

    # Parodyti sportininkes, kurioms prognozuojamas dalyvavimas
    predicted_names = df_all[df_all["PredictedParticipation"] == 1]["FullName"]
    print("\nVISOS sportininkės, kurioms prognozuojamas DALYVAVIMAS šiame etape:")
    for name in predicted_names:
        print("-", name)

if __name__ == "__main__":
    predict_participation(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2024-12-04 01 (12.5  Short Individual) W"
    )