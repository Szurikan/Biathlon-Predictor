import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def predict_participation(data_path, target_column, output_dir="models/RandomForest"):
    df = pd.read_csv(data_path)

    # Tik varžybų stulpeliai (kurie prasideda 202*)
    competition_columns = [col for col in df.columns if col.startswith("202")]

    if target_column not in df.columns:
        raise ValueError(f"Nurodytas stulpelis '{target_column}' nerastas faile.")

    # Naudojami tik varžybų stulpeliai iki prognozuojamo
    past_columns = [col for col in competition_columns if col < target_column]

    # Kiti statiniai bruožai (amžius, tautybės kodai ir kt.)
    static_features = [col for col in df.columns if not col.startswith("202") and col not in ["IBUId", "FullName"]]

    # Tik tos eilutės, kur yra žinoma, ar sportininkė dalyvavo tame etape
    df_model = df.dropna(subset=[target_column])

    # Požymiai ir tikslas
    X = df_model[static_features + past_columns]
    y = df_model[target_column].astype(int)

    # Duomenų dalijimas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelio treniravimas
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prognozės ir ataskaita
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Rezultatų spausdinimas
    print(f"Modelio rezultatai prognozuojant '{target_column}':\n")
    print(report)

# Pavyzdinis iškvietimas (galima ištrinti jei integruosi kitur)
if __name__ == "__main__":
    predict_participation(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025 07 (7.5  Sprint Competition) W"
    )
