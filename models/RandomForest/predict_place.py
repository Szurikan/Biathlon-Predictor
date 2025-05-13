import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

def predict_place_with_participation(
    cleaned_data_path,
    binary_data_path,
    target_column,
    participation_model_path,
    output_dir="models/RandomForest"
):
    # 1. Įkeliame duomenis
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_binary = pd.read_csv(binary_data_path)

    # 2. Požymių atranka - surikiuojame chronologiškai
    competition_columns = [col for col in df_binary.columns if col.startswith("202")]
    competition_columns_sorted = sorted(
        competition_columns, 
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    
    # Tikslo stulpelio indeksas
    target_index = competition_columns_sorted.index(target_column)
    
    # Tik ankstesnių varžybų stulpeliai
    past_columns = competition_columns_sorted[:target_index]
    
    # Saugiklis: tikriname, ar tikslo stulpelis tikrai nepatenka į požymius
    assert target_column not in past_columns, "Klaida: tikslo stulpelis pateko į požymius!"

    # Statiniai požymiai
    static_features = [
        col for col in df_cleaned.columns
        if not col.startswith("202") and col not in ["IBUId", "FullName", target_column]
    ]

    feature_names = static_features + past_columns

    # 3. Įkeliame dalyvavimo modelį (modelis ir požymių sąrašas)
    loaded = joblib.load(participation_model_path)
    if isinstance(loaded, tuple):
        clf, clf_feature_names = loaded
    else:
        # Jei senas modelis be požymių sąrašo, naudojame dabartinius
        clf = loaded
        clf_feature_names = feature_names

    # Saugiklis: tikriname, ar tikslo stulpelis nėra tarp dalyvavimo modelio požymių
    assert target_column not in clf_feature_names, "Klaida: tikslo stulpelis yra tarp dalyvavimo modelio požymių!"

    # 4. Dalyvavimo prognozė
    df_binary_features = df_binary[clf_feature_names].fillna(0)
    df_cleaned["PredictedParticipation"] = clf.predict(df_binary_features)

    # 5. Regresoriaus treniravimas
    df_train = df_cleaned[df_cleaned[target_column].notna()].copy()
    X_train = df_train[feature_names].fillna(0)
    y_train = df_train[target_column].astype(float)

    # Saugikliai - target_column neturi būti požymis
    assert target_column not in feature_names, "Klaida: tikslo stulpelis yra tarp požymių!"
    assert target_column not in X_train.columns, "Klaida: tikslo stulpelis yra tarp treniravimo stulpelių!"

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Prognozė dalyvaujantiems
    df_predict = df_cleaned[df_cleaned["PredictedParticipation"] == 1].copy()
    X_predict = df_predict[feature_names].fillna(0)
    
    # Dar vienas saugiklis
    assert target_column not in X_predict.columns, "Klaida: tikslo stulpelis yra tarp prognozavimo stulpelių!"

    y_pred = model.predict(X_predict)
    df_predict["PredictedPlace"] = y_pred

    # 7. Tikra vieta arba "Nedalyvavo"
    df_predict["ActualPlace"] = df_predict[target_column].apply(
        lambda x: int(x) if pd.notna(x) else "Nedalyvavo"
    )

    df_predict_sorted = df_predict[["FullName", "PredictedPlace", "ActualPlace"]].sort_values("PredictedPlace").reset_index(drop=True)
    df_predict_sorted.insert(0, "PredictedRank", df_predict_sorted.index + 1)

    print(f"\nPrognozuojamų sportininkių sąrašas ({len(df_predict_sorted)} sportininkės):")
    print(df_predict_sorted.to_string(index=False))

    # 8. Išsaugome modelį su feature_names
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(
        output_dir,
        f"regression_place_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    )
    joblib.dump((model, feature_names), model_path)

    # 9. Išsaugome prognozes
    output_csv = os.path.join(
        output_dir,
        f"predicted_places_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    )
    df_predict_sorted.to_csv(output_csv, index=False)

    print(f"\nRegresijos modelis išsaugotas: {model_path}")
    print(f"Prognozės išsaugotos: {output_csv}")

# Naudojimas
if __name__ == "__main__":
    predict_place_with_participation(
        cleaned_data_path="data/female_athletes_cleaned_final.csv",
        binary_data_path="data/female_athletes_binary_competitions.csv",
        target_column="2024-12-04 01 (12.5  Short Individual) W",
        participation_model_path="data/model_2024-12-04_01_12.5__Short_Individual_W.pkl",
        output_dir="models/RandomForest"
    )