import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
import re
from datetime import datetime

def predict_participation_xgb(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)

    # Normalizuojame stulpelių pavadinimus
    df.columns = [col.strip() for col in df.columns]
    target_column = target_column.strip()

    # Rikiuojame varžybų stulpelius pagal datą
    competition_columns = [col for col in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}", col)]
    competition_columns_sorted = sorted(
        competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )

    if target_column not in competition_columns_sorted:
        raise ValueError(f"Nurodytas stulpelis '{target_column}' nerastas tarp varžybų stulpelių.")

    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]
    static_features = [col for col in df.columns if not re.match(r"^\d{4}-\d{2}-\d{2}", col) and col not in ["IBUId", "FullName"]]

    df_model = df.dropna(subset=[target_column])
    X = df_model[static_features + past_columns].fillna(0)
    y = df_model[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"xgb_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump(model, model_path)
    print(f"Modelis išsaugotas: {model_path}")



    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    print(f"\nModelio rezultatai prognozuojant '{target_column}':\n")
    print(report)

    # Prognozuojame visoms sportininkėms
    df_all = df.copy()
    df_all_features = df_all[static_features + past_columns].fillna(0)
    df_all["PredictedParticipation"] = model.predict(df_all_features)

    print("Naudojami požymiai prognozei:", df_all_features.columns.tolist())
    assert target_column not in df_all_features.columns

    output_csv = os.path.join(output_dir, f"xgb_predictions_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    df_all[["FullName", "PredictedParticipation"]].to_csv(output_csv, index=False)
    print(f"\nPrognozių failas sukurtas: {output_csv}")

    # Išvedame sportininkes, kurioms prognozuojamas dalyvavimas
    predicted_names = df_all[df_all["PredictedParticipation"] == 1]["FullName"]
    print("\nSportininkės, kurioms prognozuojamas dalyvavimas šiame etape:")
    for name in predicted_names:
        print("-", name)

# Naudojimas:
predict_participation_xgb("data/female_athletes_binary_competitions.csv", "2025-01-16 05 (15  Individual Competition) W")
