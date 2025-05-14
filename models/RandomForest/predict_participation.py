import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

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
    
    # # Modelio kūrimas ir mokymas
    # model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # model.fit(X_train, y_train)

    # Nustatome F1-score kaip optimizavimo kriterijų
    f1 = make_scorer(f1_score, average='macro')

    # Tinklo paieška (GridSearch)
    param_grid = {
        'n_estimators': list(range(5, 305, 5))
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        scoring=f1,
        cv=5,  # 5-kartinė kirtinė validacija
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Vykdoma GridSearchCV optimizacija...")
    grid_search.fit(X_train, y_train)

    # Geriausias modelis
    model = grid_search.best_estimator_
    print(f"\n✅ Geriausias modelis: n_estimators={grid_search.best_params_['n_estimators']} su F1-score={grid_search.best_score_:.4f}")


    # Sukuriame DataFrame su požymių svarbomis
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Pasirenkame 10 svarbiausių
    top_features = feature_importance.head(10)

    # Piešiame stulpelinę diagramą
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], align="center")
    plt.gca().invert_yaxis()  # Svarbiausi viršuje
    plt.xlabel("Svarba (feature importance)")
    plt.title("10 svarbiausių požymių pagal įtaką prognozei")
    plt.tight_layout()
    plt.show()

    # Braižome visų bandymų rezultatus
    results = pd.DataFrame(grid_search.cv_results_)

    plt.figure(figsize=(10, 6))
    plt.plot(results['param_n_estimators'], results['mean_test_score'], marker='o')
    plt.xlabel("n_estimators")
    plt.ylabel("Vidutinis F1-score (5-fold CV)")
    plt.title("F1-score priklausomybė nuo n_estimators")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
        target_column="2025-02-16 07 (10  Pursuit Competition) W"
    )