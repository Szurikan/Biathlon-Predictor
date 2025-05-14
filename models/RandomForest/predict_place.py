import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def predict_place_with_participation(
    cleaned_data_path,
    binary_data_path,
    target_column,
    output_dir="models/RandomForest"
):
    # 1. Įkeliame duomenis
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_binary = pd.read_csv(binary_data_path)

    # 2. Požymių atranka ir rūšiavimas
    competition_columns = [col for col in df_binary.columns if col.startswith("202")]
    competition_columns_sorted = sorted(
        competition_columns,
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )

    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]
    assert target_column not in past_columns, "❌ Klaida: tikslo stulpelis pateko į praeities stulpelius!"

    static_features = [
        col for col in df_cleaned.columns
        if not col.startswith("202") and col not in ["IBUId", "FullName"]
    ]

    # Saugiklis: jei tikslinis stulpelis kažkaip pateko į static_features
    if target_column in static_features:
        print(f"⚠️ Perspėjimas: tikslo stulpelis buvo static_features sąraše – pašaliname: {target_column}")
        static_features = [f for f in static_features if f != target_column]

    feature_names = static_features + past_columns

    # Dar kartą šaliname iš feature_names, jei vis dar įsimaišė
    if target_column in feature_names:
        print(f"⚠️ Perspėjimas: pašaliname tikslo stulpelį iš feature_names: {target_column}")
        feature_names = [f for f in feature_names if f != target_column]

    # Patikrinimai prieš tęsiant
    assert target_column not in feature_names, f"❌ Klaida: tikslo stulpelis vis dar yra feature_names!"

    print(f"📋 Požymių skaičius: {len(feature_names)}")
    print(f"📋 Ar tikslo stulpelis tarp požymių? {'TAIP' if target_column in feature_names else 'NE'}")

    # 3. Įkeliame dalyvavimo modelį
    participation_model_filename = f"model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    participation_model_path = os.path.join("data", participation_model_filename)

    loaded = joblib.load(participation_model_path)
    if isinstance(loaded, tuple):
        clf, clf_feature_names = loaded
    else:
        clf = loaded
        clf_feature_names = feature_names

    df_binary_features = df_binary[clf_feature_names].fillna(0)
    df_cleaned["PredictedParticipation"] = clf.predict(df_binary_features)

    # 4. Mokymo duomenys
    df_train = df_cleaned[df_cleaned[target_column].notna()].copy()
    X = df_train[feature_names].fillna(0)
    y = df_train[target_column].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. GridSearchCV
    param_grid = {'n_estimators': list(range(5, 305, 5))}
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Vykdoma GridSearchCV optimizacija...")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"\n✅ Geriausias modelis: n_estimators={grid_search.best_params_['n_estimators']}")

    # 6. Modelio įvertinimas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Naudojame np.sqrt dėl senesnės sklearn versijos
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Modelio rezultatai (testiniame rinkinyje):")
    print(f"MAE (Vid. absoliutinė paklaida): {mae:.2f}")
    print(f"RMSE (Kvadratinė vid. paklaida): {rmse:.2f}")
    print(f"R2 (determinacijos koeficientas): {r2:.2f}")

    # 7. Požymių svarbos vizualizacija
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importance.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], edgecolor='black')
    plt.gca().invert_yaxis()
    plt.xlabel("Svarba (feature importance)")
    plt.title("📊 Top 10 požymių pagal įtaką vietos prognozei")
    plt.tight_layout()
    plt.show()

    # 8. GridSearch rezultatai
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    plt.plot(results['param_n_estimators'], -results['mean_test_score'], marker='o')
    plt.xlabel("n_estimators")
    plt.ylabel("Vidutinis MAE (5-fold CV)")
    plt.title("MAE priklausomybė nuo n_estimators")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9. Koreliacijų heatmap
    numeric_df = df_train[past_columns].fillna(0)
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title("Varžybų rezultatų koreliacijos matrica")
    plt.tight_layout()
    plt.show()

    # 10. Išsaugo modelį
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"regression_place_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump((model, feature_names), model_path)
    print(f"\nModelis išsaugotas: {model_path}")

if __name__ == "__main__":
    predict_place_with_participation(
        cleaned_data_path="data/female_athletes_cleaned_final.csv",
        binary_data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-03-13 09 (12.5  Short Individual) W",
        output_dir="data/"
    )
