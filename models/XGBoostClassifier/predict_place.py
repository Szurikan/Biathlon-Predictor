import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_place_with_participation_xgb(
    cleaned_data_path,
    binary_data_path,
    target_column,
    output_dir="models/XGBoost"
):
    # 1. Įkeliame duomenis
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_binary = pd.read_csv(binary_data_path)

    # 2. Nustatome požymius
    competition_columns = [col for col in df_binary.columns if col.startswith("202")]
    competition_columns_sorted = sorted(
        competition_columns,
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]

    static_features = [
        col for col in df_cleaned.columns
        if not col.startswith("202") and col not in ["IBUId", "FullName"]
    ]
    feature_names = static_features + past_columns

    # 3. Įkeliame dalyvavimo modelį
    model_name = f"xgb_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    model_path = os.path.join("data", model_name)
    loaded = joblib.load(model_path)
    clf, clf_feature_names = loaded if isinstance(loaded, tuple) else (loaded, feature_names)

    df_binary_features = df_binary[clf_feature_names].fillna(0)
    df_cleaned["PredictedParticipation"] = clf.predict(df_binary_features)

    # 4. Mokymo duomenys (tik su žinoma vieta)
    df_train = df_cleaned[df_cleaned[target_column].notna()].copy()
    X = df_train[feature_names].fillna(0)
    y = df_train[target_column].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. GridSearch
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Vykdoma GridSearchCV optimizacija...")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"\n✅ Geriausias modelis: {grid_search.best_params_}")

    # 6. Modelio įvertinimas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Modelio rezultatai:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")

    # 7. Požymių svarbos grafikas
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
    plt.title("Top 10 požymių (XGBoost)")
    plt.tight_layout()
    plt.show()

    # 8. GridSearch rezultatai
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    for depth in sorted(results["param_max_depth"].unique()):
        subset = results[results["param_max_depth"] == depth]
        mean_scores = subset.groupby("param_n_estimators")["mean_test_score"].mean()
        plt.plot(mean_scores.index, -mean_scores.values, marker='o', label=f"max_depth={depth}")
    plt.title("MAE priklausomybė nuo n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("Vidutinis MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9. Paklaidų histograma
    errors = y_pred - y_test
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Paklaida (prognozė - tikroji vieta)")
    plt.ylabel("Stebėjimų skaičius")
    plt.title("Modelio paklaidų histograma (XGBoost)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 10. Modelio išsaugojimas
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"xgb_regression_place_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump((model, feature_names), model_path)
    print(f"\n📦 Modelis išsaugotas: {model_path}")

if __name__ == "__main__":
    predict_place_with_participation_xgb(
        cleaned_data_path="data/female_athletes_cleaned_final.csv",
        binary_data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-02-23 07 (12.5  Mass Start Competition) W",
        output_dir="data/"
    )