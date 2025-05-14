import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
import re
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np

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

    # model = XGBClassifier(
    #     n_estimators=100,
    #     max_depth=5,
    #     learning_rate=0.1,
    #     objective='binary:logistic',
    #     scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    #     eval_metric='logloss',
    #     random_state=42
    # )
    # model.fit(X_train, y_train)

    # Balansavimo svoris
    scale_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# Hiperparametrų tinklelis
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }

# XGBoost su fiksuotais parametrais ir GridSearch
    base_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42
    )

    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring=make_scorer(f1_score, average='macro'),
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    print("🔍 Pradedama hiperparametrų paieška su GridSearchCV...")
    grid.fit(X_train, y_train)

    # Konvertuojame rezultatus į DataFrame
    results = pd.DataFrame(grid.cv_results_)

    # 1. F1-score pagal n_estimators skirtingiems max_depth
    plt.figure(figsize=(10, 6))
    for depth in sorted(results["param_max_depth"].unique()):
        subset = results[results["param_max_depth"] == depth]
        mean_scores = subset.groupby("param_n_estimators")["mean_test_score"].mean()
        plt.plot(mean_scores.index, mean_scores.values, marker='o', label=f"max_depth={depth}")
    plt.title("F1-score priklausomybė nuo n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("Vidutinis F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. F1-score pagal learning_rate skirtingiems max_depth
    plt.figure(figsize=(10, 6))
    for depth in sorted(results["param_max_depth"].unique()):
        subset = results[results["param_max_depth"] == depth]
        mean_scores = subset.groupby("param_learning_rate")["mean_test_score"].mean()
        plt.plot(mean_scores.index.astype(float), mean_scores.values, marker='o', label=f"max_depth={depth}")
    plt.title("F1-score priklausomybė nuo learning_rate")
    plt.xlabel("learning_rate")
    plt.ylabel("Vidutinis F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Heatmap (max_depth vs learning_rate)
    pivot = results.pivot_table(
        index="param_max_depth",
        columns="param_learning_rate",
        values="mean_test_score"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label="F1-score")
    plt.xticks(ticks=range(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
    plt.yticks(ticks=range(len(pivot.index)), labels=[str(v) for v in pivot.index])
    plt.xlabel("learning_rate")
    plt.ylabel("max_depth")
    plt.title("F1-score šilumos žemėlapis (Heatmap)")
    plt.tight_layout()
    plt.show()

    # Geriausias modelis
    model = grid.best_estimator_
    print(f"\n✅ Geriausi parametrai: {grid.best_params_}")
    print(f"🎯 Geriausias F1-score: {grid.best_score_:.4f}")

    # Sukuriame feature importance DataFrame
   
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
    plt.title("10 svarbiausių požymių pagal įtaką prognozei (XGBoost)")
    plt.tight_layout()
    plt.show()

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

    # print("Naudojami požymiai prognozei:", df_all_features.columns.tolist())
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
predict_participation_xgb("data/female_athletes_binary_competitions.csv", "2025-01-23 06 (7.5  Sprint Competition) W")
