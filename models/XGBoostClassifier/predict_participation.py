import pandas as pd
import numpy as np
import os
import joblib
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, make_scorer, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score
)

def predict_participation_xgb(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)

    df.columns = [col.strip() for col in df.columns]
    target_column = target_column.strip()

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

    scale_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }

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

    print("\n🔍 Pradedama hiperparametrų paieška su GridSearchCV...")
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print(f"\n✅ Geriausi parametrai: {grid.best_params_}")
    print(f"🎯 Geriausias F1-score: {grid.best_score_:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"xgb_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump((model, list(X.columns)), model_path)
    print(f"Modelis išsaugotas: {model_path}")

    # Vertinimas
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=False)
    print(f"\nModelio rezultatai prognozuojant '{target_column}':\n")
    print(report)

    # Sujaukimo matrica
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Sujaukimo matrica")
    plt.tight_layout()
    plt.show()

    # TP, TN, FP, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\n🔢 Statistika:\nTikri teigiami (TP): {tp}\nTikri neigiami (TN): {tn}\nKlaidingi teigiami (FP): {fp}\nKlaidingi neigiami (FN): {fn}")

    # Požymių svarba
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importance.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], align="center")
    plt.gca().invert_yaxis()
    plt.xlabel("Svarba (feature importance)")
    plt.title("10 svarbiausių požymių pagal įtaką prognozei (XGBoost)")
    plt.tight_layout()
    plt.show()

    # Slenksčio analizė
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1-score')
    plt.xlabel('Slenkstis (threshold)')
    plt.ylabel('Reikšmė')
    plt.title('Rodiklių priklausomybė nuo klasifikavimo slenksčio (XGBoost)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_participation_xgb(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-03-13 09 (12.5  Short Individual) W"
    )
