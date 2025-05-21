import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from visualizations.visualizations import save_place_metrics, save_error_distribution
import sqlite3

def evaluate_regression_phase(df, model, X, columns, phase):
    all_true, all_pred = [], []
    stats_list = []

    for col in columns[1:]:
        y_true = df[col].copy()
        y_pred = model.predict(X)
        mask = y_true.notna()
        y_true = y_true[mask]
        y_pred = pd.Series(y_pred, index=X.index)[mask]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        medae = median_absolute_error(y_true, y_pred)

        print(f"\n{phase} etapas: {col}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MedAE: {medae:.2f}")

        stats_list.append({
            "Etapas": col,
            "MAE": mae,
            "RMSE": rmse,
            "MedAE": medae
        })

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    return stats_list, all_true, all_pred

def predict_place_xgb(data_path, target_column, output_dir="data/"):
    # df = pd.read_csv(data_path)
    conn = sqlite3.connect(data_path)
    df = pd.read_sql_query("SELECT * FROM cleaned_data", conn)
    conn.close()
    comp_cols = sorted([c for c in df.columns if c.startswith("202")], key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]

    train_date, val_date = "2024-12-22", "2025-01-25"
    train_cols = [c for c in comp_cols if datetime.strptime(c.split()[0], "%Y-%m-%d") <= datetime.strptime(train_date, "%Y-%m-%d")]
    val_cols = [c for c in comp_cols if train_date < c.split()[0] <= val_date]
    test_cols = [c for c in comp_cols if c.split()[0] > val_date]

    X_train = df[static_feats + train_cols].fillna(0)
    y_train = df[val_cols[0]].copy()
    y_train = y_train[y_train.notna()]
    X_train = X_train.loc[y_train.index]

    grid = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        },
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"\nGeriausias modelis: {grid.best_params_}")
    print("\nValidacijos rezultatai:")
    evaluate_regression_phase(df, best_model, X_train, val_cols, "Val")

    X_final = df[static_feats + train_cols].fillna(0)
    y_final = df[test_cols[0]].copy()
    y_final = y_final[y_final.notna()]
    X_final = X_final.loc[y_final.index]

    final_model = XGBRegressor(**grid.best_params_, objective='reg:squarederror', random_state=42)
    final_model.fit(X_final, y_final)

    print("\nTestavimo rezultatai:")
    test_stats, y_true_all, y_pred_all = evaluate_regression_phase(df, final_model, X_final, test_cols, "Test")

    dates = [datetime.strptime(s['Etapas'].split()[0], "%Y-%m-%d") for s in test_stats]
    maes = [s["MAE"] for s in test_stats]
    rmses = [s["RMSE"] for s in test_stats]
    medaes = [s["MedAE"] for s in test_stats]

    viz_dir = os.path.join("data", "visualizations")
    prefix = "XGBoost"
    save_place_metrics(dates, maes, rmses, medaes, viz_dir, prefix)
    save_error_distribution(y_true_all, y_pred_all, viz_dir, prefix)

    print("\nBendri testavimo rezultatai visiems etapams:")
    print(f"Bendras MAE: {mean_absolute_error(y_true_all, y_pred_all):.3f}")
    print(f"Bendras RMSE: {mean_squared_error(y_true_all, y_pred_all) ** 0.5:.3f}")
    print(f"Bendras MedAE: {median_absolute_error(y_true_all, y_pred_all):.3f}")

    model_path = os.path.join(output_dir, f"next_event_Place_XGBoost.pkl")
    joblib.dump((final_model, list(X_final.columns)), model_path)
    print(f"\nModelis išsaugotas: {model_path}")

predict_place_with_participation = predict_place_xgb

if __name__ == "__main__":
    predict_place_xgb(
        data_path="data/athletes_data.db",
        target_column="2025-12-02 01 (15  Individual Competition) W"
    )
