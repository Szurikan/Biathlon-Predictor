import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from visualizations.visualizations import (
    save_metric_plots, save_accuracy_plot, save_confusion_matrix
)
import sqlite3

def adjust_predictions_by_format(pred_scores, competition_name):
    if "Mass Start" in competition_name:
        target_count = 30
    elif "Pursuit" in competition_name:
        target_count = 60
    else:
        target_count = 100
    top_indices = np.argsort(pred_scores)[-target_count:]
    binary_selection = np.zeros_like(pred_scores, dtype=int)
    binary_selection[top_indices] = 1
    return binary_selection

def summarize_stats(y_true, y_pred, raw_preds, etapas):
    return {
        "Etapas": etapas,
        "Vid. prognozė": round(np.mean(raw_preds), 4),
        "Dalyvauja": int(sum(y_pred)),
        "accuracy": round(accuracy_score(y_true, y_pred), 2),
        "precision_1": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 2),
        "recall_1": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 2),
        "f1_1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 2),
        "confusion": confusion_matrix(y_true, y_pred).tolist()
    }

def print_stats(stats):
    print(f"\nEtapas {stats['Etapas']}:")
    print(f"Vidutinė prognozė: {stats['Vid. prognozė']}, Dalyvauja: {stats['Dalyvauja']}")
    print("Statistikos:")
    print(f"Precision: {stats['precision_1']}, Recall: {stats['recall_1']}, F1: {stats['f1_1']}, Accuracy: {stats['accuracy']}")
    cm = stats['confusion']
    print("Sujaukimo matrica:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}\nFN: {cm[1][0]}, TP: {cm[1][1]}")

def evaluate_phase(df, model, X, columns, phase):
    all_y_true, all_y_pred = [], []
    stats_list = []
    for col in columns[1:]:
        y = df[col].fillna(0).astype(int)
        raw = model.predict(X)
        pred = adjust_predictions_by_format(raw, col)
        stats = summarize_stats(y, pred, raw, col)
        print_stats(stats)
        stats_list.append(stats)
        all_y_true.extend(y.tolist())
        all_y_pred.extend(pred.tolist())
    return all_y_true, all_y_pred, stats_list

def predict_participation(data_path, target_column, output_dir="data/"):
    # df = pd.read_csv(data_path)
    conn = sqlite3.connect(data_path)
    df = pd.read_sql_query("SELECT * FROM binary_data", conn)
    conn.close()
    comp_cols = sorted([c for c in df.columns if c.startswith("202")], key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]

    train_date, val_date = "2024-12-22", "2025-01-25"
    train_cols = [c for c in comp_cols if datetime.strptime(c.split()[0], "%Y-%m-%d") <= datetime.strptime(train_date, "%Y-%m-%d")]
    val_cols = [c for c in comp_cols if train_date < c.split()[0] <= val_date]
    test_cols = [c for c in comp_cols if c.split()[0] > val_date]

    X_train = df[static_feats + train_cols].fillna(0)
    y_train = df[val_cols[0]].fillna(0).astype(float)

    grid = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                        {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
                        scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"\nGeriausias modelis: {grid.best_params_} su MSE={-grid.best_score_:.4f}")
    print("\nValidacijos rezultatai:")
    evaluate_phase(df, best_model, X_train, val_cols, "Val")

    X_final = df[static_feats + train_cols + val_cols].fillna(0)
    y_final = df[test_cols[0]].fillna(0).astype(float)
    final_model = XGBRegressor(**grid.best_params_, objective='reg:squarederror', random_state=42)
    final_model.fit(X_final, y_final)

    print("\nTestavimo rezultatai:")
    y_true_all, y_pred_all, test_stats = evaluate_phase(df, final_model, X_final, test_cols, "Test")

    dates = [datetime.strptime(s['Etapas'].split()[0], "%Y-%m-%d") for s in test_stats]
    viz_dir = os.path.join("data", "visualizations")
    prefix = "XGBoost"
    save_metric_plots(dates, test_stats, viz_dir, prefix)
    save_accuracy_plot(dates, test_stats, viz_dir, prefix)
    cm_total = confusion_matrix(y_true_all, y_pred_all)
    save_confusion_matrix(cm_total, viz_dir, prefix)

    print("\nBendri testavimo rezultatai visiems etapams:")
    print(classification_report(y_true_all, y_pred_all, digits=2))
    print("\nBendra sujaukimo matrica:")
    print(f"TN: {cm_total[0][0]}, FP: {cm_total[0][1]}")
    print(f"FN: {cm_total[1][0]}, TP: {cm_total[1][1]}")


    model_path = os.path.join(output_dir, f"next_event_Participation_XGBoost.pkl")
    joblib.dump((final_model, list(X_final.columns)), model_path)
    print(f"\nModelis išsaugotas: {model_path}")

if __name__ == "__main__":
    predict_participation(data_path="data/athletes_data.db", 
                          target_column="Participated"
                          )
