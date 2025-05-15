import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

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

def predict_participation_lstm(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)
    comp_cols = sorted([c for c in df.columns if c.startswith("202")], key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]

    train_date, val_date = "2024-12-22", "2025-01-25"
    train_cols = [c for c in comp_cols if datetime.strptime(c.split()[0], "%Y-%m-%d") <= datetime.strptime(train_date, "%Y-%m-%d")]
    val_cols = [c for c in comp_cols if train_date < c.split()[0] <= val_date]
    test_cols = [c for c in comp_cols if c.split()[0] > val_date]

    # Scaler treniravimui
    scaler_static = StandardScaler()
    scaler_seq_train = StandardScaler()

    X_static_train = scaler_static.fit_transform(df[static_feats].fillna(0))
    X_seq_train = scaler_seq_train.fit_transform(df[train_cols].fillna(0))
    X_combined_train = np.hstack((X_static_train, X_seq_train)).reshape((len(df), -1, 1))
    y_train = df[val_cols[0]].fillna(0).astype(float)

    model = Sequential([
        Input(shape=(X_combined_train.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_combined_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

    # Scaler testavimui
    scaler_seq_final = StandardScaler()
    X_static_final = scaler_static.transform(df[static_feats].fillna(0))
    X_seq_final = scaler_seq_final.fit_transform(df[train_cols + val_cols].fillna(0))
    X_final = np.hstack((X_static_final, X_seq_final)).reshape((len(df), -1, 1))
    y_final = df[test_cols[0]].fillna(0).astype(float)

    final_model = model
    final_model.fit(X_final, y_final, epochs=50, batch_size=32, validation_split=0.1,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

    print("\n📋 Testavimo rezultatai:")
    all_y_true, all_y_pred, stats_list = [], [], []
    for col in test_cols[1:]:
        y = df[col].fillna(0).astype(int)
        raw = final_model.predict(X_final).flatten()
        pred = adjust_predictions_by_format(raw, col)
        stats = summarize_stats(y, pred, raw, col)
        print_stats(stats)
        stats_list.append(stats)
        all_y_true.extend(y.tolist())
        all_y_pred.extend(pred.tolist())

    # Vizualizacijos
    dates = [datetime.strptime(s['Etapas'].split()[0], "%Y-%m-%d") for s in stats_list]
    plt.figure(figsize=(12, 10))
    for i, (metric, label) in enumerate(zip(['accuracy', 'precision_1', 'recall_1', 'f1_1'], ['Accuracy', 'Precision', 'Recall', 'F1-score'])):
        plt.subplot(2, 2, i+1)
        plt.plot(dates, [s[metric] for s in stats_list], 'o-')
        plt.title(f"{label} pagal etapą")
        plt.xlabel("Data")
        plt.ylabel(label)
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(dates, [s['accuracy'] for s in stats_list], 'o-')
    plt.title("Bendras tikslumas pagal laiką")
    plt.xlabel("Data")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cm_total = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', xticklabels=['Ne', 'Taip'], yticklabels=['Ne', 'Taip'])
    plt.title("Bendra sujaukimo matrica (visi etapai)")
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.show()

    print("\n📊 Bendri testavimo rezultatai visiems etapams:")
    print(classification_report(all_y_true, all_y_pred, digits=2))
    print("\nBendra sujaukimo matrica:")
    print(f"TN: {cm_total[0][0]}, FP: {cm_total[0][1]}")
    print(f"FN: {cm_total[1][0]}, TP: {cm_total[1][1]}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"lstm_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.keras")
    final_model.save(model_path)
    print(f"\nModelis išsaugotas: {model_path}")

if __name__ == "__main__":
    predict_participation_lstm(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-12-02 01 (15  Individual Competition) W"
    )