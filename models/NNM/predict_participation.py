import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import StandardScaler

def predict_participation_lstm(data_path, target_column, output_dir="models/LSTM"):
    df = pd.read_csv(data_path)

    competition_columns = [col for col in df.columns if col.startswith("202")]
    competition_columns_sorted = sorted(competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d"))
    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]

    static_features = [col for col in df.columns if not col.startswith("202") and col not in ["IBUId", "FullName"]]

    df_model = df.dropna(subset=[target_column])
    X_static = df_model[static_features].fillna(0)
    X_seq = df_model[past_columns].fillna(0).values
    y = df_model[target_column].astype(int).values

    # Normalizuojame seka ir statinius požymius atskirai
    scaler_static = StandardScaler()
    X_static_scaled = scaler_static.fit_transform(X_static)

    scaler_seq = StandardScaler()
    X_seq_scaled = scaler_seq.fit_transform(X_seq)
    X_seq_reshaped = X_seq_scaled.reshape((X_seq_scaled.shape[0], X_seq_scaled.shape[1], 1))

    # Sujungiam statinius ir seka (čia paprastai galima būtų naudoti du "input" sluoksnius, bet paprastumo dėlei čia naudojame tik seka)

    X_train, X_test, y_train, y_test = train_test_split(X_seq_reshaped, y, test_size=0.2, random_state=42)

    # LSTM modelis
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\n🔍 Mokomas LSTM modelis...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, callbacks=[early_stopping], verbose=0)

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f"lstm_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.h5"))

    # Prognozės
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n📊 Modelio rezultatai:")
    print(classification_report(y_test, y_pred))

    # Sujaukimo matrica
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Sujaukimo matrica (LSTM)")
    plt.tight_layout()
    plt.show()

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\n🔢 Statistika:\nTikri teigiami (TP): {tp}\nTikri neigiami (TN): {tn}\nKlaidingi teigiami (FP): {fp}\nKlaidingi neigiami (FN): {fn}")

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
    plt.title('Rodiklių priklausomybė nuo klasifikavimo slenksčio (LSTM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_participation_lstm(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-03-13 09 (12.5  Short Individual) W"
    )
