import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

def show_error_percentiles(y_true, y_pred):
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    percentiles = [50, 75, 90, 95]
    print("\n📈 Klaidų paskirstymo percentiliai:")
    for p in percentiles:
        print(f"{p}%% klaida < {np.percentile(errors, p):.2f} vietų")

    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color="skyblue")
    plt.title("Prognozės klaidų pasiskirstymas")
    plt.xlabel("Absoliuti klaida (vietų skirtumas)")
    plt.ylabel("Sportininkių skaičius")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_model(units, dropout_rate, time_steps, static_dim):
    seq_input = Input(shape=(time_steps, 1), name="sequence_input")
    x = LSTM(units, activation='tanh')(seq_input)
    x = Dropout(dropout_rate)(x)

    static_input = Input(shape=(static_dim,), name="static_input")
    concat = Concatenate()([x, static_input])
    dense = Dense(units // 2, activation='relu')(concat)
    output = Dense(1)(dense)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_place_lstm(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)
    comp_cols = sorted([c for c in df.columns if c.startswith("202")], key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"))
    static_feats = [c for c in df.columns if not c.startswith("202") and c not in ["IBUId", "FullName"]]

    train_date, val_date = "2024-12-22", "2025-01-25"
    train_cols = [c for c in comp_cols if datetime.strptime(c.split()[0], "%Y-%m-%d") <= datetime.strptime(train_date, "%Y-%m-%d")]
    val_cols = [c for c in comp_cols if train_date < c.split()[0] <= val_date]
    test_cols = [c for c in comp_cols if c.split()[0] > val_date]

    y = df[val_cols[0]].copy()
    mask = y.notna()

    X_seq = df[train_cols].fillna(0).loc[mask].values
    X_static = df[static_feats].fillna(0).loc[mask].values
    y = y[mask].astype(float).values

    scaler_seq = StandardScaler()
    scaler_static = StandardScaler()

    X_seq_scaled = scaler_seq.fit_transform(X_seq)
    X_static_scaled = scaler_static.fit_transform(X_static)
    X_seq_reshaped = X_seq_scaled.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
        X_seq_reshaped, X_static_scaled, y, test_size=0.2, random_state=42)

    def objective(trial):
        units = trial.suggest_categorical("units", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 30, 100)

        model = build_model(units, dropout_rate, time_steps=X_seq.shape[1], static_dim=X_static.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            [X_seq_train, X_static_train], y_train,
            validation_data=([X_seq_val, X_static_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        val_pred = model.predict([X_seq_val, X_static_val]).flatten()
        return mean_absolute_error(y_val, val_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("\n🔍 Geriausi hiperparametrai:", study.best_params)

    best_params = study.best_params
    final_model = build_model(best_params['units'], best_params['dropout_rate'], time_steps=X_seq.shape[1], static_dim=X_static.shape[1])
    final_model.fit(
        [X_seq_train, X_static_train], y_train,
        validation_data=([X_seq_val, X_static_val], y_val),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    y_pred = final_model.predict([X_seq_val, X_static_val]).flatten()

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    medae = median_absolute_error(y_val, y_pred)

    print("\n📊 Testavimo rezultatai (LSTM su static + Optuna):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MedAE: {medae:.2f}")

    show_error_percentiles(y_val, y_pred)

    event_type = "Sprint" if "Sprint" in target_column else \
                "Pursuit" if "Pursuit" in target_column else \
                "Individual" if "Individual" in target_column else \
                "MassStart" if "Mass Start" in target_column else "Unknown"

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{event_type}_LSTM_Next.h5")
    final_model.save(model_path)
    print(f"\nModelis išsaugotas: {model_path}")

predict_place_with_participation = predict_place_lstm

if __name__ == "__main__":
    predict_place_lstm(
        data_path="data/female_athletes_cleaned_final.csv",
        target_column="2025-12-02 01 (15  Individual Competition) W"
    )