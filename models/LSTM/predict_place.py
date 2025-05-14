import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import StandardScaler

def predict_place_with_participation_lstm(
    cleaned_data_path,
    binary_data_path,
    target_column,
    output_dir="models/Dense"
):
    # 1. Įkeliame duomenis
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_binary = pd.read_csv(binary_data_path)

    # 2. Nustatome požymius
    competition_columns = [col for col in df_binary.columns if col.startswith("202")]
    competition_columns_sorted = sorted(
        competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]

    static_features = [
        col for col in df_cleaned.columns
        if not col.startswith("202") and col not in ["IBUId", "FullName"]
    ]
    feature_names = static_features + past_columns

    # 3. Įkeliame dalyvavimo modelį (arba naudojame visus)
    clf_feature_names = feature_names
    df_binary_features = df_binary[clf_feature_names].fillna(0)
    df_cleaned["PredictedParticipation"] = 1

    # 4. Duomenys su žinoma vieta
    df_train = df_cleaned[df_cleaned[target_column].notna()].copy()
    y = df_train[target_column].astype(float).values
    X_seq = df_train[past_columns].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Paprastas Dense modelis
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, callbacks=[early_stop], verbose=0)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"dense_regression_place_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.h5")
    model.save(model_path)
    print(f"\n📦 Modelis išsaugotas: {model_path}")

    # 6. Įvertinimas
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Modelio rezultatai (Dense):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")

    # 7. Paklaidų histograma
    errors = y_pred - y_test
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Paklaida (prognozė - tikroji vieta)")
    plt.ylabel("Stebėjimų skaičius")
    plt.title("Modelio paklaidų histograma (Dense)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_place_with_participation_lstm(
        cleaned_data_path="data/female_athletes_cleaned_final.csv",
        binary_data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-03-13 09 (12.5  Short Individual) W",
        output_dir="data/"
    )
