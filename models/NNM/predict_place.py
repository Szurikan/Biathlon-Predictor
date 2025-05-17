import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from visualizations.visualizations import save_place_metrics, save_error_distribution
import sqlite3

def predict_place_lstm(data_path, target_column, output_dir="data/"):
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
    
    # Geresnė duomenų transformacija LSTM modeliui
    seq_length = min(10, len(train_cols))  # Naudojame paskutinius 10 arba visus turimus etapus
    
    # Statinių požymių normalizavimas
    scaler_static = StandardScaler()
    static_features = scaler_static.fit_transform(df[static_feats].fillna(0))
    
    # SVARBUS PAKEITIMAS: Normalizuojame duomenis nenaudojant stulpelių pavadinimų
    # Tiesiog normalizuojame vertes, bet nepriskiriame jų konkretiems stulpeliams
    train_data_values = df[train_cols[-seq_length:]].fillna(0).values
    scaler_seq = StandardScaler()
    sequence_data = scaler_seq.fit_transform(train_data_values)
    
    # Paruošiame įvesties duomenis (X) ir tikslinius duomenis (y)
    X_seq = np.zeros((len(df), seq_length, 1))
    for i in range(seq_length):
        if i < len(train_cols[-seq_length:]):
            X_seq[:, i, 0] = sequence_data[:, i]
    
    # Statinių ir sekos duomenų sujungimas
    num_static = static_features.shape[1]
    X_combined = np.zeros((len(df), seq_length, num_static + 1))
    
    for i in range(seq_length):
        X_combined[:, i, :num_static] = static_features
        X_combined[:, i, num_static:] = X_seq[:, i, :]
    
    # Tikslinės reikšmės mokymo etapui
    y_train = df[val_cols[0]].copy()
    mask_train = y_train.notna()
    y_train = y_train[mask_train].values
    X_train = X_combined[mask_train]
    
    # Supaprastinta LSTM modelio architektūra
    model = Sequential([
        LSTM(32, input_shape=(seq_length, num_static + 1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping, kad išvengtume persimokymo
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Treniruojame modelį
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=16, 
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Testavimo logika su statistikos rinkimu
    all_y_true, all_y_pred, stats_list = [], [], []
    
    for test_idx, test_col in enumerate(test_cols):
        # Randame praėjusius etapus iki dabartinio testavimo etapo
        past_cols = [c for c in comp_cols if c < test_col]
        
        if len(past_cols) < seq_length:
            # Neturime pakankamai istorinių duomenų
            continue
        
        # Naudojame paskutinius seq_length etapus prieš testavimo etapą
        recent_past_cols = past_cols[-seq_length:]
        
        # SVARBUS PAKEITIMAS: Transformuojame vertes, o ne dataframe su stulpelių pavadinimais
        recent_values = df[recent_past_cols].fillna(0).values
        recent_data_scaled = scaler_seq.transform(recent_values)
        
        # Paruošiame testavimo duomenis
        X_test_seq = np.zeros((len(df), seq_length, 1))
        for i in range(seq_length):
            X_test_seq[:, i, 0] = recent_data_scaled[:, i]
        
        # Sujungiame statinius ir sekos duomenis testavimui
        X_test_combined = np.zeros((len(df), seq_length, num_static + 1))
        for i in range(seq_length):
            X_test_combined[:, i, :num_static] = static_features
            X_test_combined[:, i, num_static:] = X_test_seq[:, i, :]
        
        # Gauname tikslines reikšmes ir prognozuojame
        y_true = df[test_col]
        mask_test = y_true.notna()
        
        X_test_filtered = X_test_combined[mask_test]
        y_true_filtered = y_true[mask_test].values
        
        if len(y_true_filtered) == 0:
            continue
            
        y_pred = model.predict(X_test_filtered).flatten()
        
        # Skaičiuojame metrikus
        mae = mean_absolute_error(y_true_filtered, y_pred)
        rmse = mean_squared_error(y_true_filtered, y_pred)**0.5
        medae = median_absolute_error(y_true_filtered, y_pred)
        
        print(f"\n📊 Testas: {test_col}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MedAE: {medae:.2f}")
        
        stats_list.append({
            "Etapas": test_col,
            "MAE": mae,
            "RMSE": rmse,
            "MedAE": medae
        })
        
        all_y_true.extend(y_true_filtered)
        all_y_pred.extend(y_pred)
    
    # Grafikai ir statistika
    if len(stats_list) > 0:
        dates = [datetime.strptime(s['Etapas'].split()[0], "%Y-%m-%d") for s in stats_list]
        maes = [s['MAE'] for s in stats_list]
        rmses = [s['RMSE'] for s in stats_list]
        medaes = [s['MedAE'] for s in stats_list]
        
        viz_dir = os.path.join("data", "visualizations")
        prefix = "LSTM_Improved"
        save_place_metrics(dates, maes, rmses, medaes, viz_dir, prefix)
        save_error_distribution(all_y_true, all_y_pred, viz_dir, prefix)
        
        print("\n📊 Bendri testavimo rezultatai visiems etapams:")
        print(f"Bendras MAE: {mean_absolute_error(all_y_true, all_y_pred):.3f}")
        print(f"Bendras RMSE: {mean_squared_error(all_y_true, all_y_pred) ** 0.5:.3f}")
        print(f"Bendras MedAE: {median_absolute_error(all_y_true, all_y_pred):.3f}")
    else:
        print("\n⚠️ Nepavyko atlikti testavimo - nėra tinkamų testavimo duomenų.")
    
    
    model_path = os.path.join(output_dir, f"next_event_place_LSTM.keras")
    model.save(model_path)
    print(f"\nModelis išsaugotas: {model_path}")

predict_place_with_participation = predict_place_lstm

if __name__ == "__main__":
    predict_place_lstm(
        data_path="data/athletes_data.db",
        target_column="2025-12-02 01 (15  Individual Competition) W"
    )