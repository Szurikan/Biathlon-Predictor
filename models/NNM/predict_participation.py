import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

def predict_participation_lstm(data_path, target_column, output_dir="data/"):
    # Įkeliame duomenis
    df = pd.read_csv(data_path)
    
    # Surūšiuojame varžybų stulpelius chronologiškai
    competition_columns = [col for col in df.columns if col.startswith("202")]
    competition_columns_sorted = sorted(
        competition_columns, 
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    
    # Statiniai požymiai
    static_features = [col for col in df.columns if not col.startswith("202") and col not in ["IBUId", "FullName"]]
    
    # Padalinkime duomenis į treniravimo, validacijos ir testavimo rinkinius pagal datą
    train_end_date = "2024-12-22"  # Treniravimas iki 2024-12-22
    val_end_date = "2025-01-25"    # Validacija iki 2025-01-25
    
    # Nustatome stulpelius kiekvienam rinkiniui
    train_columns = [col for col in competition_columns_sorted 
                     if datetime.strptime(col.split(" ")[0], "%Y-%m-%d") <= datetime.strptime(train_end_date, "%Y-%m-%d")]
    val_columns = [col for col in competition_columns_sorted 
                   if datetime.strptime(train_end_date, "%Y-%m-%d") < datetime.strptime(col.split(" ")[0], "%Y-%m-%d") <= datetime.strptime(val_end_date, "%Y-%m-%d")]
    test_columns = [col for col in competition_columns_sorted 
                    if datetime.strptime(col.split(" ")[0], "%Y-%m-%d") > datetime.strptime(val_end_date, "%Y-%m-%d")]
    
    print(f"Treniravimo etapų: {len(train_columns)}")
    print(f"Validacijos etapų: {len(val_columns)}")
    print(f"Testavimo etapų: {len(test_columns)}")
    
    # Apmokome modelį naudodami treniravimo duomenis
    X_static_train = df[static_features].fillna(0)
    X_seq_train = df[train_columns].fillna(0).values
    y_train = df[val_columns[0]].fillna(0).astype(float)
    
    # Normalizuojame duomenis
    scaler_static = StandardScaler()
    X_static_train_scaled = scaler_static.fit_transform(X_static_train)
    
    scaler_seq = StandardScaler()
    X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train)
    
    # Apjungiame statinius ir sekvencijos požymius
    X_combined_train = np.hstack((X_static_train_scaled, X_seq_train_scaled))
    X_train_reshaped = X_combined_train.reshape((X_combined_train.shape[0], X_combined_train.shape[1], 1))
    
    # LSTM modelio architektūra
    print("\n🔍 Kuriamas LSTM modelis...")
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Regresinė išvestis kaip XGBoost
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Modelio treniravimas
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_reshaped, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Validacijos rezultatai
    print("\n📊 Validacijos rezultatai:")
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1 = []
    
    for val_target in val_columns[1:]:
        y_val = df[val_target].fillna(0).astype(float)
        y_val_binary = y_val.astype(int)
        
        raw_preds = model.predict(X_train_reshaped).flatten()
        val_pred = adjust_predictions_by_format(raw_preds, val_target)
        
        # Skaičiuojame statistikas
        accuracy = accuracy_score(y_val_binary, val_pred)
        precision = precision_score(y_val_binary, val_pred, zero_division=0)
        recall = recall_score(y_val_binary, val_pred, zero_division=0)
        f1 = f1_score(y_val_binary, val_pred, zero_division=0)
        
        val_accuracy.append(accuracy)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        
        # Statistikų lentelė
        print(f"\nEtapas {val_target}:")
        print(f"Vidutinė prognozė: {np.mean(raw_preds):.4f}, Pasirinktos sportininkės: {sum(val_pred)}")
        print("\nStatistikos:")
        print(f"{'':12} precision    recall  f1-score   support")
        print(f"{'0':12} {precision_score(y_val_binary, val_pred, pos_label=0, zero_division=0):.2f}      {recall_score(y_val_binary, val_pred, pos_label=0, zero_division=0):.2f}     {f1_score(y_val_binary, val_pred, pos_label=0, zero_division=0):.2f}       {sum(y_val_binary == 0)}")
        print(f"{'1':12} {precision:.2f}      {recall:.2f}     {f1:.2f}       {sum(y_val_binary == 1)}")
        print(f"\naccuracy{'':<7} {'':<5} {'':<5} {accuracy:.2f}       {len(y_val_binary)}")
        
        # Sujaukimo matrica
        cm = confusion_matrix(y_val_binary, val_pred)
        print("\nSujaukimo matrica:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Galutinio modelio treniravimas
    X_static_final = df[static_features].fillna(0)
    X_seq_final = df[train_columns + val_columns].fillna(0).values
    y_final = df[test_columns[0]].fillna(0).astype(float)
    
    # Normalizuojame duomenis
    X_static_final_scaled = scaler_static.transform(X_static_final)
    X_seq_final_scaled = scaler_seq.fit_transform(X_seq_final)
    
    # Apjungiame statinius ir sekvencijos požymius
    X_combined_final = np.hstack((X_static_final_scaled, X_seq_final_scaled))
    X_final_reshaped = X_combined_final.reshape((X_combined_final.shape[0], X_combined_final.shape[1], 1))
    
    # Galutinio modelio architektūra
    print("\n🔄 Treninguojamas galutinis LSTM modelis su visais žinomais duomenimis...")
    final_model = Sequential()
    final_model.add(LSTM(64, input_shape=(X_final_reshaped.shape[1], 1), return_sequences=True))
    final_model.add(Dropout(0.3))
    final_model.add(LSTM(32, return_sequences=False))
    final_model.add(Dropout(0.3))
    final_model.add(Dense(16, activation='relu'))
    final_model.add(Dense(1, activation='linear'))
    final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Modelio treniravimas
    final_history = final_model.fit(
        X_final_reshaped, y_final,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Testavimo rezultatai
    print("\n📋 Testavimo rezultatai:")
    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1 = []
    test_etapai = []
    
    for test_target in test_columns[1:]:
        y_test = df[test_target].fillna(0).astype(float)
        y_test_binary = y_test.astype(int)
        
        raw_preds = final_model.predict(X_final_reshaped).flatten()
        test_pred = adjust_predictions_by_format(raw_preds, test_target)
        
        # Skaičiuojame statistikas
        accuracy = accuracy_score(y_test_binary, test_pred)
        precision = precision_score(y_test_binary, test_pred, zero_division=0)
        recall = recall_score(y_test_binary, test_pred, zero_division=0)
        f1 = f1_score(y_test_binary, test_pred, zero_division=0)
        
        test_accuracy.append(accuracy)
        test_precision.append(precision)
        test_recall.append(recall)
        test_f1.append(f1)
        test_etapai.append(datetime.strptime(test_target.split(" ")[0], "%Y-%m-%d"))
        
        # Statistikų lentelė
        print(f"\nEtapas {test_target}:")
        print(f"Vidutinė prognozė: {np.mean(raw_preds):.4f}, Pasirinktos sportininkės: {sum(test_pred)}")
        print("\nStatistikos:")
        print(f"{'':12} precision    recall  f1-score   support")
        print(f"{'0':12} {precision_score(y_test_binary, test_pred, pos_label=0, zero_division=0):.2f}      {recall_score(y_test_binary, test_pred, pos_label=0, zero_division=0):.2f}     {f1_score(y_test_binary, test_pred, pos_label=0, zero_division=0):.2f}       {sum(y_test_binary == 0)}")
        print(f"{'1':12} {precision:.2f}      {recall:.2f}     {f1:.2f}       {sum(y_test_binary == 1)}")
        print(f"\naccuracy{'':<7} {'':<5} {'':<5} {accuracy:.2f}       {len(y_test_binary)}")
        
        # Sujaukimo matrica
        cm = confusion_matrix(y_test_binary, test_pred)
        print("\nSujaukimo matrica:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Mokymosi istorijos vizualizacija
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(final_history.history['loss'], label='Treniravimo klaida')
    plt.plot(final_history.history['val_loss'], label='Validacijos klaida')
    plt.xlabel('Epizodai')
    plt.ylabel('MSE klaida')
    plt.title('Mokymosi kreivė (LSTM)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(final_history.history['mae'], label='Treniravimo MAE')
    plt.plot(final_history.history['val_mae'], label='Validacijos MAE')
    plt.xlabel('Epizodai')
    plt.ylabel('MAE')
    plt.title('Vidutinė absoliuti klaida (LSTM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Testavimo rezultatų grafikas
    if test_etapai:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(test_etapai, test_accuracy, 'o-', label='Accuracy')
        plt.xlabel("Data")
        plt.ylabel("Accuracy")
        plt.title("Tikslumas pagal etapą")
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(test_etapai, test_precision, 'o-', label='Precision (klasė 1)')
        plt.xlabel("Data")
        plt.ylabel("Precision")
        plt.title("Precision pagal etapą")
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(test_etapai, test_recall, 'o-', label='Recall (klasė 1)')
        plt.xlabel("Data")
        plt.ylabel("Recall")
        plt.title("Recall pagal etapą")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(test_etapai, test_f1, 'o-', label='F1-score (klasė 1)')
        plt.xlabel("Data")
        plt.ylabel("F1-score")
        plt.title("F1-score pagal etapą")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Sujaukimo matricos vizualizacija paskutiniam testavimo etapui
        if test_columns:
            y_last = df[test_columns[-1]].fillna(0).astype(int)
            raw_preds_last = final_model.predict(X_final_reshaped).flatten()
            preds_last = adjust_predictions_by_format(raw_preds_last, test_columns[-1])
            
            plt.figure(figsize=(8, 6))
            cm_last = confusion_matrix(y_last, preds_last)
            sns.heatmap(cm_last, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Nedalyvauja', 'Dalyvauja'],
                        yticklabels=['Nedalyvauja', 'Dalyvauja'])
            plt.xlabel('Prognozuota klasė')
            plt.ylabel('Tikroji klasė')
            plt.title(f'Sujaukimo matrica ({test_columns[-1]})')
            plt.tight_layout()
            plt.show()
    
    # Išsaugome modelį
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"lstm_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.keras")
    final_model.save(model_path)
    
    # Išsaugome skalerius
    scaler_path = os.path.join(output_dir, f"scalers_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump({
        'static': scaler_static,
        'seq': scaler_seq,
        'feature_names': {
            'static': static_features,
            'sequence': train_columns + val_columns
        }
    }, scaler_path)
    
    print(f"\nModelis išsaugotas: {model_path}")
    print(f"Skaleris išsaugotas: {scaler_path}")
    
    # Prognozavimas būsimam etapui
    if target_column not in competition_columns:
        print(f"\n🔮 Prognozuojamas dalyvavimas būsimame etape: {target_column}")
        future_pred_raw = final_model.predict(X_final_reshaped).flatten()
        future_pred = adjust_predictions_by_format(future_pred_raw, target_column)
        
        # Sukuriame DataFrame su prognozėmis ir sportininkių tikimybių reitingais
        predictions_df = pd.DataFrame({
            "IBUId": df["IBUId"],
            "FullName": df["FullName"],
            "RawScore": future_pred_raw,
            "PredictedParticipation": future_pred
        }).sort_values(by="RawScore", ascending=False)
        
        # Išsaugome prognozes
        predictions_path = os.path.join(output_dir, f"lstm_predictions_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Prognozės išsaugotos: {predictions_path}")
        print(f"Prognozuojamas dalyvių skaičius: {sum(future_pred)}")
        
        # Rodome TOP-10 sportininkių prognozes
        print("\nTOP-10 sportininkių pagal prognozuojamą dalyvavimo reitingą:")
        top_10 = predictions_df.head(10)
        for _, row in top_10.iterrows():
            print(f"{row['FullName']} (Reitingas: {row['RawScore']:.4f})")

def adjust_predictions_by_format(pred_scores, competition_name):
    """
    Pritaiko prognozes pagal varžybų formatą:
    - Mass Start: 30 dalyvių
    - Pursuit: 60 dalyvių
    - Individual/Sprint: 100 dalyvių
    """
    if "Mass Start" in competition_name:
        target_count = 30
        format_type = "Mass Start"
    elif "Pursuit" in competition_name:
        target_count = 60
        format_type = "Pursuit"
    else:
        target_count = 100
        format_type = "Individual/Sprint"
    
    top_indices = np.argsort(pred_scores)[-target_count:]
    binary_selection = np.zeros_like(pred_scores, dtype=int)
    binary_selection[top_indices] = 1
    
    print(f"Pritaikytas formatas: {format_type} ({target_count} sportininkės)")
    return binary_selection

if __name__ == "__main__":
    predict_participation_lstm(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-12-02 01 (15  Individual Competition) W"
    )