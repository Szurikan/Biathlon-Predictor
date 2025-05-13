import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

def predict_participation_lstm_sequence(data_path, target_column, output_dir="data/", epochs=30):
    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]
    target_column = target_column.strip()

    # 1. Surikiuojame varžybų stulpelius pagal datą
    competition_columns = [col for col in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}", col)]
    competition_columns_sorted = sorted(competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d"))

    if target_column not in competition_columns_sorted:
        raise ValueError(f"Nurodytas stulpelis '{target_column}' nerastas tarp varžybų stulpelių.")

    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]

    # 100 % garantija, kad target netapo įėjimu
    assert target_column not in past_columns, "Klaida: target etapas netyčia pateko tarp įėjimo požymių!"

    # 2. Pašaliname NaN tikslinio stulpelio reikšmes
    df_model = df.dropna(subset=[target_column])

    # 3. Sukuriame laiko seką: tik ankstesnių etapų dalyvavimas
    X_sequence = df_model[past_columns].fillna(0).astype(float).values
    X_sequence = np.expand_dims(X_sequence, axis=-1)

    y = df_model[target_column].astype(int).values

    # 4. Split į train/test
    X_train, X_test, y_train, y_test = train_test_split(X_sequence, y, test_size=0.2, random_state=42)

    # 5. Klasės svoriai
    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_array))

    # 6. Tikras LSTM modelis
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, class_weight=class_weights)

    # 7. Vertinimas (be apribojimų)
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = np.round(y_pred_prob)

    print(f"\nModelio rezultatai (standartinis įvertinimas be apribojimų):\n")
    print(classification_report(y_test, y_pred))

    # 8. Prognozės visoms sportininkėms
    df_all = df.copy()
    X_all_seq = df_all[past_columns].fillna(0).astype(float).values
    X_all_seq = np.expand_dims(X_all_seq, axis=-1)
    all_preds = model.predict(X_all_seq).flatten()

    df_all["PredictedProbability"] = all_preds
    df_all = df_all.sort_values("PredictedProbability", ascending=False)
    df_all["PredictedParticipation"] = 0

    # 9. Dalyvių skaičiaus apribojimas (taikomas tik čia)
    event_name = target_column.lower()
    if "mass start" in event_name:
        selected = df_all.head(30)
    elif "pursuit" in event_name:
        selected = df_all.head(60)
    elif "sprint" in event_name or "individual" in event_name:
        confident = df_all[df_all["PredictedProbability"] > 0.5]
        if len(confident) >= 80:
            selected = confident
        else:
            selected = df_all.head(80)
    else:
        selected = df_all[df_all["PredictedProbability"] > 0.5]

    df_all.loc[selected.index, "PredictedParticipation"] = 1

    # 10. Išsaugome CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(
        output_dir,
        f"lstm_sequence_predictions_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    )
    df_all[["FullName", "PredictedParticipation"]].to_csv(output_csv, index=False)
    print(f"\nPrognozių failas sukurtas: {output_csv}")

    # 11. Išvedame dalyvaujančias
    print("\nSportininkės, kurioms prognozuojamas dalyvavimas šiame etape:")
    for name in df_all[df_all["PredictedParticipation"] == 1]["FullName"]:
        print("-", name)

# Naudojimas (kai reikia):
predict_participation_lstm_sequence("data/female_athletes_binary_competitions.csv", "2025-01-16 05 (15  Individual Competition) W")
