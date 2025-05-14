import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt

# def predict_place_with_participation_xgb(
#     cleaned_data_path,
#     binary_data_path,
#     target_column,
#     output_dir="models/XGBoost"
# ):
#     import pandas as pd
#     from xgboost import XGBRegressor
#     import joblib, os
#     from datetime import datetime
#     import matplotlib.pyplot as plt

#     df_cleaned = pd.read_csv(cleaned_data_path)
#     df_binary = pd.read_csv(binary_data_path)

#     # Surikiuojame datas
#     competition_columns = [col for col in df_binary.columns if col.startswith("202")]
#     competition_columns_sorted = sorted(competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d"))
#     target_index = competition_columns_sorted.index(target_column)
#     past_columns = competition_columns_sorted[:target_index]

#     static_features = [col for col in df_cleaned.columns if not col.startswith("202") and col not in ["IBUId", "FullName", target_column]]
#     feature_names = static_features + past_columns

#     # Įkeliame dalyvavimo modelį
#     model_name = f"xgb_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
#     model_path = os.path.join("data", model_name)
#     loaded = joblib.load(model_path)
#     clf, clf_feature_names = loaded if isinstance(loaded, tuple) else (loaded, feature_names)

#     # Prognozuojame dalyvavimą
#     df_cleaned["PredictedParticipation"] = clf.predict(df_binary[clf_feature_names].fillna(0))

#     # Mokymosi duomenys – naudoti tik tuos, kurie turi bent vieną žinomą praeities vietą
#     df_train = df_cleaned[df_cleaned[past_columns].notna().sum(axis=1) > 0].copy()
#     df_train["AvgPastPlace"] = df_train[past_columns].mean(axis=1)

#     X_train = df_train[feature_names].fillna(0)
#     y_train = df_train["AvgPastPlace"]

#     model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
#     model.fit(X_train, y_train)

#     # Prognozuojame dalyvaujančioms
#     df_predict = df_cleaned[df_cleaned["PredictedParticipation"] == 1].copy()
#     X_predict = df_predict[feature_names].fillna(0)
#     y_pred = model.predict(X_predict)

#     df_predict["PredictedPlace"] = y_pred
#     df_predict["ActualPlace"] = df_predict[target_column].apply(lambda x: int(x) if pd.notna(x) else "Nedalyvavo")
#     df_predict_sorted = df_predict[["FullName", "PredictedPlace", "ActualPlace"]].sort_values("PredictedPlace").reset_index(drop=True)
#     df_predict_sorted.insert(0, "PredictedRank", df_predict_sorted.index + 1)

#     print(f"\nPrognozės (vidutinė praeities vieta): {len(df_predict_sorted)} sportininkės")
#     print(df_predict_sorted.to_string(index=False))

#     os.makedirs(output_dir, exist_ok=True)
#     model_file = os.path.join(output_dir, f"xgb_regression_avg_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
#     joblib.dump((model, feature_names), model_file)

#     csv_path = os.path.join(output_dir, f"xgb_predicted_places_avg_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
#     df_predict_sorted.to_csv(csv_path, index=False)

#     print(f"\n📦 Modelis išsaugotas: {model_file}")
#     print(f"📄 Prognozės išsaugotos: {csv_path}")

def predict_place_with_participation_xgb(
    cleaned_data_path,
    binary_data_path,
    target_column,
    output_dir="models/XGBoost"
):
    import pandas as pd
    from xgboost import XGBRegressor
    import joblib, os
    from datetime import datetime
    import matplotlib.pyplot as plt

    df_cleaned = pd.read_csv(cleaned_data_path)
    df_binary = pd.read_csv(binary_data_path)

    competition_columns = [col for col in df_binary.columns if col.startswith("202")]
    competition_columns_sorted = sorted(competition_columns, key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d"))
    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]

    static_features = [col for col in df_cleaned.columns if not col.startswith("202") and col not in ["IBUId", "FullName", target_column]]
    feature_names = static_features + past_columns

    model_name = f"xgb_model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    model_path = os.path.join("data", model_name)
    loaded = joblib.load(model_path)
    clf, clf_feature_names = loaded if isinstance(loaded, tuple) else (loaded, feature_names)

    df_cleaned["PredictedParticipation"] = clf.predict(df_binary[clf_feature_names].fillna(0))

    # Naudojame paskutinę žinomą vietą
    df_train = df_cleaned[df_cleaned[past_columns].notna().sum(axis=1) > 0].copy()
    df_train["LastKnownPlace"] = df_train[past_columns].T.ffill().iloc[-1]

    X_train = df_train[feature_names].fillna(0)
    y_train = df_train["LastKnownPlace"]

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    df_predict = df_cleaned[df_cleaned["PredictedParticipation"] == 1].copy()
    X_predict = df_predict[feature_names].fillna(0)
    y_pred = model.predict(X_predict)

    df_predict["PredictedPlace"] = y_pred
    df_predict["ActualPlace"] = df_predict[target_column].apply(lambda x: int(x) if pd.notna(x) else "Nedalyvavo")
    df_predict_sorted = df_predict[["FullName", "PredictedPlace", "ActualPlace"]].sort_values("PredictedPlace").reset_index(drop=True)
    df_predict_sorted.insert(0, "PredictedRank", df_predict_sorted.index + 1)

    print(f"\nPrognozės (paskutinė žinoma vieta): {len(df_predict_sorted)} sportininkės")
    print(df_predict_sorted.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, f"xgb_regression_last_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump((model, feature_names), model_file)

    csv_path = os.path.join(output_dir, f"xgb_predicted_places_last_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    df_predict_sorted.to_csv(csv_path, index=False)

    print(f"\n📦 Modelis išsaugotas: {model_file}")
    print(f"📄 Prognozės išsaugotos: {csv_path}")

if __name__ == "__main__":
    predict_place_with_participation_xgb(
        cleaned_data_path="data/female_athletes_cleaned_final.csv",
        binary_data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-01-23 06 (7.5  Sprint Competition) W",
        output_dir="data/"
    )

