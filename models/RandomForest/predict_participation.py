import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def predict_participation(data_path, target_column, output_dir="data/"):
    df = pd.read_csv(data_path)

    competition_columns = [col for col in df.columns if col.startswith("202")]
    
    if target_column not in df.columns:
        raise ValueError(f"Nurodytas stulpelis '{target_column}' nerastas faile.")

    competition_columns_sorted = sorted(
        competition_columns, 
        key=lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d")
    )
    
    target_index = competition_columns_sorted.index(target_column)
    past_columns = competition_columns_sorted[:target_index]
    
    assert target_column not in past_columns, "Klaida: tikslo stulpelis pateko į požymius!"
    
    static_features = [col for col in df.columns if not col.startswith("202") and col not in ["IBUId", "FullName"]]
    df_model = df.dropna(subset=[target_column])
    
    X = df_model[static_features + past_columns]
    y = df_model[target_column].astype(int)
    
    assert target_column not in X.columns, "Klaida: tikslo stulpelis yra tarp požymių!"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    f1 = make_scorer(f1_score, average='macro')
    param_grid = {'n_estimators': list(range(5, 305, 5))}

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        scoring=f1,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Vykdoma GridSearchCV optimizacija...")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"\n✅ Geriausias modelis: n_estimators={grid_search.best_params_['n_estimators']} su F1-score={grid_search.best_score_:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{target_column.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump((model, list(X.columns)), model_path)
    print(f"Modelis išsaugotas: {model_path}")

    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred, output_dict=False)
    print(f"\nModelio rezultatai prognozuojant '{target_column}':\n")
    print(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Sujaukimo matrica")
    plt.show()

    # Tikri ir klaidingi atvejai
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\n🔢 Statistika:\nTikri teigiami (TP): {tp}\nTikri neigiami (TN): {tn}\nKlaidingi teigiami (FP): {fp}\nKlaidingi neigiami (FN): {fn}")

    # Požymių svarba
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importance.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], align="center")
    plt.gca().invert_yaxis()
    plt.xlabel("Svarba (feature importance)")
    plt.title("10 svarbiausių požymių pagal įtaką prognozei")
    plt.tight_layout()
    plt.show()

    # GridSearch n_estimators rezultatai
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    plt.plot(results['param_n_estimators'], results['mean_test_score'], marker='o')
    plt.xlabel("n_estimators")
    plt.ylabel("Vidutinis F1-score (5-fold CV)")
    plt.title("F1-score priklausomybė nuo n_estimators")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Koreliacijos matrica
    numeric_df = df_model[past_columns].fillna(0)
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title("Varžybų rezultatų koreliacijos matrica")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_participation(
        data_path="data/female_athletes_binary_competitions.csv",
        target_column="2025-03-13 09 (12.5  Short Individual) W"
    )
