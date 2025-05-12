from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def train_model(X, y, n_estimators=100, random_state=42):
    """
    Apmoko modelį su duotais požymiais ir taikiniais.
    
    Args:
        X (pandas.DataFrame): Požymių matrica
        y (pandas.Series): Taikinių vektorius
        n_estimators (int): Medžių skaičius Random Forest modelyje
        random_state (int): Atsitiktinių skaičių generatoriaus sėkla
    
    Returns:
        tuple: (model, scaler) - apmokytas modelis ir duomenų normalizavimo objektas
    """
    # Normalizuojame duomenis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apmokome modelį
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_results(model, scaler, X, names, nations):
    """
    Prognozuoja sportininkių rezultatus ir grąžina surūšiuotą DataFrame.
    
    Args:
        model: Apmokytas modelis
        scaler: Duomenų normalizavimo objektas
        X (pandas.DataFrame): Požymių matrica
        names (pandas.Series): Sportininkių vardai
        nations (pandas.Series): Sportininkių šalys
    
    Returns:
        pandas.DataFrame: Prognozuotos vietos, surūšiuotos nuo geriausios
    """
    # Normalizuojame duomenis
    X_scaled = scaler.transform(X)
    
    # Prognozuojame rezultatus
    y_pred = model.predict(X_scaled)
    
    # Sukuriame prognozių DataFrame
    idx_list = X.index.tolist()
    
    predictions_df = pd.DataFrame({
        'idx': idx_list,
        'FullName': names.iloc[idx_list].values,
        'Nation': nations.iloc[idx_list].values,
        'Predicted': y_pred
    })
    
    # Surūšiuojame pagal prognozę (mažesnė vertė = geresnė vieta)
    predictions_df_sorted = predictions_df.sort_values('Predicted').reset_index(drop=True)
    
    return predictions_df_sorted

def get_top_predictions(predictions_df, top_n=10):
    """
    Grąžina geriausias prognozes.
    
    Args:
        predictions_df (pandas.DataFrame): Surūšiuotos prognozės
        top_n (int): Kiek geriausių rezultatų grąžinti
    
    Returns:
        pandas.DataFrame: Geriausios prognozės
    """
    return predictions_df.head(top_n).copy()