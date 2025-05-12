import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_top_predictions(top_predictions, test_df, target_col, actual_participants):
    """
    Įvertina geriausių prognozių tikslumą.
    
    Args:
        top_predictions (pandas.DataFrame): Geriausios prognozės
        test_df (pandas.DataFrame): Testavimo duomenys
        target_col (str): Tikslo stulpelis
        actual_participants (list): Faktinių dalyvių indeksai
    
    Returns:
        pandas.DataFrame: Įvertintos prognozės su papildoma informacija
    """
    evaluated_predictions = top_predictions.copy()
    
    # Tikriname, kiek iš šių sportininkių iš tikrųjų dalyvavo
    for i, row in evaluated_predictions.iterrows():
        idx = row['idx']
        actual_result = test_df.loc[idx, target_col]
        
        if pd.isna(actual_result):
            evaluated_predictions.loc[i, 'ActualRank'] = np.nan
            evaluated_predictions.loc[i, 'Participated'] = False
        else:
            # Skaičiuojame faktinę vietą tarp visų dalyvių
            all_results = test_df.loc[actual_participants, target_col]
            rank = (all_results <= actual_result).sum()
            evaluated_predictions.loc[i, 'ActualRank'] = rank
            evaluated_predictions.loc[i, 'Participated'] = True
    
    return evaluated_predictions

def calculate_model_metrics(model, scaler, X, y_actual):
    """
    Apskaičiuoja modelio veikimo metrikas.
    
    Args:
        model: Apmokytas modelis
        scaler: Duomenų normalizavimo objektas
        X (pandas.DataFrame): Požymių matrica
        y_actual (pandas.Series): Faktinės reikšmės
    
    Returns:
        dict: Modelio metrikos (MAE, R²)
    """
    try:
        # Normalizuojame duomenis
        X_scaled = scaler.transform(X)
        
        # Prognozuojame rezultatus
        y_pred = model.predict(X_scaled)
        
        # Užtikriname, kad y_actual yra skaitinės reikšmės
        y_actual_clean = pd.to_numeric(y_actual, errors='coerce').dropna()
        
        if len(y_actual_clean) > 0:
            mae = mean_absolute_error(y_actual_clean, y_pred[:len(y_actual_clean)])
            r2 = r2_score(y_actual_clean, y_pred[:len(y_actual_clean)])
            
            print(f"  Modelio metrikos (testuojant ant faktinių dalyvių):")
            print(f"  MAE: {mae:.2f}, R2: {r2:.3f}")
            
            return {'mae': mae, 'r2': r2}
    except Exception as e:
        print(f"  Klaida skaičiuojant metrikas: {str(e)}")
    
    return {'mae': None, 'r2': None}

def evaluate_top10_accuracy(top_predictions, actual_participants, test_df, target_col):
    """
    Įvertina TOP10 prognozių tikslumą.
    
    Args:
        top_predictions (pandas.DataFrame): Geriausios prognozės
        actual_participants (list): Faktinių dalyvių indeksai
        test_df (pandas.DataFrame): Testavimo duomenys
        target_col (str): Tikslo stulpelis
    
    Returns:
        dict: TOP10 prognozių metrikos
    """
    # Vertinimo statistika
    participated_in_top10 = top_predictions['Participated'].sum()
    print(f"\n👥 TOP10 dalyvavimo tikslumas: {participated_in_top10}/10 ({participated_in_top10/10:.0%})")
    
    top10_accuracy = 0
    avg_position_error = None
    correct_top10_count = 0
    
    if participated_in_top10 > 0:
        # Kiek prognozuotų TOP10 pateko į tikrąjį TOP10
        actual_top10_indices = np.argsort(test_df.loc[actual_participants, target_col].values)[:10]
        actual_top10_athletes = [actual_participants[i] for i in actual_top10_indices]
        
        # Sportininkės, kurios buvo prognozuotos TOP10 ir faktiškai pateko į TOP10
        predicted_top10_indices = top_predictions[top_predictions['Participated']]['idx'].tolist()
        correct_top10 = set(predicted_top10_indices) & set(actual_top10_athletes)
        correct_top10_count = len(correct_top10)
        
        top10_accuracy = correct_top10_count / min(10, participated_in_top10)
        print(f"📊 TOP10 pozicijų tikslumas: {correct_top10_count}/{participated_in_top10} ({top10_accuracy:.0%})")
        
        # Vidutinis prognozės poslinkis pozicijomis (kiek pozicijų klydo)
        participated_rows = top_predictions[top_predictions['Participated']]
        if not participated_rows.empty:
            avg_position_error = abs(participated_rows['ActualRank'] - range(1, len(participated_rows) + 1)).mean()
            print(f"📏 Vidutinis pozicijų nuokrypis: {avg_position_error:.1f}")
    
    return {
        'participated_in_top10': participated_in_top10,
        'top10_accuracy': top10_accuracy,
        'correct_top10_count': correct_top10_count,
        'avg_position_error': avg_position_error
    }

def calculate_overall_metrics(results_list):
    """
    Apskaičiuoja bendrą modelio tikslumą pagal visų etapų rezultatus.
    
    Args:
        results_list (list): Visų etapų rezultatai
    
    Returns:
        dict: Bendros modelio metrikos
    """
    if not results_list:
        return {}
    
    # Filtruojame None reikšmes
    dalyvavimo_tikslumai = [r['dalyvavimo_tikslumas'] for r in results_list]
    top10_dalyvavimo_tikslumai = [r['top10_dalyvavimo_tikslumas'] for r in results_list]
    top10_poziciju_tikslumai = [r['top10_poziciju_tikslumas'] for r in results_list if r.get('top10_poziciju_tikslumas', 0) > 0]
    poziciju_nuokrypiai = [r['vid_poziciju_nuokrypis'] for r in results_list if r.get('vid_poziciju_nuokrypis') is not None]
    mae_values = [r['mae'] for r in results_list if r.get('mae') is not None]
    r2_values = [r['r2'] for r in results_list if r.get('r2') is not None]
    
    avg_results = {
        'vid_dalyvavimo_tikslumas': np.mean(dalyvavimo_tikslumai) if dalyvavimo_tikslumai else 0,
        'vid_top10_dalyvavimo_tikslumas': np.mean(top10_dalyvavimo_tikslumai) if top10_dalyvavimo_tikslumai else 0,
        'vid_top10_poziciju_tikslumas': np.mean(top10_poziciju_tikslumai) if top10_poziciju_tikslumai else 0,
        'vid_poziciju_nuokrypis': np.mean(poziciju_nuokrypiai) if poziciju_nuokrypiai else 0,
        'vid_mae': np.mean(mae_values) if mae_values else 0,
        'vid_r2': np.mean(r2_values) if r2_values else 0,
    }
    
    return avg_results