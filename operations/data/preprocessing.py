import pandas as pd
import numpy as np

def categorize_columns(df):
    """
    Suskirstoma duomenų stulpelius pagal jų tipus.
    
    Args:
        df (pandas.DataFrame): Įkelti duomenys
        
    Returns:
        tuple: (race_cols, shooting_cols, skiing_cols, kmb_cols, starts_cols, birth_year_col, sorted_race_cols)
    """
    print("[4] Išrenkami stulpeliai pagal tipus...")
    
    # Išrenkame visus stulpelius pagal kategorijas
    race_cols = [col for col in df.columns if "(" in col and ")" in col]
    shooting_cols = [col for col in df.columns if "StatShooting" in col]
    skiing_cols = [col for col in df.columns if "Skiing" in col]
    kmb_cols = [col for col in df.columns if "SkiKMB" in col]
    starts_cols = [col for col in df.columns if "Starts_" in col]
    birth_year_col = "BirthYear" if "BirthYear" in df.columns else None
    
    # Rūšiuojam varžybų stulpelius chronologiškai
    sorted_race_cols = sorted(race_cols, key=lambda x: (int(x[:4]), int(x[5:7])))
    
    return race_cols, shooting_cols, skiing_cols, kmb_cols, starts_cols, birth_year_col, sorted_race_cols

def calculate_age(df, target_col, birth_year_col="BirthYear"):
    """
    Apskaičiuoja sportininkių amžių.
    
    Args:
        df (pandas.DataFrame): Duomenų rinkinys
        target_col (str): Tikslo stulpelis (naudojamas metams nustatyti)
        birth_year_col (str): Gimimo metų stulpelis
        
    Returns:
        pandas.DataFrame: DataFrame su pridėtu amžiaus stulpeliu
    """
    if birth_year_col not in df.columns:
        return df
    
    df = df.copy()
    df[birth_year_col] = pd.to_numeric(df[birth_year_col], errors='coerce')
    current_year = int(target_col[:4])
    df["Age"] = current_year - df[birth_year_col]
    df.drop(columns=[birth_year_col], errors='ignore', inplace=True)
    
    return df

def convert_percentage_columns(df):
    """
    Konvertuoja procentinius stulpelius į skaitines reikšmes.
    
    Args:
        df (pandas.DataFrame): Duomenų rinkinys
        
    Returns:
        pandas.DataFrame: DataFrame su konvertuotais procentiniais stulpeliais
    """
    df = df.copy()
    
    for col in df.columns:
        if isinstance(df[col].dtype, object) and df[col].astype(str).str.contains('%').any():
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).astype(float)
            
    return df

def fill_missing_values(df):
    """
    Užpildo trūkstamas reikšmes.
    
    Args:
        df (pandas.DataFrame): Duomenų rinkinys
        
    Returns:
        pandas.DataFrame: DataFrame su užpildytomis trūkstamomis reikšmėmis
    """
    df = df.copy()
    
    for col in df.columns:
        if col in ["FullName", "Nation"]:
            df[col] = df[col].fillna("missing")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_knowable_columns(df, target_col, sorted_race_cols, shooting_cols, skiing_cols, 
                             kmb_cols, starts_cols, birth_year_col, is_column_available_func):
    """
    Paruošia stulpelius, kurie būtų žinomi prieš tikslo etapą.
    
    Args:
        df (pandas.DataFrame): Įkelti duomenys
        target_col (str): Tikslo stulpelis
        sorted_race_cols (list): Surūšiuoti varžybų stulpeliai
        shooting_cols (list): Šaudymo statistikos stulpeliai
        skiing_cols (list): Slidinėjimo statistikos stulpeliai
        kmb_cols (list): KMB statistikos stulpeliai
        starts_cols (list): Startų statistikos stulpeliai
        birth_year_col (str): Gimimo metų stulpelis
        is_column_available_func (function): Funkcija, tikrinanti stulpelių pasiekiamumą
        
    Returns:
        tuple: (current_df, prev_cols) - parengti duomenys ir ankstesnių varžybų stulpeliai
    """
    # Išrenkame tik stulpelius, kurie būtų žinomi PRIEŠ šį etapą
    prev_cols = [col for col in sorted_race_cols if col < target_col]
    
    # Sukuriame laikiną DataFrame su tik žinomais duomenimis iki šio etapo
    knowable_cols = ["FullName", "Nation"]
    if birth_year_col and birth_year_col in df.columns:
        knowable_cols.append(birth_year_col)
    
    knowable_cols += prev_cols
    knowable_cols += [col for col in shooting_cols if is_column_available_func(col, target_col)]
    knowable_cols += [col for col in skiing_cols if is_column_available_func(col, target_col)]
    knowable_cols += [col for col in kmb_cols if is_column_available_func(col, target_col)]
    knowable_cols += [col for col in starts_cols if is_column_available_func(col, target_col)]
    
    # Filtruojame stulpelius, kurie faktiškai egzistuoja DataFrame
    knowable_cols = [col for col in knowable_cols if col in df.columns]
    
    current_df = df[knowable_cols].copy()
    
    return current_df, prev_cols

def create_statistics_features(df, target_col, shooting_cols, skiing_cols, kmb_cols, is_column_available_func):
    """
    Sukuria statistikos požymius iš istorinių duomenų.
    
    Args:
        df (pandas.DataFrame): Duomenų rinkinys
        target_col (str): Tikslo stulpelis
        shooting_cols (list): Šaudymo statistikos stulpeliai
        skiing_cols (list): Slidinėjimo statistikos stulpeliai
        kmb_cols (list): KMB statistikos stulpeliai
        is_column_available_func (function): Funkcija, tikrinanti stulpelių pasiekiamumą
        
    Returns:
        pandas.DataFrame: DataFrame su pridėtais statistikos požymiais
    """
    # Filtruojame statistikos stulpelius, kad naudotume tik tuos, kurie būtų žinomi prieš etapą
    available_shooting_cols = [col for col in df.columns if col in shooting_cols and is_column_available_func(col, target_col)]
    available_skiing_cols = [col for col in df.columns if col in skiing_cols and is_column_available_func(col, target_col)]
    available_kmb_cols = [col for col in df.columns if col in kmb_cols and is_column_available_func(col, target_col)]
    
    # Konvertuojame visus statistinius stulpelius į skaitines reikšmes
    df = df.copy()
    all_stat_cols = available_shooting_cols + available_skiing_cols + available_kmb_cols
    
    for col in all_stat_cols:
        if df[col].dtype == object:
            # Bandome konvertuoti procentus
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            # Konvertuojame į skaitinę reikšmę
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Statistikos vidurkių skaičiavimas
    stats_df = pd.DataFrame(index=df.index)
    
    if available_shooting_cols:
        numeric_shooting = [col for col in available_shooting_cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_shooting:
            stats_df['CurrentAvgShooting'] = df[numeric_shooting].mean(axis=1)
        else:
            stats_df['CurrentAvgShooting'] = np.nan
    else:
        stats_df['CurrentAvgShooting'] = np.nan
        
    if available_skiing_cols:
        numeric_skiing = [col for col in available_skiing_cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_skiing:
            stats_df['CurrentAvgSki'] = df[numeric_skiing].mean(axis=1)
        else:
            stats_df['CurrentAvgSki'] = np.nan
    else:
        stats_df['CurrentAvgSki'] = np.nan
        
    if available_kmb_cols:
        numeric_kmb = [col for col in available_kmb_cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_kmb:
            stats_df['CurrentAvgKMB'] = df[numeric_kmb].mean(axis=1)
        else:
            stats_df['CurrentAvgKMB'] = np.nan
    else:
        stats_df['CurrentAvgKMB'] = np.nan
    
    # Sujungiame su pagrindiniu DataFrame
    result_df = pd.concat([df, stats_df], axis=1)
    
    # Užpildome likusias trūkstamas reikšmes
    result_df['CurrentAvgShooting'] = result_df['CurrentAvgShooting'].fillna(-1)
    result_df['CurrentAvgSki'] = result_df['CurrentAvgSki'].fillna(-1)
    result_df['CurrentAvgKMB'] = result_df['CurrentAvgKMB'].fillna(-1)
    
    print(f"  Statistika paruošta: {len(available_shooting_cols)} šaudymo, {len(available_skiing_cols)} slidinėjimo, {len(available_kmb_cols)} KMB stulpeliai")
    
    return result_df