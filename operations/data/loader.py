import os
import pandas as pd
import sqlite3

def load_data(db_path, table_name="cleaned_data"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def clean_team_events(df):
    """
    Pašalina komandinių varžybų stulpelius.
    
    Args:
        df (pandas.DataFrame): Pradiniai duomenys
    
    Returns:
        pandas.DataFrame: Duomenys be komandinių varžybų stulpelių
    """
    print("[2] Šalinami visiškai tušti ir komandinių varžybų stulpeliai...")
    df = df.copy()
    df.dropna(axis=1, how='all', inplace=True)
    
    relay_keywords = ["relay", "team", "mixed"]
    columns_to_drop = [col for col in df.columns if any(k in col.lower() for k in relay_keywords)]
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df

def extract_identity_columns(df):
    """
    Išskiria identifikavimo stulpelius ir grąžina juos kartu su pagrindine duomenų lentele.
    
    Args:
        df (pandas.DataFrame): Pradiniai duomenys
    
    Returns:
        tuple: (df_bez_id, id_df) - duomenys be perteklinių ID stulpelių ir atskiras ID DataFrame
    """
    print("[3] Išsaugomi identifikaciniai stulpeliai...")
    df = df.copy()
    
    # Išsaugome vardus ir šalis
    id_columns = ["FullName", "Nation"]
    if "IBUId" in df.columns:
        id_columns.append("IBUId")
    
    id_df = df[id_columns].copy()
    
    # Šaliname ID stulpelius, išskyrus FullName ir Nation
    id_cols_to_drop = [col for col in id_columns if col not in ["FullName", "Nation"]]
    if id_cols_to_drop:
        df.drop(columns=id_cols_to_drop, errors='ignore', inplace=True)
    
    return df, id_df