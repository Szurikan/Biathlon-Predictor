import pandas as pd
import sqlite3

def load_and_clean_columns(input_path):
    df = pd.read_csv(input_path)
    columns_to_drop = [col for col in df.columns if col.strip().endswith("M")]
    df = df.drop(columns=columns_to_drop)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~(df == '').all()]
    return df

def remove_empty_rows(df, column_groups):
    flattened_ET = sum(column_groups[:3], [])
    columns_ET = [col for col in flattened_ET if col in df.columns]
    df = df[~df[columns_ET].replace('', pd.NA).isna().all(axis=1)].copy()
    return df

def fill_group_means(df, column_groups):
    for group in column_groups:
        group_cols = [col for col in group if col in df.columns]
        if not group_cols:
            continue
        for col in group_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(',', '.', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
        row_mean = df[group_cols].mean(axis=1)
        for col in group_cols:
            df[col] = df[col].fillna(row_mean)
    return df

def fill_result_columns(df, last_group_col):
    if last_group_col in df.columns:
        result_columns = df.columns[df.columns.get_loc(last_group_col) + 1:]
        for col in result_columns:
            df[col] = (
                df[col].astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(',', '.', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fill_missing_with_personal_average(df):
    competition_columns = [col for col in df.columns if col.startswith("202")]
    for idx, row in df.iterrows():
        values = row[competition_columns]
        mean_val = values[values.notna()].mean()
        if pd.isna(mean_val):
            continue
        for col in competition_columns:
            if pd.isna(df.at[idx, col]):
                df.at[idx, col] = mean_val
    return df

def final_cleaning_and_encoding(df):
    df["FullName"] = df["FullName"].fillna("missing")
    df["Nation"] = df["Nation"].fillna("missing")
    df = pd.get_dummies(df, columns=["Nation"], prefix="Nation")
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Galutinai išvalytas failas išsaugotas kaip: {output_path}")

def encode_competition_participation(df, binary_output_path):
    competition_columns = [col for col in df.columns if col.startswith("202")]
    df_binary = df.copy()
    df_binary[competition_columns] = df_binary[competition_columns].notna().astype(int)
    df_binary.to_csv(binary_output_path, index=False)
    print(f"Failas su užkoduotais varžybų duomenimis išsaugotas kaip: {binary_output_path}")
    return df_binary

def save_to_database(df_cleaned, df_binary, db_path="data/athletes_data.db"):
    conn = sqlite3.connect(db_path)
    df_cleaned.to_sql("cleaned_data", conn, if_exists="replace", index=False)
    df_binary.to_sql("binary_data", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Duomenys išsaugoti SQLite duomenų bazėje: {db_path}")
