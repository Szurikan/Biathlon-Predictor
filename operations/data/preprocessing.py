import pandas as pd

# Nurodyk įvesties ir išvesties failų kelius
input_path = "data/female_athletes_2425_full_stats_with_ranks.csv"
output_path = "data/female_athletes_cleaned_final.csv"

# Įkelti CSV failą
df = pd.read_csv(input_path)

# Pašalinti stulpelius, kurių pavadinimai baigiasi " M"
columns_to_drop = [col for col in df.columns if col.strip().endswith("M")]
df_cleaned = df.drop(columns=columns_to_drop)

# Pašalinti stulpelius, kuriuose visos reikšmės yra NaN arba tuščios eilutės
df_cleaned = df_cleaned.dropna(axis=1, how='all')
df_cleaned = df_cleaned.loc[:, ~(df_cleaned == '').all()]

# Apibrėžti grupes stulpelių (nuo E iki X)
column_groups = [
    ['StatShooting_24_25', 'StatShooting_23_24', 'StatShooting_22_23', 'StatShooting_21_22'],
    ['Prone_24_25', 'Prone_23_24', 'Prone_22_23', 'Prone_21_22'],
    ['Standing_24_25', 'Standing_23_24', 'Standing_22_23', 'Standing_21_22'],
    ['Skiing_24_25', 'Skiing_23_24', 'Skiing_22_23', 'Skiing_21_22'],
    ['SkiKMB_24_25', 'SkiKMB_23_24', 'SkiKMB_22_23', 'SkiKMB_21_22']
]

# Pašalinti eilutes, kur visos reikšmės nuo E iki T yra tuščios
flattened_ET = sum(column_groups[:3], [])
columns_ET = [col for col in flattened_ET if col in df_cleaned.columns]
df_cleaned = df_cleaned[~df_cleaned[columns_ET].replace('', pd.NA).isna().all(axis=1)].copy()

# Užpildyti trūkstamas reikšmes grupių vidurkiais (E iki X)
for group in column_groups:
    group_cols = [col for col in group if col in df_cleaned.columns]
    if not group_cols:
        continue

    for col in group_cols:
        df_cleaned[col] = (
            df_cleaned[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.strip()
        )
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    row_mean = df_cleaned[group_cols].mean(axis=1)
    for col in group_cols:
        df_cleaned[col] = df_cleaned[col].fillna(row_mean)

# Užpildyti trūkstamas reikšmes varžybų rezultatuose (nuo Y iki galo)
last_group_col = 'SkiKMB_21_22'
if last_group_col in df_cleaned.columns:
    result_columns = df_cleaned.columns[df_cleaned.columns.get_loc(last_group_col) + 1:]
    for col in result_columns:
        df_cleaned[col] = (
            df_cleaned[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.strip()
        )
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        max_rank = df_cleaned[col].max(skipna=True)
        df_cleaned[col] = df_cleaned[col].fillna(max_rank + 1)

# Užpildyti trūkstamas reikšmes "FullName" ir "Nation"
df_cleaned["FullName"] = df_cleaned["FullName"].fillna("missing")
df_cleaned["Nation"] = df_cleaned["Nation"].fillna("missing")

# One-hot encoding Nation
df_cleaned = pd.get_dummies(df_cleaned, columns=["Nation"], prefix="Nation")

# Išsaugoti galutinį CSV
df_cleaned.to_csv(output_path, index=False)
print(f"Galutinai išvalytas failas išsaugotas kaip: {output_path}")
