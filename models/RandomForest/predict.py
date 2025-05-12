import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Įkeliame duomenis
df = pd.read_csv("data/female_athletes_cleaned_final.csv")

# Sportininkės vardas
name_col = "FullName"

# Renkame visus rezultatų stulpelius
result_cols = [col for col in df.columns if col.strip().endswith(") W")]

# Sukuriame mapping stulpelis -> data (YYYY-MM)
date_map = {}
for col in result_cols:
    try:
        parts = col.split()
        year = int(parts[0])
        month = int(parts[1])
        date_map[col] = f"{year:04d}-{month:02d}"
    except:
        continue

# Sezonų logika
train_cols = [col for col in result_cols if date_map[col] < "2024-06"]
val_cols   = [col for col in result_cols if "2024-06" <= date_map[col] <= "2024-12"]
test_cols  = [col for col in result_cols if date_map[col] >= "2025-01"]

# Pasirenkame konkretų etapą
target_col = "2025 10 (7.5  Sprint Competition) W"

if target_col not in result_cols:
    raise ValueError(f"{target_col} nėra rezultatų stulpeliuose.")

# X ir y
X = df[train_cols]
y = df[target_col]

# Modelis
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['prognoze'] = model.predict(X)

# Tikros vietos (pagal tikrą rezultatą)
if df[target_col].isnull().sum() < len(df):
    df['Tikroji vieta'] = df[target_col].rank(method='min')
    df['Tikroji vieta'] = df['Tikroji vieta'].apply(
        lambda x: int(x) if not pd.isnull(x) else "Nedalyvavo"
    )
else:
    df['Tikroji vieta'] = "Nedalyvavo"

# Prognozuotas reitingas
df['Prognozuota vieta'] = df['prognoze'].rank(method='min')

# Išrenkame top 10 pagal prognozę
top10 = df.nsmallest(10, 'prognoze')[[name_col, 'prognoze', 'Prognozuota vieta', 'Tikroji vieta']]

# Išvedimas
print(f"\n📊 TOP 10 prognozė vs tikras rezultatas ({target_col}):")
print(top10.to_string(index=False))
