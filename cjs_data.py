import pandas as pd

FILE = "PR AEROSOLS DATA/OneDrive_2_02-12-2025/Aerosol optical properties_CSJ (1).xlsx"
SHEET = "Sheet1"

df = pd.read_excel(FILE, sheet_name=SHEET)

# First row is a header-like row (has BsB0_S11 etc.), skip it
df = df.iloc[1:].copy()

# Date column in your file is "Unnamed: 0"
df = df.rename(columns={"Unnamed: 0": "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Columns of interest
cols = ["SAE_PM10", "AAE_PM10", "SAE_PM1", "AAE_PM1"]

# Convert to numeric and replace common missing flags
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[cols] = df[cols].replace([-999, -999.0, -9999, -9999.0], pd.NA)

# Count unique days with BOTH SAE and AAE
pm10_days_both = df.loc[df["SAE_PM10"].notna() & df["AAE_PM10"].notna(), "date"].dt.normalize().nunique()
pm1_days_both  = df.loc[df["SAE_PM1"].notna()  & df["AAE_PM1"].notna(),  "date"].dt.normalize().nunique()

print(f"PM10: unique days with valid SAE & AAE = {int(pm10_days_both)}")
print(f"PM1 : unique days with valid SAE & AAE = {int(pm1_days_both)}")
