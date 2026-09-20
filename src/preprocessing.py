import pandas as pd

df = pd.read_csv("data/raw/Raw_Data.csv", low_memory=False)

# Rename first column
df.rename(columns={df.columns[0]: "ID"}, inplace=True)

# Convert date columns
date_columns = [
    "PQ First Sent to Client Date",
    "PO Sent to Vendor Date",
    "Scheduled Delivery Date",
    "Delivered to Client Date",
    "Delivery Recorded Date"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

print(df.columns.tolist())

print("\nData types after date conversion:")
print(df[date_columns].dtypes)

# Convert numerical columns
numeric_columns = [
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ""),
        errors="coerce"
    )

print("\nNumerical column types:")
print(df[numeric_columns].dtypes)

print("\nMissing values in numerical columns:")
print(df[numeric_columns].isnull().sum())

# Check missing values in all columns

missing_values = df.isnull().sum()

print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# Handle missing values

# Categorical columns
df["Shipment Mode"] = df["Shipment Mode"].fillna("Unknown")
df["Dosage"] = df["Dosage"].fillna("Unknown")

# Numerical column
df["Line Item Insurance (USD)"] = df["Line Item Insurance (USD)"].fillna(0)

# Remove columns with almost no useful information
df.drop(columns=["Load_Date", "Source_System"], inplace=True)

print("\nMissing values after treatment:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Detailed inspection of remaining missing values

for col in df.columns:
    missing = df[col].isnull().sum()

    if missing > 0:
        percentage = (missing / len(df)) * 100
        print(f"{col}: {missing} missing ({percentage:}%)")

# Inspect missing values and related information

columns_to_check = [
    "PQ First Sent to Client Date",
    "PO Sent to Vendor Date",
    "Weight (Kilograms)",
    "Freight Cost (USD)"
]

for col in columns_to_check:
    print(f"\n--- {col} ---")
    print("Missing:", df[col].isnull().sum())
    print("Non-missing:", df[col].notnull().sum())
    print("Data type:", df[col].dtype)
    print("Sample values:")
    print(df[col].dropna().head(10).tolist())

    # Create shipment delay

df["Delay_Days"] = (
    df["Delivered to Client Date"] - df["Scheduled Delivery Date"]
).dt.days

# Create binary target
df["Is_Delayed"] = (df["Delay_Days"] > 0).astype(int)

print("\nDelay statistics:")
print(df["Delay_Days"].describe())

print("\nDelayed vs On-time shipments:")
print(df["Is_Delayed"].value_counts())

# Create planned lead time

df["Planned_Lead_Time_Days"] = (
    df["Scheduled Delivery Date"] - df["PO Sent to Vendor Date"]
).dt.days

print("\nPlanned lead time statistics:")
print(df["Planned_Lead_Time_Days"].describe())
# Investigate invalid planned lead times

invalid_lead_time = df[df["Planned_Lead_Time_Days"] < 0]

print("\nNegative planned lead-time records:")
print(invalid_lead_time[
    [
        "ID",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Planned_Lead_Time_Days"
    ]
].head(20))

print("\nNumber of negative lead times:")
print(len(invalid_lead_time))

# Check unusually large planned lead times

print("\nLead times greater than 365 days:")
print(
    (df["Planned_Lead_Time_Days"] > 365).sum()
)

df.loc[df["Planned_Lead_Time_Days"] < 0, "Planned_Lead_Time_Days"] = pd.NA

# Handle invalid planned lead times

df.loc[
    df["Planned_Lead_Time_Days"] < 0,
    "Planned_Lead_Time_Days"
] = pd.NA

print("\nNegative planned lead times after correction:")
print(
    (df["Planned_Lead_Time_Days"] < 0).sum()
)

# Check for duplicate records

print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nDuplicate shipment IDs:")
print(df["ID"].duplicated().sum())
# Identify categorical columns

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical columns:")
print(categorical_columns.tolist())
# Inspect unique values in categorical columns

for col in categorical_columns:
    print(f"\n--- {col} ---")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts(dropna=False).head(20))

# Check categorical columns for inconsistencies

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    print(f"\n{'=' * 60}")
    print(f"Column: {col}")
    print(f"Unique values: {df[col].nunique(dropna=False)}")

    print("\nUnique values:")
    print(df[col].dropna().unique())

    # Check spaces
    original = df[col].dropna().astype(str)
    stripped = original.str.strip()

    print(
        "\nValues with leading/trailing spaces:",
        (original != stripped).sum()
    )

    # Check case/spacing inconsistencies
    normalized = stripped.str.lower()

    if normalized.nunique() < original.nunique():
        print("Possible duplicate categories after normalization")
    else:
        print("No obvious case/spacing duplicates")

# Summary of categorical columns

categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nCategorical column summary:")

for col in categorical_columns:
    print(
        f"{col}: "
        f"{df[col].nunique(dropna=False)} unique values"
    )

# Remove pure identifier columns

identifier_columns = [
    "PQ #",
    "PO / SO #",
    "ASN/DN #"
]

df.drop(columns=identifier_columns, inplace=True)

print("\nRemoved identifier columns:")
print(identifier_columns)

# Clean categorical text values

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()

print("\nCategorical text cleaned.")

# Numerical data summary

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumerical columns:")
print(numeric_columns.tolist())

print("\nNumerical summary:")
print(df[numeric_columns].describe().T)

# Check negative values in numerical columns

print("\nNegative value check:")

for col in numeric_columns:
    negative_count = (df[col] < 0).sum()

    if negative_count > 0:
        print(f"{col}: {negative_count} negative values")

# Check zero values

print("\nZero value check:")

for col in numeric_columns:
    zero_count = (df[col] == 0).sum()

    if zero_count > 0:
        print(f"{col}: {zero_count} zero values")

# Inspect suspicious zero values

zero_check_columns = [
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Weight (Kilograms)",
    "Line Item Insurance (USD)"
]

for col in zero_check_columns:
    zero_rows = df[df[col] == 0]

    print(f"\n--- {col} ---")
    print(f"Zero values: {len(zero_rows)}")

    if len(zero_rows) > 0:
        print(
            zero_rows[
                [
                    "ID",
                    col,
                    "Product Group",
                    "Vendor"
                ]
            ].head(10)
        )
# Treat zero weight as missing because shipment weight should not be zero
df.loc[df["Weight (Kilograms)"] == 0, "Weight (Kilograms)"] = pd.NA
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(
    df[numerical_columns]
    .describe()
    .T
)
# Check potential outliers using the IQR method

numerical_columns = [
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)",
    "Planned_Lead_Time_Days"
]

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"\n--- {col} ---")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {len(outliers)}")
# Check skewness of numerical variables

for col in numerical_columns:
    skewness = df[col].skew()
    print(f"{col}: {skewness:.2f}")

