import pandas as pd

# Load metadata
df = pd.read_csv("dataset/meta.csv", sep=";")

# Convert types
df["age_corrected"] = pd.to_numeric(df["age_corrected"], errors="coerce")
df["doctor_predicted_age"] = pd.to_numeric(df["doctor_predicted_age"], errors="coerce")

# Determine incorrect predictions
df["incorrect"] = df["doctor_predicted_age"] != df["age_corrected"]

# Count incorrect per group
incorrect_by_group = df.groupby("group")["incorrect"].sum()
total_by_group = df.groupby("group")["incorrect"].count()

# Create a summary table
summary = pd.DataFrame({
    "incorrect": incorrect_by_group,
    "total": total_by_group,
    "percent_incorrect (%)": (incorrect_by_group / total_by_group * 100).round(2)
})

print(summary)

# RESULTS by age
#        incorrect  total  percent_incorrect (%)
# group                                         
# test           5    123                   4.07
# train         17    710                   2.39

# RESULTS by corrected age
#        incorrect  total  percent_incorrect (%)
# group                                         
# test           2    123                   1.63
# train         11    710                   1.55