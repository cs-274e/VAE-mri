import os
import pandas as pd
import numpy as np

def analyze_meta(meta_path="dataset/meta.csv", out_path="meta_analysis.txt"):
    # ---------- Load ----------
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find: {meta_path}")

    df = pd.read_csv(meta_path, sep=";")
    
    # ---------- Basic cleanup / typing ----------
    # Strip whitespace in string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Try converting obvious numeric columns if present
    numeric_candidates = ["age", "age_corrected", "doctor_predicted_age"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If age_corrected missing but age exists, keep as is (no imputation)
    
    lines = []
    lines.append("=== METADATA ANALYSIS REPORT ===\n")
    lines.append(f"File: {meta_path}\n")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}\n")
    lines.append("Columns:")
    lines.append(", ".join(df.columns) + "\n")

    # ---------- Missing values ----------
    lines.append("=== MISSING VALUES PER COLUMN ===")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for c in df.columns:
        lines.append(f"{c}: {missing[c]} ({missing_pct[c]}%)")
    lines.append("")

    # ---------- Duplicates ----------
    if "image_id" in df.columns:
        dup_ids = df["image_id"].duplicated().sum()
        lines.append("=== DUPLICATE image_id CHECK ===")
        lines.append(f"Duplicate image_id rows: {dup_ids}\n")

    # ---------- Feature types ----------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    lines.append("=== NUMERIC FEATURES ===")
    lines.append(", ".join(num_cols) if num_cols else "None")
    lines.append("")

    lines.append("=== CATEGORICAL FEATURES ===")
    lines.append(", ".join(cat_cols) if cat_cols else "None")
    lines.append("")

    # ---------- Numeric summaries ----------
    if num_cols:
        lines.append("=== NUMERIC SUMMARY (overall) ===")
        desc = df[num_cols].describe().T
        lines.append(desc.to_string())
        lines.append("")

    # ---------- Categorical summaries ----------
    if cat_cols:
        lines.append("=== CATEGORICAL SUMMARY (overall) ===")
        for c in cat_cols:
            lines.append(f"\n-- {c} --")
            vc = df[c].value_counts(dropna=False)
            lines.append(vc.to_string())
        lines.append("")

    # ---------- Group-wise summaries ----------
    def group_summary(by_col, name):
        if by_col not in df.columns:
            return
        lines.append(f"=== SUMMARY BY {name.upper()} ({by_col}) ===")
        
        # counts
        lines.append("\nCounts:")
        lines.append(df[by_col].value_counts(dropna=False).to_string())

        # numeric by group
        if num_cols:
            lines.append("\nNumeric feature means by group:")
            means = df.groupby(by_col)[num_cols].mean(numeric_only=True)
            lines.append(means.to_string())
            
            lines.append("\nNumeric feature std by group:")
            stds = df.groupby(by_col)[num_cols].std(numeric_only=True)
            lines.append(stds.to_string())

        # categorical by group
        for c in cat_cols:
            if c == by_col:
                continue
            lines.append(f"\nTop values of {c} by {by_col}:")
            tab = df.pivot_table(index=by_col, columns=c, aggfunc="size", fill_value=0)
            lines.append(tab.to_string())
        lines.append("")

    group_summary("group", "split")
    group_summary("diagnosis", "diagnosis")

    # ---------- Prediction error analysis ----------
    if set(["age", "doctor_predicted_age"]).issubset(df.columns):
        lines.append("=== DOCTOR PREDICTION ERROR ===")
        df["prediction_error"] = df["doctor_predicted_age"] - df["age"]
        err_desc = df["prediction_error"].describe()
        lines.append(err_desc.to_string())

        # MAE / RMSE
        mae = df["prediction_error"].abs().mean()
        rmse = np.sqrt((df["prediction_error"] ** 2).mean())
        lines.append(f"\nMAE (mean abs error): {mae:.3f}")
        lines.append(f"RMSE: {rmse:.3f}\n")

        # error by diagnosis/group if present
        for by_col in ["diagnosis", "group"]:
            if by_col in df.columns:
                lines.append(f"Prediction error mean by {by_col}:")
                lines.append(df.groupby(by_col)["prediction_error"].mean().to_string())
                lines.append("")

    # ---------- Correlations ----------
    if len(num_cols) >= 2:
        lines.append("=== CORRELATION MATRIX (numeric features) ===")
        corr = df[num_cols].corr()
        lines.append(corr.to_string())
        lines.append("")

    # ---------- Save report ----------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Analysis written to: {out_path}")


if __name__ == "__main__":
    analyze_meta("dataset/meta.csv", "meta_analysis.txt")
