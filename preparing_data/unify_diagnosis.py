import os
import re
import pandas as pd
import numpy as np


# ----------------------------
# 1) Canonical mapping rules
#    Expand this as you find new variants
# ----------------------------
MAPPING = {
    # normal / premature
    "normal": "normal",
    "premature": "premature",
    "preterm": "premature",

    # HIE
    "hie": "hypoxic ischemic encephalopathy",
    "hypoxic ischemic encephalopathy": "hypoxic ischemic encephalopathy",

    # hydrocephalus variants
    "hydrocephalus": "hydrocephalus",
    "vp shunt hydrocephalus": "hydrocephalus",

    # hemorrhage variants
    "subdurale hemorrhage": "subdural hemorrhage",
    "subdural hemorrhage": "subdural hemorrhage",
    "subdural hygroma": "subdural hygroma",
    "subarachnoid hemorrhage": "subarachnoid hemorrhage",
    "subararachnoidal hemorrhage": "subarachnoid hemorrhage",
    "intracerebral hemorrhage": "intracerebral hemorrhage",
    "intraventricular hemorrhage": "intraventricular hemorrhage",

    # atrophy variants
    "cerebral atrophy": "cerebral atrophy",
    "cerebellar atrophy": "cerebellar atrophy",
    "cerebral/cerebellar atrophy": "cerebral + cerebellar atrophy",
    "cerebral + cerebellar atrophy": "cerebral + cerebellar atrophy",

    # edema variants
    "brain edema": "edema",
    "cerebral edema": "edema",
    "edema": "edema",
    "hypoxic ischemic encephalopathy + cerebral edema":
        "hypoxic ischemic encephalopathy + edema",

    # infection
    "meningitis": "meningitis",
    "congenital cmv infection": "congenital cmv",
    "congenital cmv": "congenital cmv",

    # infarct
    "infarct": "infarct",

    # malformations / structural
    "corpus callosum agenesis": "corpus callosum agenesis",
    "arnold chiari malformation": "arnold-chiari malformation",
    "arnold-chiari malformation": "arnold-chiari malformation",
    "dandy-walker continuum": "dandy-walker continuum",
    "encephalocele": "encephalocele",
    "meningocele": "meningocele",
    "septooptic dysplasia": "septooptic dysplasia",
    "septooptic dysplasy": "septooptic dysplasia",
    "craniosynostosis": "craniosynostosis",
    "crouzon syndrome": "crouzon syndrome",
    "heterotopia": "heterotopia",
    "polymicroglia": "polymicrogyria",
    "polymicrogyria": "polymicrogyria",
    "focal cortical dysplasia": "focal cortical dysplasia",

    # tumors
    "tumor": "tumor",
    "ependymom": "ependymoma",
    "ependymoma": "ependymoma",
    "pons glioma": "pons glioma",
    "pylocytic astrocytoma": "pilocytic astrocytoma",
    "pilocytic astrocytoma": "pilocytic astrocytoma",
    "medulloblastoma": "medulloblastoma",
    "neuroectodermal tumor": "neuroectodermal tumor",
    "dnet": "dnet",
    "gmb": "glioblastoma",
    "glioblastoma": "glioblastoma",

    # metabolic / genetic
    "leigh syndrome": "leigh syndrome",
    "mitochondriopathy": "mitochondriopathy",
    "neurofibromatosis type 1": "neurofibromatosis type 1",
    "nf1": "neurofibromatosis type 1",
    "nf 1": "neurofibromatosis type 1",

    # skull shape
    "macrocephaly": "macrocephaly",
    "plagiocephaly": "plagiocephaly",
    "brachycephaly": "brachycephaly",

    # other
    "motion artifact": "motion artifact",
    "postoperative": "postoperative",
    "encephalomalacia": "encephalomalacia",
    "searing injuries": "searing injuries",
    "melanin deposition": "melanin deposition",
    "polyneuropathy": "polyneuropathy",
    "sturge-weber syndrome": "sturge-weber syndrome",
}

# ----------------------------
# 2) Text cleanup helpers
# ----------------------------
def basic_normalize(text: str) -> str:
    """
    Lowercase, strip, unify separators, collapse whitespace.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # unify separators to commas
    text = re.sub(r"[;/]+", ",", text)
    text = re.sub(r"[/]+", ",", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text


def fix_common_typos(term: str) -> str:
    """
    Light typo/synonym fixes BEFORE mapping.
    """
    t = term.lower().strip()

    # spelling / language variants
    t = t.replace("oedem", "edema")
    t = t.replace("oedema", "edema")
    t = t.replace("edemae", "edema")               # fix stray plural/typo
    t = t.replace("artefact", "artifact")
    t = t.replace("hemorrhagy", "hemorrhage")
    t = t.replace("subdurale", "subdural")         # fix subdurale
    t = t.replace("hygrom", "hygroma")
    t = t.replace("gliom", "glioma")
    t = t.replace("astrocytom", "astrocytoma")
    t = t.replace("dysplasy", "dysplasia")
    t = t.replace("encephalomalasy", "encephalomalacia")
    t = t.replace("mithocond", "mitochond")        # mithocondriopathy -> mitochondriopathy

    # clean spacing around parentheses
    t = re.sub(r"\s*\(\s*", " (", t)
    t = re.sub(r"\s*\)\s*", ")", t)

    return t.strip()


def map_to_canonical(term: str) -> str:
    """
    Map a cleaned term to canonical form using MAPPING.
    If not found, return term as fallback.
    """
    return MAPPING.get(term, term)


def normalize_diagnosis_entry(diag_text: str) -> str:
    """
    Full normalization for a single row's diagnosis field:
    - basic normalize
    - split into parts
    - fix typos
    - map to canonical
    - de-duplicate
    - join with ' + '
    """
    diag_text = basic_normalize(diag_text)
    if not diag_text:
        return ""

    # split by comma into parts
    raw_parts = [p.strip() for p in diag_text.split(",") if p.strip()]

    cleaned_parts = []
    for p in raw_parts:
        p = fix_common_typos(p)
        p = map_to_canonical(p)
        cleaned_parts.append(p)

    # remove duplicates, keep stable order
    seen = set()
    unique_parts = []
    for p in cleaned_parts:
        if p not in seen:
            unique_parts.append(p)
            seen.add(p)

    return " + ".join(unique_parts)


# ----------------------------
# 3) Main pipeline
# ----------------------------
def clean_diagnoses(
    meta_path="dataset/meta.csv",
    out_csv="dataset/meta_cleaned.csv",
    report_path="dataset/diagnosis_cleaning_report.txt"
):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path, sep=";")

    if "diagnosis" not in df.columns:
        raise ValueError("Column 'diagnosis' not found in meta.csv")

    # original counts
    original_counts = df["diagnosis"].fillna("NA").value_counts()

    # clean
    df["diagnosis_clean"] = df["diagnosis"].apply(normalize_diagnosis_entry)

    # cleaned counts
    cleaned_counts = df["diagnosis_clean"].fillna("NA").value_counts()

    # save cleaned csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, sep=";", index=False)

    # write report (alphabetical ordering)
    lines = []
    lines.append("=== DIAGNOSIS CLEANING REPORT ===\n")
    lines.append(f"Input: {meta_path}")
    lines.append(f"Output CSV: {out_csv}\n")

    original_sorted = original_counts.sort_index()
    cleaned_sorted = cleaned_counts.sort_index()

    lines.append("---- ORIGINAL VALUE COUNTS (alphabetical) ----")
    lines.append(original_sorted.to_string())
    lines.append("\n---- CLEANED VALUE COUNTS (alphabetical) ----")
    lines.append(cleaned_sorted.to_string())

    # show examples of changes (sorted)
    changed = df[df["diagnosis"].fillna("") != df["diagnosis_clean"].fillna("")]
    changed_sorted = changed.sort_values("diagnosis_clean").head(50)

    lines.append("\n---- EXAMPLES OF CHANGES (first 50, alphabetical) ----")
    for _, row in changed_sorted.iterrows():
        lines.append(f"{row['diagnosis']}  -->  {row['diagnosis_clean']}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Cleaned CSV saved to: {out_csv}")
    print(f"✅ Report saved to: {report_path}")


if __name__ == "__main__":
    clean_diagnoses()

