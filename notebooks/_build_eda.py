"""Generates notebooks/01_exploratory_data_analysis.ipynb (run once)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# HealthGuard AI - Exploratory Data Analysis

**Goal:** understand the Diabetes 130-US Hospitals dataset well enough to make
principled cleaning, feature-engineering and modelling decisions for predicting
**30-day hospital readmission**.

This notebook tells the *exploration story*; the productionised logic lives in
`src/` (`data_quality.py`, `preprocessing.py`).""")

code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent / "src"))
import config as C

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 60)

raw = pd.read_csv(C.DATA_RAW).replace("?", np.nan)
print("Shape:", raw.shape)
raw.head(3)""")

md("## 1. Target variable\n\nThe raw label has three classes; we predict the "
   "minority `<30` (readmitted within 30 days).")

code("""ax = raw["readmitted"].value_counts().plot(kind="bar", color="#2c7fb8")
ax.set_title("Raw readmission label distribution"); ax.set_ylabel("encounters")
plt.show()

pos_rate = (raw["readmitted"] == "<30").mean()
print(f"Positive class (<30) prevalence: {pos_rate:.2%}  -> imbalanced problem")""")

md("## 2. Missingness\n\nWhich columns are missing, and *why*? Some 'missing' "
   "values are clinically meaningful (a lab simply wasn't ordered).")

code("""miss = raw.isnull().mean().mul(100).sort_values(ascending=False)
miss = miss[miss > 0]
ax = miss.plot(kind="barh", figsize=(7, 4), color="#fc8d59")
ax.set_title("% missing by column"); ax.set_xlabel("% missing")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
miss.round(2)""")

md("""**Reading this:**
- `weight` (~97%) - too sparse to impute → drop.
- `max_glu_serum`, `A1Cresult` - high "missing", but here missing means the
  **test was not performed** → keep as an explicit category, do not drop.
- `medical_specialty`, `race` → impute an explicit `Unknown` level.""")

md("## 3. The leakage trap: repeat patients\n\nMany patients appear more than "
   "once. Training and testing on the same patient inflates performance.")

code("""print("Total encounters:", len(raw))
print("Unique patients :", raw["patient_nbr"].nunique())
print("Repeat-encounter rows:", int(raw["patient_nbr"].duplicated().sum()))
print("\\n-> We keep only the FIRST encounter per patient (Strack et al., 2014).")""")

md("## 4. What relates to readmission?\n\nA few clinically intuitive signals.")

code("""df = raw.copy()
df["y"] = (df["readmitted"] == "<30").astype(int)

age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
             "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
by_age = df.groupby("age")["y"].mean().reindex(age_order)
ax = by_age.plot(marker="o", figsize=(8,4), color="#2c7fb8")
ax.axhline(df["y"].mean(), ls="--", color="grey", label="overall")
ax.set_title("Readmission rate by age band"); ax.set_ylabel("rate"); ax.legend()
plt.show()""")

code("""# Prior inpatient visits is one of the strongest signals.
tmp = df.copy()
tmp["inp"] = np.where(tmp["number_inpatient"] >= 4, "4+",
                      tmp["number_inpatient"].astype(int).astype(str))
order = ["0","1","2","3","4+"]
by_inp = tmp.groupby("inp")["y"].mean().reindex(order)
ax = by_inp.plot(kind="bar", figsize=(7,4), color="#3182bd")
ax.axhline(df["y"].mean(), ls="--", color="grey", label="overall")
ax.set_title("Readmission rate vs. prior inpatient visits")
ax.set_ylabel("rate"); ax.legend(); plt.show()""")

code("""# Numeric features: distribution and correlation with the target.
num = ["time_in_hospital","num_lab_procedures","num_medications",
       "number_inpatient","number_emergency","number_outpatient",
       "number_diagnoses"]
corr = df[num + ["y"]].corr()["y"].drop("y").sort_values()
ax = corr.plot(kind="barh", figsize=(7,4), color="#2c7fb8")
ax.set_title("Correlation of numeric features with 30-day readmission")
plt.tight_layout(); plt.show()
corr""")

md("""## 5. Takeaways → modelling decisions

1. **Imbalanced** (~11% raw, ~9% after cleaning) → use class weights and
   evaluate with PR-AUC / recall, not accuracy.
2. **Leakage** → first encounter per patient; remove death/hospice discharges.
3. **Meaningful missingness** → keep `A1Cresult` / `max_glu_serum` as categories.
4. **High-cardinality diagnoses** → map ICD-9 to 9 clinical groups.
5. **Strongest signals** are prior utilisation (inpatient/emergency), length of
   stay, age and comorbidity count - later confirmed by SHAP.

These directly motivate `src/preprocessing.py` and `src/modeling.py`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).parent / "01_exploratory_data_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
