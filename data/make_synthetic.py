"""Generate a synthetic symptom dataset for three pharyngitis types.

The prevalence profiles below are illustrative only, loosely inspired by textbook
presentations, and are NOT derived from any real patient records.
"""
import numpy as np
import pandas as pd

SYMPTOMS = ["feeling cold", "sore throat", "cephalalgia", "myalgia", "rhinorrhea", "diarrhea",
            "vomit", "cough", "pyrexia", "rigors", "lethargy", "sputum", "abdominal pain", "vertigo",
            "sternutation", "eye infection", "otalgia", "nasal congestion", "asthma", "anorexia"]

#                  P(symptom | class)
PROFILE = {
    # 0: viral-like — coryza, cough, sneezing, conjunctivitis
    0: dict(zip(SYMPTOMS, [.35, .75, .45, .40, .80, .10, .05, .75, .35, .10, .45, .40, .05, .10, .70, .35, .15, .75, .10, .20])),
    # 1: bacterial-like — high fever, rigors, no cough, abdominal pain in children
    1: dict(zip(SYMPTOMS, [.55, .95, .55, .35, .10, .05, .25, .10, .85, .55, .50, .10, .30, .10, .05, .05, .35, .10, .05, .55])),
    # 2: mixed / other — GI symptoms, lethargy, moderate fever
    2: dict(zip(SYMPTOMS, [.40, .70, .35, .55, .35, .45, .40, .40, .55, .20, .70, .25, .40, .30, .20, .10, .20, .35, .15, .45])),
}
CLASS_SIZES = {0: 180, 1: 140, 2: 80}


def make(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, n in CLASS_SIZES.items():
        p = np.array([PROFILE[label][s] for s in SYMPTOMS])
        X = (rng.random((n, len(SYMPTOMS))) < p).astype(int)
        X[:, SYMPTOMS.index("sore throat")] |= (rng.random(n) < .5).astype(int)  # most patients present with a sore throat
        rows.append(pd.DataFrame(X, columns=SYMPTOMS).assign(pharyngitis_type=label))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "patient", [f"P{i:03d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = make()
    df.to_csv("synthetic_patients.csv", index=False)
    print(df.shape, df["pharyngitis_type"].value_counts().to_dict())
