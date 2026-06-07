"""Kaggle Notebook baseline.

Copy this file into a Kaggle Notebook cell or upload it as a script.
It trains on the visible train.csv available at scoring time and writes
submission.csv in the current working directory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
DATA_DIR = Path("/kaggle/input/neurips-open-polymer-prediction-2025")
ATOM_PATTERN = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOFPSIKHbcno]")


try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


def clean_smiles(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def select_smiles(x):
    if isinstance(x, pd.DataFrame):
        return x["SMILES"].map(clean_smiles)
    return pd.Series(x).map(clean_smiles)


def numeric_features(values):
    rows = []
    for value in values:
        smiles = clean_smiles(value)
        atoms = ATOM_PATTERN.findall(smiles)
        row = {
            "smiles_len": float(len(smiles)),
            "atom_count_text": float(len(atoms)),
            "branch_open": float(smiles.count("(")),
            "branch_close": float(smiles.count(")")),
            "ring_digits": float(sum(ch.isdigit() for ch in smiles)),
            "double_bonds": float(smiles.count("=")),
            "triple_bonds": float(smiles.count("#")),
            "aromatic_chars": float(sum(ch in "bcnops" for ch in smiles)),
            "star_count": float(smiles.count("*")),
            "bracket_count": float(smiles.count("[") + smiles.count("]")),
        }
        for atom in ["C", "N", "O", "F", "S", "Cl", "Br", "Si"]:
            row[f"atom_{atom}_text"] = float(atoms.count(atom))

        rdkit_row = {
            "rdkit_valid": 0.0,
            "mol_wt": np.nan,
            "heavy_atom_count": np.nan,
            "num_rings": np.nan,
            "num_aromatic_rings": np.nan,
            "tpsa": np.nan,
            "logp": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "rotatable_bonds": np.nan,
            "fraction_csp3": np.nan,
        }
        if RDKIT_AVAILABLE and smiles:
            mol = Chem.MolFromSmiles(smiles.replace("*", "C"))
            if mol is not None:
                rdkit_row.update(
                    {
                        "rdkit_valid": 1.0,
                        "mol_wt": float(Descriptors.MolWt(mol)),
                        "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),
                        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
                        "num_aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                        "logp": float(Crippen.MolLogP(mol)),
                        "hbd": float(Lipinski.NumHDonors(mol)),
                        "hba": float(Lipinski.NumHAcceptors(mol)),
                        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                    }
                )
        row.update(rdkit_row)
        rows.append(row)
    return pd.DataFrame(rows)


def make_features():
    tfidf = Pipeline(
        [
            ("select", FunctionTransformer(select_smiles, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 6),
                    min_df=2,
                    max_features=50000,
                    lowercase=False,
                ),
            ),
        ]
    )
    numeric = Pipeline(
        [
            ("select_numeric", FunctionTransformer(lambda x: numeric_features(select_smiles(x)), validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("to_sparse", FunctionTransformer(sparse.csr_matrix, accept_sparse=True)),
        ]
    )
    return FeatureUnion([("tfidf", tfidf), ("numeric", numeric)])


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    submission = pd.DataFrame({"id": test["id"].to_numpy()})
    for target in TARGETS:
        mask = train[target].notna()
        model = Pipeline([("features", make_features()), ("regressor", Ridge(alpha=5.0))])
        model.fit(train.loc[mask, ["SMILES"]], train.loc[mask, target])
        submission[target] = model.predict(test[["SMILES"]])

    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission = submission[["id", *TARGETS]]
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
