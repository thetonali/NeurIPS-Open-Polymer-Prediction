from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False


ATOM_PATTERN = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOFPSIKHbcno]")


def clean_smiles(smiles: object) -> str:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return ""
    return str(smiles).strip()


def smiles_basic_features(smiles_values: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value in smiles_values:
        smiles = clean_smiles(value)
        atoms = ATOM_PATTERN.findall(smiles)
        length = len(smiles)
        row = {
            "smiles_len": float(length),
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
        rows.append(row)
    return pd.DataFrame(rows)


def rdkit_descriptors(smiles_values: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value in smiles_values:
        smiles = clean_smiles(value).replace("*", "C")
        row = {
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
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                row.update(
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
        rows.append(row)
    return pd.DataFrame(rows)


def make_numeric_feature_frame(smiles_values: Iterable[str]) -> pd.DataFrame:
    smiles_list = [clean_smiles(value) for value in smiles_values]
    basic = smiles_basic_features(smiles_list)
    rdkit = rdkit_descriptors(smiles_list)
    return pd.concat([basic, rdkit], axis=1)


def _select_smiles_column(x: pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if "SMILES" in x.columns:
            return x["SMILES"].map(clean_smiles)
        return x.iloc[:, 0].map(clean_smiles)
    if isinstance(x, pd.Series):
        return x.map(clean_smiles)
    return pd.Series(x.ravel()).map(clean_smiles)


@dataclass
class FeatureBuilder:
    tfidf_params: dict

    def build(self) -> FeatureUnion:
        ngram_range = tuple(self.tfidf_params.get("ngram_range", [2, 6]))
        tfidf = TfidfVectorizer(
            analyzer=self.tfidf_params.get("analyzer", "char"),
            ngram_range=ngram_range,
            min_df=self.tfidf_params.get("min_df", 2),
            max_features=self.tfidf_params.get("max_features", 50000),
            lowercase=False,
        )

        smiles_text = FunctionTransformer(_select_smiles_column, validate=False)
        numeric_features = FunctionTransformer(
            lambda x: make_numeric_feature_frame(_select_smiles_column(x)),
            validate=False,
        )

        numeric_pipeline = Pipeline(
            [
                ("numeric_features", numeric_features),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("to_sparse", FunctionTransformer(sparse.csr_matrix, accept_sparse=True)),
            ]
        )

        return FeatureUnion(
            [
                ("smiles_tfidf", Pipeline([("selector", smiles_text), ("tfidf", tfidf)])),
                ("numeric", numeric_pipeline),
            ]
        )
