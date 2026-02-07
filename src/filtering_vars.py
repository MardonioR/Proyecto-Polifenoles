import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
# ------------------------------------------------------------------------------------
# TECNICAS DE FILTRADO DE VARIABLES
# ------------------------------------------------------------------------------------

# ----------------------------
# Inferir Tipo De Variables
# ----------------------------

def infer_var_types(df: pd.DataFrame, feature_cols: List[str], max_unique_for_cat: int = 10) -> Dict[str, str]:
    var_types = {}
    for c in feature_cols:
        s = df[c]
        nunique = s.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(s):
            # Para numericas tipo binarias
            if nunique <= 2:
                var_types[c] = 'Binaria'
            elif nunique <= max_unique_for_cat:
                # Para variables ordinales
                if (s.dropna() % 1 == 0).all():
                    var_types[c] = 'Ordinal'
                else:
                    var_types[c] = 'Numerica'
            else:
                var_types[c] = 'Numerica'
        else:
            # Los no numéricos se tratarán como categóricos
            var_types[c] = 'Categorica'
    return var_types

# ----------------------------
# Z-Score Transformador
# ----------------------------

def zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)
    return (X - mu) / sd

# ----------------------------------------------------
# Seleccionador De Variables Por Umbral De Varianza
# ----------------------------------------------------
def variance_filter(
    df_frame: pd.DataFrame,
    feature_cols: List[str],
    var_types: Dict[str, str],
    continuous_var_threshold: float = 0.01,
    binary_minority_freq: float = 0.05,
    ordinal_min_unique: int = 3,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns:
      kept_features, report_df
    """
    report = []

    cont_cols = [c for c in feature_cols if var_types.get(c) == "Numerica"]
    ord_cols  = [c for c in feature_cols if var_types.get(c) == "Ordinal"]
    bin_cols  = [c for c in feature_cols if var_types.get(c) == "Binaria"]
    nom_cols  = [c for c in feature_cols if var_types.get(c) == "Categorica"]

    kept = set()

    # Numerica: calcula z-score despues aplica el umbral de varianza
    if cont_cols:
        Xc = df_frame[cont_cols].astype(float)
        Xc_z = zscore_df(Xc)
        variances = Xc_z.var(axis=0, skipna=True, ddof=0)

        for c, v in variances.items():
            keep = (v >= continuous_var_threshold)
            report.append((c, "Numerica", "zvar", v, keep))
            if keep:
                kept.add(c)

    # Ordinal: Requiere al menos cierto número de diferentes clases
    for c in ord_cols:
        nunique = df_frame[c].nunique(dropna=True)
        keep = nunique >= ordinal_min_unique
        report.append((c, "Ordinal", "nunique", float(nunique), keep))
        if keep:
            kept.add(c)

    # Binaria: Prevalecen si la clase minoritaria tiene al menos x% de participación
    for c in bin_cols:
        vc = df_frame[c].value_counts(dropna=True, normalize=True)
        top_freq = vc.iloc[0] if len(vc) else 1.0
        minority = 1.0 - top_freq
        keep = minority >= binary_minority_freq
        report.append((c, "Binaria", "minority_freq", float(minority), keep))
        if keep:
            kept.add(c)

    # Categorica Se quedan si tienen al menos dos clases
    for c in nom_cols:
        nunique = df_frame[c].nunique(dropna=True)
        keep = nunique >= 2
        report.append((c, "Categorica", "nunique", float(nunique), keep))
        if keep:
            kept.add(c)

    report_df = pd.DataFrame(report, columns=["feature", "type", "metric", "value", "keep"])
    kept_features = [c for c in feature_cols if c in kept]
    return kept_features, report_df

# ----------------------------
# Correlación
# ----------------------------

def correlation_filter_bdnf(
    df_frame: pd.DataFrame,
    feature_cols: List[str],
    var_types: Dict[str, str],
    target_col: str = "bdnf",
    corr_threshold: float = 0.90,
    method: str = "spearman",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns kept_features, decisions_df
    """
    # dejamos por default elegibles al tratamiento las continuas y ordinales
    eligible = [c for c in feature_cols if var_types.get(c) in ("Numerica", "Ordinal")]

    # Asegurandonos que existe la variable target
    if target_col not in df_frame.columns:
        raise ValueError(f"{target_col} no encontrada.")
    if target_col in eligible:
        pass

    # Calculando las correlaciones
    X = df_frame[eligible].copy()
    corr = X.corr(method=method, numeric_only=True).abs()

    # Correlacion con variable objetivo
    bdnf_corr = df_frame[eligible + [target_col]].corr(method=method, numeric_only=True)[target_col].abs()
    bdnf_corr = bdnf_corr.drop(labels=[target_col], errors="ignore")

    to_drop = set()
    decisions = []

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            r = corr.iloc[i, j]
            if pd.isna(r) or r < corr_threshold:
                continue

            if a in to_drop or b in to_drop:
                continue

            if a == target_col:
                to_drop.add(b)
                decisions.append((a, b, r, "keep_target", a, b))
                continue
            if b == target_col:
                to_drop.add(a)
                decisions.append((a, b, r, "keep_target", b, a))
                continue

            ca = bdnf_corr.get(a, np.nan)
            cb = bdnf_corr.get(b, np.nan)

            if pd.isna(ca) and pd.isna(cb):
                miss_a = df_frame[a].isna().mean()
                miss_b = df_frame[b].isna().mean()
                keep = a if miss_a <= miss_b else b
            else:
                # Eligiendo la más asociada a bdnf
                keep = a if (ca >= cb) else b

            drop = b if keep == a else a
            to_drop.add(drop)
            decisions.append((a, b, r, "keep_higher_corr_with_target", keep, drop))

    kept = [c for c in feature_cols if c not in to_drop]
    decisions_df = pd.DataFrame(decisions, columns=["var_a", "var_b", "abs_corr", "rule", "kept", "dropped"])
    return kept, decisions_df