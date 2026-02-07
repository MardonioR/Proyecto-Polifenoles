import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from src import functions as f

# ---------------------------------------------------------
# AJUSTE DE MODELO LMM
# ---------------------------------------------------------
def fit_lmm_for_outcome(
        df, outcome, id_col="id", group_col="grupo",
        treatment_col="Treatment", time_col="Time",
        reml=False, method="lbfgs", maxiter=2000):
    
    formula = f"{outcome} ~ C({group_col}) * C({treatment_col}) * C({time_col})"
    model = smf.mixedlm(formula=formula, data=df, groups=df[id_col], re_formula="1")
    res = model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
    return res

# ---------------------------------------------------------
# BUSQUEDA DE PARAMETROS
# ---------------------------------------------------------

def extract_key_terms(res, treatment_col="Treatment", time_col="Time", group_col="Group"):
    """
    Extrae Interacciones (DiD, Triple) Y Efectos Principales (Tiempo, Tratamiento).
    """
    param_names = res.params.index.tolist()
    
    # 1. Triple Interacción (Group x Treat x Time)
    triple_candidates = [
        name for name in param_names 
        if treatment_col in name and time_col in name and group_col in name and name.count(":") >= 2
    ]
    
    # 2. DiD (Treat x Time) - Efecto del Tratamiento en el tiempo (para el Grupo Ref)
    did_candidates = [
        name for name in param_names 
        if treatment_col in name and time_col in name and ":" in name and group_col not in name
    ]

    # --- NUEVO: 3. Efecto Principal Tiempo (Time) ---
    # Representa el cambio Pre->Post en el grupo de Referencia (Control)
    time_candidates = [
        name for name in param_names
        if time_col in name and treatment_col not in name and group_col not in name and ":" not in name
    ]

    # --- NUEVO: 4. Efecto Principal Tratamiento (Treatment) ---
    # Representa la diferencia Intervención vs Control en el momento PRE (Baseline)
    treat_candidates = [
        name for name in param_names
        if treatment_col in name and time_col not in name and group_col not in name and ":" not in name
    ]
    
    # Función auxiliar
    def get_stats(candidates):
        if not candidates:
            return pd.DataFrame() 
        term = candidates[0] 
        return pd.DataFrame({
            "beta": [res.params[term]],
            "pvalue": [res.pvalues[term]],
            "term_name": [term]
        })

    return (
        res.params.to_frame(), 
        get_stats(did_candidates), 
        get_stats(triple_candidates),
        get_stats(time_candidates),      # Retornamos Time
        get_stats(treat_candidates)      # Retornamos Treat
    )

# ---------------------------------------------------------
# FUNCION DE EJECUCIÓN
# ---------------------------------------------------------

def run_lmm_screen(
        df, id_col="ID", group_col="Group",
        treatment_col="Treatment", time_col="Time",
        exclude_cols=None,
        min_nonmissing=5):
    
    df_prep, numeric_cols = f.prepare_dataframe(df, id_col, group_col, treatment_col, time_col, exclude_cols)

    rows = []
    failed = []

    for y in numeric_cols:
        # Filtros de calidad de datos
        dsub = df_prep[[id_col, group_col, treatment_col, time_col, y]].dropna()
        if len(dsub) < min_nonmissing or dsub[y].nunique() < 2:
            continue

        try:
            res = fit_lmm_for_outcome(dsub, y, id_col, group_col, treatment_col, time_col)

            # Desempaquetamos los 5 resultados
            full_table, did, triple, time_eff, treat_eff = extract_key_terms(
                res, treatment_col=treatment_col, time_col=time_col, group_col=group_col
            )

            # Helper para extraer valores seguros
            def get_val(df_term, col):
                return df_term[col].iloc[0] if len(df_term) else np.nan

            rows.append({
                "variable": y,
                "n": len(dsub),
                "n_subjects": dsub[id_col].nunique(),
                
                # --- A. Interacciones (Lo más importante) ---
                "Triple_beta": get_val(triple, "beta"),
                "Triple_p":    get_val(triple, "pvalue"),
                "DiD_beta":    get_val(did, "beta"),
                "DiD_p":       get_val(did, "pvalue"),
                
                # --- B. Efectos Principales (Nuevos) ---
                "Time_beta":   get_val(time_eff, "beta"), # Cambio por paso del tiempo (Placebo)
                "Time_p":      get_val(time_eff, "pvalue"),
                "Treat_beta":  get_val(treat_eff, "beta"), # Diferencia basal entre tratamientos
                "Treat_p":     get_val(treat_eff, "pvalue"),
                
                "converged": getattr(res, "converged", True)
            })

        except Exception as e:
            failed.append((y, str(e)))

    results = pd.DataFrame(rows)

    # --- AJUSTE FDR (Benjamini-Hochberg) ---
    # Ajustamos FDR para todas las columnas de p-value relevantes
    p_cols_to_adjust = ["Triple_p", "DiD_p", "Time_p", "Treat_p"]
    
    for col in p_cols_to_adjust:
        if col in results.columns and results[col].notna().any():
            p_vals = results[col].values
            mask = np.isfinite(p_vals)
            q_vals = np.full_like(p_vals, np.nan, dtype=float)
            
            if mask.sum() > 0:
                _, q, _, _ = multipletests(p_vals[mask], method="fdr_bh")
                q_vals[mask] = q
            
            # Nombre de la nueva columna (ej. Triple_q_FDR)
            results[col.replace("_p", "_q_FDR")] = q_vals

    # Ordenar por relevancia clínica (Triple > DiD > Time)
    sort_cols = [c for c in ["Triple_q_FDR", "Triple_p"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols, ascending=True)

    return results, failed