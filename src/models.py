import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import functions as f

# ---------------------------------------------------------
# AJUSTE DE MODELO LMM
# ---------------------------------------------------------

# Definiendo función para ajustar LMM a variable
def fit_lmm_for_outcome(
        df, outcome, id_col="id", group_col="grupo",
        treatment_col="Treatment", time_col="Time",
        reml=False, method="lbfgs", maxiter=2000):
    """
    Ajusta un MixedLM:
        outcome ~ Group * Treatment * Time
    con intercepto aleatorio por ID.

    Devuelve:
    - resultado statsmodels
    """
    formula = f"{outcome} ~ C({group_col}) * C({treatment_col}) * C({time_col})"

    # Random intercept por sujeto
    model = smf.mixedlm(formula=formula, data=df, groups=df[id_col], re_formula="1")

    # Nota: MixedLM a veces requiere ajustar optimizador/iteraciones
    res = model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
    return res

# ---------------------------------------------------------
# BUSQUEDA DE PARAMETROS
# ---------------------------------------------------------

# Definiendo función para extraer parametros del modelo
def extract_key_terms(res, treatment_col="Treatment", time_col="Time", group_col="Group"):
    """
    Busca en los parámetros del modelo los términos de interacción correctos, sin importar si statsmodels les pone 'C(...)[T...]' alrededor.
    """
    
    # Obtenemos todos los nombres de los coeficientes generados
    param_names = res.params.index.tolist()
    
    # Buscar el término DiD (Interacción Tratamiento x Tiempo) ---
    # Debe contener el nombre de la col tratamiento, la col tiempo y dos puntos ":"
    did_candidates = [
        name for name in param_names 
        if treatment_col in name and time_col in name and ":" in name and group_col not in name
    ]
    
    # Buscar Triple Interacción (Grupo x Tratamiento x Tiempo) ---
    # Contiene los tres nombres y dos puntos
    triple_candidates = [
        name for name in param_names 
        if treatment_col in name and time_col in name and group_col in name and name.count(":") >= 2
    ]
    
    # Función auxiliar para extraer beta y pvalue de una lista de candidatos
    def get_stats(candidates):
        if not candidates:
            return pd.DataFrame() # Vacío si no encuentra
        
        # Tomamos el primer candidato encontrado (usualmente solo hay uno relevante)
        term = candidates[0] 
        return pd.DataFrame({
            "beta": [res.params[term]],
            "pvalue": [res.pvalues[term]],
            "term_name": [term] # Útil para debug
        })

    did_df = get_stats(did_candidates)
    triple_df = get_stats(triple_candidates)
    
    # Regresando tabla completa, y los dataframes específicos
    return res.params.to_frame(), did_df, triple_df


# ---------------------------------------------------------
# FUNCION DE EJECUCIÓN
# ---------------------------------------------------------

# Corriendo el podelo para todas las variables
def run_lmm_screen(
        df,id_col="ID", group_col="Group",
        treatment_col="Treatment", time_col="Time",
        exclude_cols=None,
        min_nonmissing=5):
    """
    Ejecuta MixedLM para cada outcome numérico.
    - Filtra outcomes con demasiados missing
    - Extrae p-values de términos clave
    - Ajusta FDR (Benjamini-Hochberg) a través de outcomes
    """
    df_prep, numeric_cols = f.prepare_dataframe(df, id_col, group_col, treatment_col, time_col, exclude_cols)

    rows = []
    failed = []

    for y in numeric_cols:
        # Filtrar si hay pocos datos no-missing
        n_nonmiss = df_prep[y].notna().sum()
        if n_nonmiss < min_nonmissing:
            continue

        # MixedLM requiere eliminar filas con missing en variables usadas
        dsub = df_prep[[id_col, group_col, treatment_col, time_col, y]].dropna()

        # Necesitas variación: si es constante no se puede ajustar bien
        if dsub[y].nunique() < 2:
            continue

        try:
            res = fit_lmm_for_outcome(dsub, y, id_col, group_col, treatment_col, time_col)

            full_table, did, triple = extract_key_terms(res, treatment_col=treatment_col, time_col=time_col, group_col=group_col)

            # Guardamos términos clave
            # Si por alguna razón no se encuentra el nombre exacto, lo marcamos NaN.
            did_beta = did["beta"].iloc[0] if len(did) else np.nan
            did_p = did["pvalue"].iloc[0] if len(did) else np.nan

            triple_beta = triple["beta"].iloc[0] if len(triple) else np.nan
            triple_p = triple["pvalue"].iloc[0] if len(triple) else np.nan

            rows.append({
                "variable": y,
                "n": len(dsub),
                "n_subjects": dsub[id_col].nunique(),
                "DiD_beta(Treat×Time)": did_beta,
                "DiD_pvalue": did_p,
                "Triple_beta(Group×Treat×Time)": triple_beta,
                "Triple_pvalue": triple_p,
                "converged": getattr(res, "converged", True)
            })

        except Exception as e:
            failed.append((y, str(e)))

    results = pd.DataFrame(rows)

    # Ajuste FDR por múltiples outcomes:
    # (1) para el DiD
    if results["DiD_pvalue"].notna().any():
        p = results["DiD_pvalue"].values
        mask = np.isfinite(p)
        q = np.full_like(p, np.nan, dtype=float)
        rej, qvals, _, _ = multipletests(p[mask], method="fdr_bh")
        q[mask] = qvals
        results["DiD_qvalue_FDR"] = q

    # (2) para la triple interacción
    if results["Triple_pvalue"].notna().any():
        p = results["Triple_pvalue"].values
        mask = np.isfinite(p)
        q = np.full_like(p, np.nan, dtype=float)
        rej, qvals, _, _ = multipletests(p[mask], method="fdr_bh")
        q[mask] = qvals
        results["Triple_qvalue_FDR"] = q

    # Orden sugerido: primero por evidencia en triple interacción (diferencia de DiD entre grupos)
    if "Triple_qvalue_FDR" in results.columns:
        results = results.sort_values(["Triple_qvalue_FDR", "Triple_pvalue"], ascending=True)

    return results, failed