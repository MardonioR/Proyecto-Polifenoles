
import pandas as pd
import numpy as np

# ----------------------------
# FUNCIONES DE AYUDA
# ----------------------------

# Función para inferir tipo de variable
def inferir_tipo(s: pd.Series, max_cat_unique=10):
    """
    Infiere el tipo de variable. Basado en tipo de dato y máximo de valores diferentes para no ser categórica.

    Devuelve:
        - vacia
        - numerica
        - categorica 
    """
    s_nonmissing = s.dropna()
    if s_nonmissing.empty:
        return "vacia"
    
    if pd.api.types.is_numeric_dtype(s_nonmissing):
        nunique = s_nonmissing.nunique()
        if nunique > max_cat_unique:
            return "numerica"
        # numerica pero bajo numero de valores diferentes -> podría ser binaria u ordinal
        return "categorica"
    
    # no numerica -> categorica
    return "categorica"

# ---------------------------------------------------------
# PREPARACIÓN DEL DATAFRAME PARA MODELO LMM
# ---------------------------------------------------------

# Definiendo una función para preparación del DataFrame
def prepare_dataframe(
        df,
        id_col="id",
        group_col="grupo",
        treatment_col="Treatment",
        time_col="Time",
        exclude_cols=None):
    """
    - Asegura tipos categóricos y orden (Pre->Post).
    - Selecciona solo columnas numéricas (excepto columnas clave/excluidas).
    """
    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    key_cols = {id_col, group_col, treatment_col, time_col}
    exclude = set(exclude_cols) | key_cols

    # Tipos categóricos
    df[id_col] = df[id_col].astype("category")

    # Group como categórico
    df[group_col] = df[group_col].astype("category")

    # Orden explícito para Time: Pre < Post
    df[time_col] = pd.Categorical(df[time_col], categories=["Pre", "Post"], ordered=True)

    # Orden explícito para Treatment: Control como referencia
    df[treatment_col] = pd.Categorical(df[treatment_col],
                                       categories=["Control", "Intervencion"],
                                       ordered=True)

    # Selección de columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Quitar numéricas que sean clave o excluidas (p.ej. si Group está como int)
    numeric_cols = [
        c for c in numeric_cols 
        if c not in exclude 
        and inferir_tipo(s=df[c], max_cat_unique=10) == "numerica"
    ]

    return df, numeric_cols

    # ---------------------------------------------------------
    # INTERPRETADOR DE HALLAZGOS
    # ---------------------------------------------------------
def interpretar_hallazgos_final(df, 
                                p_triple="Triple_q_FDR", 
                                p_did="DiD_q_FDR", 
                                p_time="Time_q_FDR",
                                alpha=0.05):
    """
    Interpretación ajustada para diseño intra-sujeto (Sin vs Con Suplementación).
    """
    df_out = df.copy()
    
    # 1. CÁLCULO DE MAGNITUDES
    # Efecto Puro del Suplemento en G1 (Restando la deriva temporal)
    df_out['Efecto_Suplemento_G1'] = df_out['DiD_beta']
    
    # Efecto Puro del Suplemento en G2
    df_out['Efecto_Suplemento_G2'] = df_out['DiD_beta'] + df_out['Triple_beta']
    
    # Variación Natural (Lo que cambia el paciente cuando NO toma suplemento)
    df_out['Variacion_Natural'] = df_out['Time_beta']

    # 2. GENERACIÓN DE TEXTO
    def generar_texto(row):
        # Helper para p-values
        def check(col): return row.get(col, 1.0) < alpha

        eff_g1 = row['Efecto_Suplemento_G1']
        eff_g2 = row['Efecto_Suplemento_G2']
        natural = row['Variacion_Natural']
        
        # --- A. TRIPLE INTERACCIÓN (Diferencia de Respuesta entre Perfiles) ---
        if check(p_triple):
            if (eff_g1 * eff_g2) < 0:
                tipo = "INVERSA"
            elif abs(eff_g2) > abs(eff_g1):
                tipo = "INTENSIFICADA"
            else:
                tipo = "ATENUADA"
            
            return (f"RESPUESTA DIFERENCIADA ({tipo}): "
                    f"El Grupo 1 cambia {eff_g1:.2f} con el suplemento, mientras que "
                    f"el Grupo 2 cambia {eff_g2:.2f}. "
                    f"El perfil del paciente determina la eficacia.")

        # --- B. EFECTO SUPLEMENTO (DiD Significativo) ---
        elif check(p_did):
            dir_ = "MEJORA" if eff_g1 > 0 else "REDUCE" # Ajustar según si subir es bueno o malo
            return (f"SUPLEMENTO EFECTIVO: La suplementación genera un cambio de {eff_g1:.2f} "
                    f"más allá de la variación natural. Funciona igual para ambos grupos.")

        # --- C. EFECTO TEMPORAL / APRENDIZAJE (Time Significativo) ---
        elif check(p_time):
            return (f"VARIACIÓN TEMPORAL: La variable cambia ({natural:.2f}) por el paso del tiempo "
                    f"(o efecto aprendizaje) incluso SIN suplemento. La intervención no añadió beneficio extra.")

        # --- D. NEUTRO ---
        else:
            return "SIN EFECTO: Ni el tiempo ni el suplemento alteraron esta variable."

    df_out["Interpretacion"] = df_out.apply(generar_texto, axis=1)
    
    # Prioridad para ordenar tabla
    df_out["Ranking"] = df_out.apply(lambda x: 1 if x.get(p_triple,1)<alpha else (2 if x.get(p_did,1)<alpha else 3), axis=1)
    
    return df_out.sort_values("Ranking")