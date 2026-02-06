# ----------------------------
# FUNCIONES DE AYUDA
# ----------------------------

import pandas as pd
import numpy as np

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
        and f.inferir_tipo(s=df[c], max_cat_unique=10) == "numerica"
    ]

    return df, numeric_cols

    # ---------------------------------------------------------
    # INTERPRETADOR DE HALLAZGOS
    # ---------------------------------------------------------

def interpretar_hallazgos(df, p_col="Triple_pvalue", did_p_col="DiD_pvalue", alpha=0.05):
    """
    Agrega columnas de interpretación verbal calculando el efecto neto para cada grupo.
    """
    df_out = df.copy()
    
    # CÁLCULO PREVIO DE EFECTOS NETOS
    # Efecto estimado para el Grupo 1 (Referencia) es directo del Beta DiD
    df_out['Est_Efecto_G1'] = df_out['DiD_beta(Treat×Time)']
    
    # Efecto estimado para el Grupo 2 es la SUMA (Base + Diferencial)
    df_out['Est_Efecto_G2'] = df_out['DiD_beta(Treat×Time)'] + df_out['Triple_beta(Group×Treat×Time)']
    
    # LÓGICA DE INTERPRETACIÓN VERBAL
    def generar_texto(row):
        # Variables auxiliares para limpieza del código
        eff_g1 = row['Est_Efecto_G1']
        eff_g2 = row['Est_Efecto_G2']
        diff_beta = row['Triple_beta(Group×Treat×Time)']
        
        # --- A. Analizar Triple Interacción (Prioridad Máxima) ---
        if row[p_col] < alpha:
            # Determinar la naturaleza de la diferencia
            tipo_cambio = ""
            
            # Caso 1: Signos opuestos (Interacción cualitativa / Inversa)
            if (eff_g1 * eff_g2) < 0:
                tipo_cambio = "INVERSA (Los grupos van en direcciones opuestas)"
            
            # Caso 2: Mismo signo, pero G2 es más fuerte (mayor valor absoluto)
            elif abs(eff_g2) > abs(eff_g1):
                tipo_cambio = "INTENSIFICADA (Grupo 2 responde más fuerte)"
                
            # Caso 3: Mismo signo, pero G2 es más débil
            else:
                tipo_cambio = "ATENUADA (Grupo 2 responde menos)"

            dir_g1 = "AUMENTA" if eff_g1 > 0 else "DISMINUYE"
            dir_g2 = "AUMENTA" if eff_g2 > 0 else "DISMINUYE"

            return (f"INTERACCIÓN {tipo_cambio}: "
                    f"El Grupo 1 {dir_g1} ({eff_g1:.2f}), mientras que "
                    f"el Grupo 2 {dir_g2} ({eff_g2:.2f}). "
                    f"(Diferencial: {diff_beta:.2f})")
        
        # --- B. Analizar Efecto General (DiD) ---
        elif row[did_p_col] < alpha:
            direction = "AUMENTA" if eff_g1 > 0 else "DISMINUYE"
            return (f"EFECTO GENERAL (Ambos Grupos): El tratamiento {direction} "
                    f"la variable aprox {eff_g1:.2f}. "
                    f"No hay diferencia significativa entre grupos.")
            
        # --- C. Sin hallazgos ---
        else:
            return "Neutro: No se detectan cambios significativos atribuibles al tratamiento."

    # Aplicar la función
    df_out["Interpretacion_Clinica"] = df_out.apply(generar_texto, axis=1)
    
    # Ordenando por prioridad
    conditions = [
        (df_out[p_col] < alpha),      # Prioridad 1: Interacción compleja
        (df_out[did_p_col] < alpha)   # Prioridad 2: Efecto simple
    ]
    choices = [1, 2]
    df_out["Prioridad"] = np.select(conditions, choices, default=3)
    
    return df_out.sort_values("Prioridad")

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