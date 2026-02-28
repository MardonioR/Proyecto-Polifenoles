import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from src import functions as f

# Librerias de modelos
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import set_config
from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV, LeaveOneOut, KFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# ---------------------------------------------------------
# MODELOS DEL CUARTO AVANCE
# ---------------------------------------------------------
def models_comparison_and_train(X_train, y_train, X_test, y_test, params_dict, n_splits = 7, n_repeats = 5):
    # Registry de modelos
    MODEL_REGISTRY = {
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "SVR": SVR,
        "KNeighborsRegressor": KNeighborsRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "MLPRegressor": MLPRegressor,
    }

    cv_strategy = RepeatedKFold(
        n_splits= n_splits,
        n_repeats= n_repeats,
        random_state= 42
    )

    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    results = []
    best_estimators = {}

    for name, model_config in params_dict.items():

        start_time = time.time()
        logger.info(f"Optimizando modelo: {name}")

        # -------- Extraer configuración ----------
        model_class_name = model_config['model']['class']
        model_params = model_config['model']['params']
        param_grid = model_config['param_grid']

        model = MODEL_REGISTRY[model_class_name](**model_params)

        # -------- Pipeline  ----------
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', model)
            ],
            memory=None 
        )

        # -------- RandomizedSearch ----------
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(40, np.prod([len(v) for v in param_grid.values()])),
            cv=cv_strategy,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        best_estimators[name] = best_pipeline

        # -------- Evaluación en test ----------
        y_pred = best_pipeline.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        clean_params = {
            k.replace('model__', ''): v
            for k, v in search.best_params_.items()
        }

        elapsed = time.time() - start_time

        results.append({
            'Modelo': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Mejores Hiperparámetros': clean_params,
            'Tiempo (s)': round(elapsed, 2)
        })

        logger.info(f"{name} terminado en {elapsed:.2f} segundos")
    return results

def models_comparison_and_train_v2(X, y, params_dict):
    """
    Evalúa modelos usando Nested Leave-One-Out Cross-Validation para conjuntos de datos pequeños.
    """
    # Obliga a imprimir todos los parámetros, incluso los que están por defecto
    set_config(print_changed_only=False)

    # Configuración de logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    # Registry de modelos 
    MODEL_REGISTRY = {
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "SVR": SVR,
        "KNeighborsRegressor": KNeighborsRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "PLSRegression": PLSRegression 
    }

    # Estrategias de Validación Cruzada
    # Outer CV (Evaluación del rendimiento real): LOOCV deja 1 muestra fuera 42 veces
    outer_cv = LeaveOneOut()
    
    # Inner CV (Optimización de hiperparámetros): KFold simple para los 41 datos restantes
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_estimators = {}

    for name, model_config in params_dict.items():
        start_time = time.time()
        logger.info(f"Iniciando Nested CV para: {name}")

        # -------- Extraer configuración ----------
        model_class_name = model_config['model']['class']
        model_params = model_config['model']['params']
        param_grid = model_config['param_grid']

        model = MODEL_REGISTRY[model_class_name](**model_params)

        # -------- Pipeline ----------
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', model)
            ]
        )

        # -------- RandomizedSearch (Bucle Interno) ----------
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(40, np.prod([len(v) for v in param_grid.values()])),
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        # -------- Evaluación LOOCV (Bucle Externo) ----------
        # cross_val_predict ejecuta el 'search' 42 veces. 
        # Cada vez, optimiza hiperparámetros con 41 muestras y predice la muestra restante.
        y_pred_cv = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=-1)
        
        # Métricas de generalización (Test real)
        cv_rmse = root_mean_squared_error(y, y_pred_cv)
        cv_mae = mean_absolute_error(y, y_pred_cv)
        cv_r2 = r2_score(y, y_pred_cv)

        # -------- Ajuste Final y Brecha de Entrenamiento ----------
        # Ajustamos el search sobre todo el dataset (las 42 muestras) para obtener
        # el modelo final, sus hiperparámetros y su rendimiento en Train.
        search.fit(X, y)
        best_pipeline = search.best_estimator_
        best_estimators[name] = best_pipeline

        # Predicción sobre Train (los mismos datos completos) para ver la brecha
        y_pred_train = best_pipeline.predict(X)
        train_rmse = root_mean_squared_error(y, y_pred_train)
        train_r2 = r2_score(y, y_pred_train)

        clean_params = {
            k.replace('model__', ''): v
            for k, v in search.best_params_.items()
        }

        elapsed = time.time() - start_time

        # Guardar resultados con la brecha explícita
        results.append({
            'Modelo': name,
            'Train R2': train_r2,
            'CV (Test) R2': cv_r2,
            'Brecha R2': train_r2 - cv_r2,
            'Train RMSE': train_rmse,
            'CV RMSE': cv_rmse,
            'Mejores Hiperparámetros': clean_params,
            'Tiempo (s)': round(elapsed, 2)
        })

        logger.info(f"{name} finalizado. Train R2: {train_r2:.3f} | CV R2: {cv_r2:.3f} | Brecha R2: {train_r2 - cv_r2:.3f}")

    return pd.DataFrame(results), best_estimators

# Y - Randomization test
def run_y_randomization_test(best_estimators, X, y, n_permutations=100):
    """
    Ejecuta la prueba de Y-Randomization para comprobar la validez estadística de los modelos.
    
    Parámetros:
    - best_estimators: Diccionario con los hiperparametros ya optimizados.
    - X, y: Datasets completo.
    - n_permutations: Número de veces que se desordenará 'y'.
    """
    # Configuración de logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Usamos KFold de 8 para la prueba para no hacerla eternamente lenta
    cv_strategy = KFold(n_splits=8, shuffle=True, random_state=42)
    
    resultados_permutacion = {}

    # Configurar la gráfica para visualizar los 4 modelos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (nombre, pipeline) in enumerate(best_estimators.items()):
        logger.info(f"Iniciando Y-Randomization para: {nombre} ({n_permutations} permutaciones)...")
        
        # Ejecución del Permutation Test
        score_real, permuted_scores, pvalue = permutation_test_score(
            estimator=pipeline,
            X=X,
            y=y,
            scoring='r2',
            cv=cv_strategy,
            n_permutations=n_permutations,
            n_jobs=-1,
            random_state=42
        )
        
        resultados_permutacion[nombre] = {
            'R2 Real': score_real,
            'R2 Promedio Ruido': np.mean(permuted_scores),
            'p-value': pvalue
        }
        
        # --- Visualización del test ---
        ax = axes[idx]
        ax.hist(permuted_scores, bins=20, density=True, alpha=0.7, color='gray', label='Scores Aleatorios (Ruido)')
        ax.axvline(score_real, color='red', linestyle='dashed', linewidth=2, label=f'R2 Real ({score_real:.2f})')
        ax.axvline(np.mean(permuted_scores), color='black', linestyle='solid', linewidth=1, label='Media Ruido')
        
        ax.set_title(f"{nombre} | p-value: {pvalue:.4f}")
        ax.set_xlabel("R2 Score")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        
        logger.info(f"{nombre} -> R2 Real: {score_real:.3f} | R2 Ruido: {np.mean(permuted_scores):.3f} | p-value: {pvalue:.4f}")

    plt.tight_layout()
    plt.show()
    
    return resultados_permutacion

def build_and_evaluate_ensemble(best_estimators, X, y):
    """
    Construye un VotingRegressor a partir de los mejores modelos optimizados
    y lo evalúa utilizando Leave-One-Out Cross-Validation.
    """
    logger = logging.getLogger(__name__)
    logger.info("Construyendo el ensamble unificado...")
    
    # Extraer solo los modelos optimizados (ignorando los scalers individuales)
    modelos_aprobados = ['Ridge', 'Lasso', 'ElasticNet', 'SVR']
    estimadores_ensamble = []
    
    for nombre, pipeline in best_estimators.items():
        if nombre in modelos_aprobados:
            # Extraemos el modelo con los hiperparámetros que el RandomizedSearchCV ya encontró
            modelo_optimizado = pipeline.named_steps['model']
            estimadores_ensamble.append((nombre, modelo_optimizado))
            
    # Crear el VotingRegressor con los modelos base
    ensamble_voting = VotingRegressor(estimators=estimadores_ensamble)
    
    # Crear el Pipeline maestro (Escalado único -> Ensamble)
    pipeline_final = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('ensamble', ensamble_voting)
        ]
    )
    
    # Evaluación rigurosa (LOOCV) para N=42
    logger.info("Evaluando el ensamble con LOOCV...")
    loo = LeaveOneOut()
    
    # cross_val_predict ajustará el pipeline 42 veces dejando un paciente fuera a la vez
    y_pred_cv = cross_val_predict(pipeline_final, X, y, cv=loo, n_jobs=-1)
    
    # Métricas de generalización
    cv_r2 = r2_score(y, y_pred_cv)
    cv_rmse = root_mean_squared_error(y, y_pred_cv)
    
    # Ajuste final con todos los datos para medir la brecha
    pipeline_final.fit(X, y)
    y_pred_train = pipeline_final.predict(X)
    train_r2 = r2_score(y, y_pred_train)
    
    # Imprimir reporte
    print("\n" + "="*50)
    print("RESULTADOS DEL MODELO FINAL ENSAMBLADO")
    print("="*50)
    print(f"R2 Entrenamiento (Train):      {train_r2:.4f}")
    print(f"R2 Validación (LOOCV Test):    {cv_r2:.4f}")
    print(f"Brecha de R2 (Train - Test):   {train_r2 - cv_r2:.4f}")
    print(f"RMSE de Validación:            {cv_rmse:.4f}")
    print("="*50)
    
    return pipeline_final
