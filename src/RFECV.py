import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
from typing import List, Tuple
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from yellowbrick.model_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def select_features_rfecv_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    step: int = 5,
    cv_splits: int = 5
) -> Tuple[List[str], pd.DataFrame]:
    
    print(f" Analizando {X.shape[1]} variables.")

    model = XGBRegressor(n_jobs=-1, random_state=42)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    visualizer = RFECV(estimator=model, step=step, cv=cv, scoring="r2", n_jobs=-1)
    
    start_time = time.time()
    visualizer.fit(X, y)
    
    # --- Extracción de datos para la gráfica ---
    mean_scores = visualizer.cv_scores_.mean(axis=1)
    # Reconstruimos el conteo de variables según el 'step'
    n_features_steps = np.linspace(X.shape[1] % step or step, X.shape[1], len(mean_scores))
    
    # --- Gráfica con Eje Y = Número de Variables ---
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=mean_scores, y=n_features_steps, marker='o', color='#2c3e50', lw=2)
    
    # Encontrar el punto óptimo
    best_idx = np.argmax(mean_scores)
    best_score = mean_scores[best_idx]
    best_n = n_features_steps[best_idx]
    
    # Resaltar el punto óptimo
    plt.axhline(y=best_n, color='#e74c3c', linestyle='--', alpha=0.6)
    plt.scatter(best_score, best_n, color='#e74c3c', s=100, zorder=5, label=f'Óptimo: {int(best_n)} vars')

    plt.title("Curva de Eficiencia: Precisión vs. Complejidad", fontsize=14, pad=20)
    plt.xlabel("Métrica de Rendimiento (R²)", fontsize=12)
    plt.ylabel("Cantidad de Variables Seleccionadas", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Resultados y Ranking ---
    ranking = visualizer.ranking_
    
    # Obtener las importancias del estimador final
    importances = visualizer.estimator_.feature_importances_
    
    full_importances = np.zeros(X.shape[1])
    # Llenamos con la importancia solo donde el ranking es 1
    full_importances[ranking == 1] = importances

    # 3. Crear el DataFrame detallado
    df_ranking = pd.DataFrame({
        "Feature": X.columns,
        "Ranking": ranking,
        "Importance": full_importances
    })

    # Ordenamos: Primero por Ranking (ascendente) y luego por Importancia (descendente)
    df_ranking = df_ranking.sort_values(
        by=["Ranking", "Importance"], 
        ascending=[True, False]
    ).reset_index(drop=True)

    # --- Reporte de las Top Variables ---
    selected_features = df_ranking[df_ranking["Ranking"] == 1]["Feature"].tolist()

    print(f"Proceso terminado en {time.time() - start_time:.2f}s")
    print(f"Resultado: Se eliminaron {X.shape[1] - len(selected_features)} variables.")
    
    return selected_features, df_ranking

def select_features_rfecv_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    step: int = 1,          
    cv_splits: int = 5,
    alpha: float = 1.0      # Parámetro de penalización
) -> Tuple[List[str], pd.DataFrame]:
    
    print(f"Analizando {X.shape[1]} variables con Lasso (Penalización L1)...")

    # Excalando para que la penalización sea justa para todas las variables.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Definimos el modelo Lasso
    # Aumentamos max_iter para asegurar convergencia
    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # En Lasso, las "importancias" son los coeficientes absolutos
    visualizer = RFECV(estimator=model, step=step, cv=cv, scoring="r2", n_jobs=-1)
    
    start_time = time.time()
    visualizer.fit(X_scaled, y)
    
    # --- Extracción de datos para la gráfica ---
    mean_scores = visualizer.cv_scores_.mean(axis=1)
    x_axis = np.linspace(X.shape[1] % step or step, X.shape[1], len(mean_scores))
    
    # --- Gráfica ---
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=mean_scores, y=x_axis, marker='o', color='#16a085', lw=2)
    
    best_idx = np.argmax(mean_scores)
    plt.axhline(y=x_axis[best_idx], color='#e74c3c', linestyle='--')
    plt.scatter(mean_scores[best_idx], x_axis[best_idx], color='#e74c3c', s=100, zorder=5, 
                label=f'Óptimo: {int(x_axis[best_idx])} vars')

    plt.title("Selección Lasso: Precisión vs. Complejidad", fontsize=14)
    plt.xlabel("Métrica de Rendimiento (R²)", fontsize=12)
    plt.ylabel("Cantidad de Variables Seleccionadas", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

    # --- Resultados y Ranking ---
    ranking = visualizer.ranking_
    
    # En modelos lineales, usamos coef_ en lugar de feature_importances_
    # Tomamos el valor absoluto porque un coeficiente negativo grande también es importante
    coefs = np.abs(visualizer.estimator_.coef_)
    
    full_importances = np.zeros(X.shape[1])
    full_importances[ranking == 1] = coefs

    df_ranking = pd.DataFrame({
        "Feature": X.columns,
        "Ranking": ranking,
        "Importance_Coef": full_importances
    }).sort_values(by=["Ranking", "Importance_Coef"], ascending=[True, False]).reset_index(drop=True)

    selected_features = df_ranking[df_ranking["Ranking"] == 1]["Feature"].tolist()

    print(f"Proceso terminado en {time.time() - start_time:.2f}s")
    print(f"Resultado: Lasso seleccionó {len(selected_features)} variables.")
    
    return selected_features, df_ranking
