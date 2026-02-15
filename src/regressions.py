"""
Docstring for regressions
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# MODELO BASE
# ---------------------------------------------------------
def train_and_evaluate_xgboost(X, y, selected_features):
    """
    Entrena el modelo final usando solo las variables seleccionadas.
    """
    
    # Filtrar solo las variables ganadoras (Ranking 1)
    X_selected = X[selected_features]
    
    # Separación en Entrenamiento y Prueba (80/20)
    # Usamos random_state para que tus resultados sean reproducibles
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42
    )
    
    print(f"✅ Datos preparados: Train {X_train.shape[0]} | Test {X_test.shape[0]}")

    # Configuración del Modelo
    # Ajustamos hiperparámetros básicos para evitar overfitting
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,              # Usamos solo el 70% de las filas por árbol
        colsample_bytree=0.7,       # Usamos solo el 70% de las variables por árbol
        
        # --- Regularización ---
        reg_alpha=10,               # L1 (Lasso): ayuda a ignorar variables que aún sean ruido
        reg_lambda=1,               # L2 (Ridge): reduce la magnitud de los pesos
        
        n_jobs=-1,
        random_state=42,
        verbosity=0                 # Menos distracciones en el log
    )

    # Entrenamiento
    print("Entrenando modelo final...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predicciones
    y_pred = model.predict(X_test)

    # Cálculo de Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("\n" + "="*30)
    print("METRICAS DE DESEMPEÑO")
    print("="*30)
    print(f"R² Score:   {r2:.4f}  (Proporción de varianza explicada)")
    print(f"MAE:        {mae:.4f}  (Error promedio en unidades reales)")
    print(f"RMSE:       {rmse:.4f}  (Penaliza errores grandes)")
    print(f"MAPE:       {mape:.2f}% (Error relativo porcentual)")
    print("="*30)

    return model, X_test, y_test, y_pred

from sklearn.ensemble import RandomForestRegressor

def train_and_evaluate_random_forest(X, y, selected_features):
    """
    Entrena el modelo final usando Random Forest con las variables seleccionadas.
    """
    
    # Filtrar solo las variables ganadoras (Ranking 1)
    X_selected = X[selected_features]
    
    # Separación en Entrenamiento y Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42
    )
    
    print(f"✅ Datos preparados: Train {X_train.shape[0]} | Test {X_test.shape[0]}")

    # Configuración del Modelo (Random Forest)
    # Usamos parámetros que favorecen la generalización en datasets pequeños
    model = RandomForestRegressor(
        n_estimators=1000,          # Suficientes árboles para estabilizar el promedio
        max_depth=5,                # Controlamos la profundidad para evitar sobreajuste
        min_samples_split=5,        # Mínimo de muestras para dividir un nodo
        min_samples_leaf=3,         # Mínimo de muestras en cada "hoja"
        max_features='sqrt',        # Cada árbol solo ve una fracción de las variables (reduce correlación)
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    # Entrenamiento
    print("Entrenando Random Forest final...")
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Cálculo de Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Manejo de MAPE para evitar divisiones por cero si hay valores nulos
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

    print("\n" + "="*30)
    print("METRICAS DE DESEMPEÑO (Random Forest)")
    print("="*30)
    print(f"R² Score:   {r2:.4f}  (Proporción de varianza explicada)")
    print(f"MAE:        {mae:.4f}  (Error promedio en unidades reales)")
    print(f"RMSE:       {rmse:.4f}  (Penaliza errores grandes)")
    print(f"MAPE:       {mape:.2f}% (Error relativo porcentual)")
    print("="*30)

    return model, X_test, y_test, y_pred
