
# Predicción de Respuesta Cognitiva a Intervención con Polifenoles (Estudio MITOS)

Este repositorio contiene el código fuente y la documentación para el Trabajo de Fin de Máster en Inteligencia Artificial Aplicada. El proyecto se centra en el desarrollo de un modelo predictivo supervisado para identificar perfiles de respuesta en el ensayo clínico MITOS.

## 📋 Descripción del Proyecto

El objetivo principal de este trabajo es desarrollar y validar un modelo de Machine Learning supervisado que integre múltiples dimensiones de datos de los participantes del ensayo clínico **MITOS** ([NCT05891977](https://clinicaltrials.gov/ct2/show/NCT05891977)).

El modelo busca identificar las características asociadas a una mayor mejora cognitiva inducida por una intervención dietética rica en polifenoles. A través de este análisis, se pretende:
1. Integrar datos clínicos, analíticos, cognitivos y de estilo de vida (dieta, actividad física, sueño y composición corporal).
2. Definir el perfil multidimensional de los "mejores respondedores" (*best responders*).
3. Avanzar hacia recomendaciones nutricionales personalizadas basadas en evidencia.

## 🛠️ Tecnologías Utilizadas

El proyecto utiliza librerías clásicas de Python para el análisis de datos y aprendizaje automático:

* **Python 3.12**
* **Pandas & NumPy:** Manipulación y análisis de datos.
* **Scikit-learn:** Modelado predictivo y preprocesamiento.
* **Matplotlib & Seaborn:** Visualización de datos exploratoria y de resultados.
* **SciPy:** Análisis estadístico complementario.

## 🚀 Instalación y Configuración

Sigue estos pasos para configurar el entorno de desarrollo local y ejecutar el proyecto.

### 1. Clonar el repositorio

Descarga el código fuente a tu máquina local:

```bash
git clone [https://github.com/tu-usuario/nombre-del-repo.git](https://github.com/tu-usuario/nombre-del-repo.git)
cd nombre-del-repo

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```
