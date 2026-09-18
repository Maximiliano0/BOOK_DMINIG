# Books Sales and Ratings

Análisis reproducible de ventas, calificaciones y generación de ingresos editoriales mediante Machine Learning.

Proyecto integrador de Machine Learning aplicado a un catálogo editorial. El análisis combina exploración y limpieza de datos, ingeniería de características, clasificación, regresión y simulación de elasticidad de precio.

> El notebook está escrito en español y puede ejecutarse en Google Colab o en un entorno local de Jupyter.

## Objetivos

El proyecto responde tres preguntas de negocio:

1. **¿Qué libros serán bestsellers?** Clasificación binaria desbalanceada con XGBoost.
2. **¿Cuántas unidades se venderán y cómo responde la demanda al precio?** Regresión de `log1p(units_sold)` comparando XGBoost y LightGBM, seguida de una simulación de precios.
3. **¿Qué libros generan ingresos para la editorial?** Clasificación con SVM y Random Forest.

El horizonte de decisión es el de un libro **ya publicado**. Por eso se consideran disponibles sus calificaciones y cantidad de reseñas. No es un modelo de predicción para lanzamientos.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| [`Books_Sales_and_Ratings.ipynb`](Books_Sales_and_Ratings.ipynb) | Notebook completo con el análisis, los modelos, las visualizaciones y las simulaciones. |
| [`Books_Data_Clean.csv`](Books_Data_Clean.csv) | Dataset CSV utilizado como referencia local. |
| `README.md` | Documentación del proyecto. |

El dataset original proviene de [Books Sales and Ratings en Kaggle](https://www.kaggle.com/datasets/thedevastator/books-sales-and-ratings), publicado por `thedevastator`.

## Hallazgos principales

- El archivo contiene dos poblaciones con escalas de venta incompatibles. El análisis conserva sólo el bloque que respeta la identidad `gross_sales = units_sold * sale_price`.
- La variable objetivo de bestsellers se define dentro del bloque consistente, usando el percentil 80 de `units_sold`; así se evita que la etiqueta represente el origen del registro en lugar del desempeño del libro.
- La prueba social (`book_average_rating`, `book_ratings_count` y `social_proof_score`) se considera válida porque el caso de uso corresponde a títulos ya publicados. Para un lanzamiento nuevo, esas variables no estarían disponibles.
- El margen editorial no es continuo en este dataset: `margin_ratio` toma esencialmente dos estados. Por eso el tercer experimento predice `publisher_revenue > 0` en lugar de separar por la mediana del margen.
- La simulación de precios es observacional. Permite priorizar experimentos comerciales, pero no identifica por sí sola un efecto causal del precio.

Las tablas con métricas, modelos ganadores, gráficos y valores de las simulaciones se generan dentro del notebook al ejecutar todas las celdas. No se incluyen cifras estáticas en este README porque pueden cambiar según las versiones instaladas y la copia del dataset descargada desde Kaggle.

## Flujo del análisis

El notebook está organizado en las siguientes etapas:

- **Configuración:** dependencias, semillas aleatorias y estilo visual.
- **Adquisición:** descarga del dataset mediante la API de Kaggle.
- **EDA y limpieza:** nulos, duplicados, outliers, variantes ortográficas, nombres inconsistentes y auditoría de integridad.
- **Ingeniería de características:** antigüedad del libro, precio relativo al género, prueba social y proporción de ingresos editoriales.
- **Preparación:** división train/test 70/30, imputación ajustada sólo con train, One-Hot Encoding o Target Encoding out-of-fold, y escalado cuando corresponde.
- **Experimento 1:** clasificación de bestsellers con corte en el percentil 80 de `units_sold`.
- **Experimento 2:** regresión de demanda, comparación de modelos y simulación de ingreso bajo distintos precios.
- **Experimento 3:** clasificación de títulos que generan ingresos editoriales.
- **Anexos:** pruebas de humo con datos sintéticos para verificar la mecánica de los modelos.

## Limpieza y control de fuga de información

El notebook no modela directamente todas las columnas del CSV. Entre las principales decisiones:

- Se eliminan duplicados exactos y años de publicación imposibles.
- Los montos negativos se convierten en valores faltantes para imputarlos después del split.
- Se normalizan autores, títulos, editoriales y categorías con reglas explícitas.
- Se elimina `language_code` por sus códigos inconsistentes.
- La auditoría detecta dos poblaciones con escalas incompatibles. Se conserva el bloque donde se cumple aproximadamente:

  `gross_sales = units_sold * sale_price`

- Se excluyen `gross_sales`, `publisher_revenue`, `sales_rank`, `margin_ratio` y otras variables derivadas del resultado para evitar fuga de información (*data leakage*).
- Las estadísticas de imputación, las medias de precio por género, el Target Encoding y el escalado se ajustan después del `train_test_split` y sólo sobre train.

Las variables principales utilizadas por los modelos son `author`, `author_rating`, `genre`, `publisher`, `sale_price`, `book_age`, `price_relative_to_genre`, `book_average_rating`, `book_ratings_count` y `social_proof_score`.

## Instalación

Se recomienda Python 3.10 o superior y un entorno virtual.

```bash
python -m venv .venv
```

Activación en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalación de dependencias:

```bash
pip install jupyter pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm shap optuna kaggle
```

El notebook también contiene una celda de instalación que actualiza `xgboost`, `lightgbm`, `shap`, `optuna` y `kaggle` al ejecutarse en Colab. Para una ejecución reproducible en un entorno compartido, conviene fijar las versiones después de la primera ejecución y generar un archivo `requirements.txt`.

## Credenciales de Kaggle

La primera ejecución descarga el dataset desde Kaggle. Para habilitarla:

1. En Kaggle, abrir **Settings > API > Create New Token**.
2. Descargar `kaggle.json`.
3. Guardarlo en:

   `~/.kaggle/kaggle.json`

En Google Colab, el notebook ofrece un flujo para subir el archivo durante la ejecución. No se debe subir `kaggle.json` al repositorio ni escribir sus credenciales dentro del notebook.

## Ejecución

Desde la carpeta del proyecto:

```bash
jupyter notebook Books_Sales_and_Ratings.ipynb
```

También puede abrirse con JupyterLab o cargarse en Google Colab. Ejecutar las celdas en orden, desde el inicio, porque cada experimento reutiliza objetos construidos en las secciones anteriores.

La descarga crea una carpeta `data/` y selecciona el CSV descargado de mayor tamaño. El archivo local incluido en el repositorio sirve como referencia y respaldo del dataset, pero el flujo principal del notebook obtiene los datos mediante la API de Kaggle.

### Ejecución sin Colab

El bloque de credenciales detecta si se ejecuta en Colab. En un entorno local, `kaggle.json` debe existir antes de iniciar el notebook en `~/.kaggle/kaggle.json`; de lo contrario, la celda de adquisición detiene la ejecución con un mensaje explicativo.

## Modelos, métricas y resultados

- **Clasificación:** Accuracy, Precision, Recall, F1-Score, ROC-AUC y PR-AUC. En el experimento de bestsellers se prioriza PR-AUC porque la clase positiva representa aproximadamente el 20 % de los casos.
- **Regresión:** RMSE, MAE y R² en escala logarítmica y en unidades vendidas.
- **Optimización:** Optuna para XGBoost y LightGBM; GridSearchCV para SVM y Random Forest.
- **Interpretabilidad:** SHAP para XGBoost, importancias de Random Forest, coeficientes de SVM lineal y permutation importance.

La selección de modelos se realiza con criterios distintos según el problema:

| Experimento | Criterio principal | Modelos comparados |
|---|---|---|
| Bestsellers | PR-AUC en validación y evaluación en test | XGBoost |
| Demanda | RMSE en escala logarítmica | Baseline, XGBoost y LightGBM |
| Ingreso editorial | F1-Score | SVM RBF, SVM lineal y Random Forest |

El notebook imprime las métricas de test y marca el modelo ganador en cada experimento. También muestra la línea base correspondiente, matrices de confusión, curvas ROC/Precision-Recall y diagnósticos de residuos cuando aplican.

## Interpretación de la elasticidad

La simulación mantiene constantes los demás atributos del libro y modifica `sale_price` dentro de los percentiles 2 y 98 observados en train. Esto evita extrapolar con modelos de árboles fuera del rango de datos.

La elasticidad se calcula como:

`elasticidad = d ln(Q) / d ln(P)`

Esta sección es una simulación observacional, no una estimación causal. El precio no fue asignado aleatoriamente en el dataset, por lo que las curvas sirven para priorizar pruebas de precio, no para afirmar que un cambio producirá necesariamente ese efecto en el mercado. Si el ingreso máximo aparece en el borde superior de la ventana, debe interpretarse como una señal para probar precios más altos, no como un precio óptimo demostrado.

## Limitaciones

- El dataset es pequeño para la complejidad de las preguntas planteadas.
- La auditoría revela poblaciones con escalas de venta incompatibles y obliga a trabajar sólo con el bloque internamente consistente.
- Las reseñas y calificaciones mejoran la predicción de libros ya publicados, pero no están disponibles para un lanzamiento nuevo.
- `publisher_revenue` presenta una estructura prácticamente binaria: el experimento usa `publisher_revenue > 0` en lugar de un corte por la mediana de `margin_ratio`.
- Las predicciones reflejan asociaciones observadas y no prueban causalidad.
- Las métricas finales dependen de volver a ejecutar el notebook con las versiones instaladas y la semilla configurada.

## Reproducibilidad

El proyecto fija `RANDOM_STATE = 42`, usa una partición de test del 30 % y aplica el mismo criterio de features en los tres experimentos. Aun así, pueden existir pequeñas diferencias por versiones de Python, scikit-learn, XGBoost, LightGBM, Optuna o por cambios en el dataset descargado desde Kaggle.

## Licencia y fuente de datos

Este repositorio documenta un trabajo académico. Consultar la licencia y las condiciones de uso del dataset en su [página original de Kaggle](https://www.kaggle.com/datasets/thedevastator/books-sales-and-ratings) antes de redistribuir los datos.
