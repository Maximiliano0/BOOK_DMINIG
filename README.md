# Books Sales and Ratings

**Autor:** Vega Maximiliano

Proyecto Integrador de Machine Learning (Talento Tech) sobre un catálogo editorial: limpieza y auditoría de datos, pipelines de scikit-learn, clasificación, regresión y simulación de elasticidad de precio. El notebook está en español y se ejecuta en Google Colab o en Jupyter local.

## Contenido

| Archivo | Descripción |
|---|---|
| [`Books_Sales_and_Ratings.ipynb`](Books_Sales_and_Ratings.ipynb) | Entrega final, ejecutada de principio a fin y organizada según la consigna (Pre-Entrega y Entrega Final). |
| [`Books_Data_Clean.csv`](Books_Data_Clean.csv) | Copia local del dataset. |

Dataset: [Books Sales and Ratings](https://www.kaggle.com/datasets/thedevastator/books-sales-and-ratings) (Kaggle, thedevastator).

## Objetivos

1. Clasificar si un libro pertenece al quintil superior de ventas (XGBoost).
2. Estimar las unidades vendidas en escala logarítmica (Ridge, XGBoost y LightGBM) y simular la respuesta al precio.
3. Clasificar si la editorial obtiene ingreso por el título (SVM lineal y RBF, Random Forest), con una regla por editorial como referencia.

Los modelos están pensados para títulos ya publicados, cuyas calificaciones de lectores existen. El notebook mide cuánto cae el desempeño sin esas variables.

## Metodología

- Limpieza antes del pipeline: exclusión de obras anteriores a 1455 (el año es de composición, no de edición), reparación de texto con doble codificación, unificación de variantes de autores y editoriales, eliminación de `language_code` y normalización de categorías.
- Auditoría de integridad: el archivo combina dos poblaciones con escalas de venta incompatibles. Se conserva el bloque en el que se cumple `gross_sales = units_sold × sale_price`.
- Las transformaciones que estiman parámetros (precio medio por género, imputación, One-Hot, codificación por objetivo de `author` y escalado) forman parte de un `Pipeline` y se ajustan solo con el conjunto de entrenamiento.
- Validación cruzada de 5 particiones con `GridSearchCV` sobre el pipeline completo. Hiperparámetros y modelos se eligen con la regla de un error estándar.
- Modelo de referencia en cada experimento y una única evaluación sobre el conjunto de prueba.
- Interpretación con SHAP, importancia por permutación, análisis de errores y predicción sobre títulos nuevos.

## Resultados principales

- En el bloque consistente, `book_ratings_count` es el principal predictor de las ventas y `sales_rank` es casi una función de `units_sold`, por lo que se excluye.
- El clasificador de superventas funciona para títulos ya publicados y queda cerca del azar sin las calificaciones.
- En la regresión, Ridge generaliza mejor que los modelos de boosting.
- El modelo de demanda es casi insensible al precio, así que los datos no permiten recomendar precios.
- El ingreso editorial nulo aparece solo en títulos de Amazon y ningún modelo supera a la regla por editorial.

Las métricas y los gráficos están en el notebook ejecutado.

## Ejecución

En Google Colab basta con subir el notebook y ejecutar todas las celdas: el dataset se descarga con `kagglehub` (sin credenciales) y las librerías vienen instaladas. En un entorno local, con `Books_Data_Clean.csv` en la misma carpeta:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install jupyter pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm shap kagglehub
jupyter notebook Books_Sales_and_Ratings.ipynb
```

Las celdas deben ejecutarse en orden. Se usa `RANDOM_STATE = 42`; con otras versiones de las librerías (impresas en la primera celda) los resultados pueden variar levemente.

## Licencia

Trabajo académico. Antes de redistribuir los datos, consultar las condiciones de uso en la [página del dataset](https://www.kaggle.com/datasets/thedevastator/books-sales-and-ratings).
