import os
import json
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# Paso 1: Cargar datasets escalados preprocesados
# -------------------------------------------------------------
# Resolver rutas absolutas basándose en la ubicación de este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

train_path = os.path.join(OUTPUT_DIR, 'train_clean_scaled.csv')
test_path = os.path.join(OUTPUT_DIR, 'test_clean_scaled.csv')

# Búsqueda de rutas alternativas (fallback) si no se encuentran en la carpeta de salida relativa al script
if not (os.path.exists(train_path) and os.path.exists(test_path)):
    possible_dirs = [
        'output',
        '../output',
        'fede/output',
        '/home/fzacchigna/ceia/2-amq/TP_ADM/output',
        '/home/fzacchigna/ceia/2-amq/TP_ADM/fede/output'
    ]
    for d in possible_dirs:
        tr = os.path.join(d, 'train_clean_scaled.csv')
        te = os.path.join(d, 'test_clean_scaled.csv')
        if os.path.exists(tr) and os.path.exists(te):
            train_path = tr
            test_path = te
            break

if train_path is None or not os.path.exists(train_path):
    raise FileNotFoundError("No se pudieron encontrar 'train_clean_scaled.csv' y 'test_clean_scaled.csv' en las rutas por defecto o alternativas.")

print(f"Cargando datasets escalados:\n - Train: {train_path}\n - Test: {test_path}")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separar target (BodyFat) y características (X)
X_train = train_df.drop(columns=['BodyFat'])
y_train = train_df['BodyFat']
X_test = test_df.drop(columns=['BodyFat'])
y_test = test_df['BodyFat']

print(f"Dataset de entrenamiento: Características {X_train.shape}, Target {y_train.shape}")
print(f"Dataset de prueba:        Características {X_test.shape}, Target {y_test.shape}")

# -------------------------------------------------------------
# Paso 2: Definir función auxiliar de métricas
# -------------------------------------------------------------
def calculate_metrics(y_true, y_pred, n, p):
    """
    Calcula las métricas MSE, MAE, R^2 y R^2 Ajustado.
    n: número de muestras
    p: número de predictores (características)
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Fórmula de R^2 Ajustado: 1 - (1 - R^2) * (n - 1) / (n - p - 1)
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan
        
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Adjusted_R2': adj_r2
    }

# -------------------------------------------------------------
# Paso 3: Entrenar modelos no lineales
# -------------------------------------------------------------
models = {
    'SVR (Linear Kernel)': SVR(kernel='linear'),
    'SVR (RBF Kernel)': SVR(kernel='rbf'),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'HistGradientBoosting Regressor': HistGradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nEntrenando y evaluando el modelo: {name}...")
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcular métricas
    n_train, p_train = X_train.shape
    n_test, p_test = X_test.shape
    
    metrics_train = calculate_metrics(y_train, y_train_pred, n_train, p_train)
    metrics_test = calculate_metrics(y_test, y_test_pred, n_test, p_test)
    
    results.append({
        'Model': name,
        'Train MSE': metrics_train['MSE'],
        'Train MAE': metrics_train['MAE'],
        'Train R²': metrics_train['R2'],
        'Train Adj R²': metrics_train['Adjusted_R2'],
        'Test MSE': metrics_test['MSE'],
        'Test MAE': metrics_test['MAE'],
        'Test R²': metrics_test['R2'],
        'Test Adj R²': metrics_test['Adjusted_R2']
    })

# -------------------------------------------------------------
# Paso 4: Mostrar y guardar resultados no lineales
# -------------------------------------------------------------
df_results = pd.DataFrame(results)

df_sorted = df_results.copy()
df_sorted['Model'] = pd.Categorical(df_sorted['Model'], categories=list(models.keys()), ordered=True)
df_sorted = df_sorted.sort_values(by='Model')

sorted_by_performance = df_results.sort_values(by='Test R%', ascending=False)['Model'].tolist() if 'Test R%' in df_results.columns else df_results.sort_values(by='Test R²', ascending=False)['Model'].tolist()
model_to_rank = {model: idx + 1 for idx, model in enumerate(sorted_by_performance)}

COLOR_MAP = {
    1: "\033[1;92m",
    2: "\033[1;32m",
    3: "\033[1;93m",
    4: "\033[38;5;208m",
    5: "\033[1;31m",
}
RESET = "\033[0m"

df_print = df_sorted.copy()
df_print['Model'] = df_print['Model'].astype(str)

# Calcular Δ MSE (Test MSE - Train MSE)
df_print['Δ MSE'] = df_print['Test MSE'] - df_print['Train MSE']

# Formatear la columna de sobreajuste agregando la señal de alerta
df_print['Δ MSE'] = df_print['Δ MSE'].apply(lambda x: f"{x:.4f} ⚠️" if x > 5.0 else f"{x:.4f}")

table_str = df_print.to_string(index=False, float_format=lambda x: f"{x:.4f}")
lines = table_str.split('\n')

width = len(lines[0])
print("\n" + "="*width)
print("                      RENDIMIENTO DE MODELOS NO LINEALES".center(width))
print("="*width)
print(lines[0])

# Encontrar índices de columnas para el recorte de colores
idx_train_mse = lines[0].find("Train MSE")
idx_test_r2 = lines[0].find("Test R²")
idx_test_adj_r2 = lines[0].find("Test Adj R²")

for line in lines[1:]:
    matched_model = None
    for model in sorted(list(models.keys()), key=len, reverse=True):
        if model in line:
            matched_model = model
            break
    if matched_model:
        rank = model_to_rank.get(matched_model, 99)
        color = COLOR_MAP.get(rank, "")
        if color and idx_train_mse != -1 and idx_test_r2 != -1 and idx_test_adj_r2 != -1:
            part_model = line[:idx_train_mse]
            part_mid = line[idx_train_mse:idx_test_r2]
            part_test_r2 = line[idx_test_r2:idx_test_adj_r2]
            part_end = line[idx_test_adj_r2:]
            print(f"{color}{part_model}{RESET}{part_mid}{color}{part_test_r2}{RESET}{part_end}")
        elif color:
            print(f"{color}{line}{RESET}")
        else:
            print(line)
    else:
        print(line)
print("="*width)

output_dir = os.path.dirname(train_path)
results_csv_path = os.path.join(output_dir, 'nonlinear_model_performance.csv')
df_results.to_csv(results_csv_path, index=False)
print(f"Resultados de rendimiento no lineal guardados en: {results_csv_path}")

# -------------------------------------------------------------
# Paso 5: Fusionar con la comparación JSON de todos los modelos
# -------------------------------------------------------------
json_path = os.path.join(output_dir, 'metrics_comparison.json')

# Cargar datos JSON existentes si el archivo existe
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    except Exception:
        existing_results = []
else:
    existing_results = []

# Fusionar resultados existentes y nuevos por nombre de modelo
merged_dict = {item['Model']: item for item in existing_results}
for item in results:
    merged_dict[item['Model']] = item

# Guardar de vuelta al archivo JSON
all_results = list(merged_dict.values())
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
print(f"Métricas actualizadas en el archivo JSON: {json_path}")



