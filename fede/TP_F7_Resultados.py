import os
import json
import pandas as pd

# -------------------------------------------------------------
# Resolución de ruta para el archivo JSON de métricas
# -------------------------------------------------------------
# Resolver la ruta basándose en la ubicación de este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(SCRIPT_DIR, 'output', 'metrics_comparison.json')

if not os.path.exists(json_path):
    possible_dirs = [
        'output',
        '../output',
        'fede/output',
        '/home/fzacchigna/ceia/2-amq/TP_ADM/output',
        '/home/fzacchigna/ceia/2-amq/TP_ADM/fede/output'
    ]
    json_path = None
    for d in possible_dirs:
        p = os.path.join(d, 'metrics_comparison.json')
        if os.path.exists(p):
            json_path = p
            break

if json_path is None or not os.path.exists(json_path):
    print("❌ Error: No se encontró el archivo 'metrics_comparison.json'.")
    print("Asegúrate de haber ejecutado los scripts de modelado primero:")
    print("  - TP_F4_ModelosLineales.py")
    print("  - TP_F5_ModelosNoLineales.py")
    exit(1)

# -------------------------------------------------------------
# Cargar y mostrar resultados
# -------------------------------------------------------------
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Ordenar y mostrar resultados según el orden de creación, coloreado según el rango de rendimiento
CREATION_ORDER = [
    'Dummy Regressor (Mean)',
    'Linear Regression',
    'Lasso (alpha=1)',
    'Ridge (alpha=1)',
    'SVR (Linear Kernel)',
    'SVR (RBF Kernel)',
    'Random Forest Regressor',
    'HistGradientBoosting Regressor',
    'Lasso (Optimized)',
    'Ridge (Optimized)',
    'Random Forest (Optimized)',
    'HistGradientBoosting (Optimized)'
]

# Asegurar que todos los modelos del df estén en las categorías
existing_models = [m for m in CREATION_ORDER if m in df['Model'].values]
for m in df['Model'].values:
    if m not in existing_models:
        existing_models.append(m)

df_sorted = df.copy()
df_sorted['Model'] = pd.Categorical(df_sorted['Model'], categories=existing_models, ordered=True)
df_sorted = df_sorted.sort_values(by='Model')

# Obtener ranking basado en el rendimiento (Test R² descendente)
sorted_by_performance = df.sort_values(by='Test R%', ascending=False)['Model'].tolist() if 'Test R%' in df.columns else df.sort_values(by='Test R²', ascending=False)['Model'].tolist()
model_to_rank = {model: idx + 1 for idx, model in enumerate(sorted_by_performance)}

# Colores: Rango 1 = Verde, 2 = Menos Verde, 3 = Amarillo, 4 = Naranja, 5 = Rojo, Resto = Blanco
COLOR_MAP = {
    1: "\033[1;92m",      # Verde claro (Negrita Verde Claro)
    2: "\033[1;32m",      # Verde oscuro (Negrita Verde)
    3: "\033[1;93m",      # Amarillo (Negrita Amarillo Claro)
    4: "\033[38;5;208m",   # Naranja (Negrita Naranja)
    5: "\033[1;31m",      # Rojo (Negrita Rojo)
}
RESET = "\033[0m"

df_print = df_sorted.copy()
df_print['Model'] = df_print['Model'].astype(str)

# Calcular Δ MSE (Test MSE - Train MSE)
df_print['Δ MSE'] = df_print['Test MSE'] - df_print['Train MSE']

mapping = {
    'Lasso (Optimized)': 'Lasso (alpha=1)',
    'Ridge (Optimized)': 'Ridge (alpha=1)',
    'Random Forest (Optimized)': 'Random Forest Regressor',
    'HistGradientBoosting (Optimized)': 'HistGradientBoosting Regressor'
}

for opt_name, base_name in mapping.items():
    if opt_name in df_print['Model'].values and base_name in df_print['Model'].values:
        opt_r2 = df_print.loc[df_print['Model'] == opt_name, 'Test R²'].values[0]
        base_r2 = df_print.loc[df_print['Model'] == base_name, 'Test R²'].values[0]
        marker = " (+)" if opt_r2 > base_r2 else " (-)"
        df_print.loc[df_print['Model'] == opt_name, 'Model'] = f"{opt_name}{marker}"

# Formatear la columna de sobreajuste agregando la señal de alerta
df_print['Δ MSE'] = df_print['Δ MSE'].apply(lambda x: f"{x:.4f} ⚠️" if x > 5.0 else f"{x:.4f}")

table_str = df_print.to_string(index=False, float_format=lambda x: f"{x:.4f}")
lines = table_str.split('\n')

width = len(lines[0])
print("\n" + "="*width)
print("                    COMPARATIVA GLOBAL DE MODELOS (ORDEN DE CREACIÓN)".center(width))
print("="*width)
# Imprimir encabezado
print(lines[0])

# Encontrar índices de columnas para el recorte de colores
idx_train_mse = lines[0].find("Train MSE")
idx_test_r2 = lines[0].find("Test R²")
idx_test_adj_r2 = lines[0].find("Test Adj R²")

# Imprimir cada fila coloreada por rango (solo las columnas Model y Test R2)
for line in lines[1:]:
    matched_model = None
    for model in sorted(existing_models, key=len, reverse=True):
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

# Resaltar el mejor modelo
best_model_name = sorted_by_performance[0]
best_model = df[df['Model'] == best_model_name].iloc[0]
print(f"\n🏆 El mejor modelo actual es '{best_model['Model']}' con:")
print(f"   - Test R²: {best_model['Test R²']:.4f} (Train R²: {best_model['Train R²']:.4f})")
print(f"   - Test Adj R²: {best_model['Test Adj R²']:.4f} (Train Adj R²: {best_model['Train Adj R²']:.4f})")
print(f"   - Test MSE: {best_model['Test MSE']:.4f} (Train MSE: {best_model['Train MSE']:.4f})")
print(f"   - Test MAE: {best_model['Test MAE']:.4f} (Train MAE: {best_model['Train MAE']:.4f})")
print("="*width + "\n")
