import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
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

# Asegurar que el directorio de salida exista para los gráficos
output_dir = os.path.dirname(train_path)

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
# Paso 3: Validación Cruzada (K-Fold)
# -------------------------------------------------------------
print("\n" + "="*80)
print("                       VALIDACIÓN CRUZADA (K-Fold, K=5)")
print("="*80)
# Usar KFold para particiones reproducibles
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelos base a validar
baseline_lasso = Lasso(alpha=1.0, random_state=42)
baseline_ridge = Ridge(alpha=1.0, random_state=42)
baseline_rf = RandomForestRegressor(random_state=42)
baseline_hgb = HistGradientBoostingRegressor(random_state=42)

# Evaluar utilizando MSE negativo y R^2
lasso_cv_r2 = cross_val_score(baseline_lasso, X_train, y_train, cv=kf, scoring='r2')
lasso_cv_mse = -cross_val_score(baseline_lasso, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

ridge_cv_r2 = cross_val_score(baseline_ridge, X_train, y_train, cv=kf, scoring='r2')
ridge_cv_mse = -cross_val_score(baseline_ridge, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

rf_cv_r2 = cross_val_score(baseline_rf, X_train, y_train, cv=kf, scoring='r2')
rf_cv_mse = -cross_val_score(baseline_rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

hgb_cv_r2 = cross_val_score(baseline_hgb, X_train, y_train, cv=kf, scoring='r2')
hgb_cv_mse = -cross_val_score(baseline_hgb, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

print(f"Lasso CV (K=5): Promedio R² = {lasso_cv_r2.mean():.4f} (± {lasso_cv_r2.std():.4f}) | Promedio MSE = {lasso_cv_mse.mean():.4f}")
print(f"Ridge CV (K=5): Promedio R² = {ridge_cv_r2.mean():.4f} (± {ridge_cv_r2.std():.4f}) | Promedio MSE = {ridge_cv_mse.mean():.4f}")
print(f"RF CV (K=5):    Promedio R² = {rf_cv_r2.mean():.4f} (± {rf_cv_r2.std():.4f}) | Promedio MSE = {rf_cv_mse.mean():.4f}")
print(f"HGB CV (K=5):   Promedio R² = {hgb_cv_r2.mean():.4f} (± {hgb_cv_r2.std():.4f}) | Promedio MSE = {hgb_cv_mse.mean():.4f}")

# -------------------------------------------------------------
# Paso 4: Optimización de Hiperparámetros con GridSearchCV
# -------------------------------------------------------------
print("\n" + "="*80)
print("                 OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearchCV)")
print("="*80)

# Grilla para Lasso
print("Optimizando Lasso...")
lasso_param_grid = {'alpha': np.logspace(-4, 2, 100)}
grid_lasso = GridSearchCV(Lasso(random_state=42), lasso_param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_lasso.fit(X_train, y_train)
best_lasso = grid_lasso.best_estimator_
print(f"Mejores parámetros de Lasso: {grid_lasso.best_params_}")

# Grilla para Ridge
print("Optimizando Ridge...")
ridge_param_grid = {'alpha': np.logspace(-4, 4, 100)}
grid_ridge = GridSearchCV(Ridge(random_state=42), ridge_param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_ridge.fit(X_train, y_train)
best_ridge = grid_ridge.best_estimator_
print(f"Mejores parámetros de Ridge: {grid_ridge.best_params_}")

# Grilla para Random Forest
print("Optimizando Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"Mejores parámetros de RF: {grid_rf.best_params_}")

# Grilla para HistGradientBoosting
print("Optimizando HistGradientBoosting...")
hgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [50, 100, 150],
    'max_depth': [3, 5, 7, None],
    'l2_regularization': [0.0, 0.1, 1.0, 10.0]
}
grid_hgb = GridSearchCV(HistGradientBoostingRegressor(random_state=42), hgb_param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_hgb.fit(X_train, y_train)
best_hgb = grid_hgb.best_estimator_
print(f"Mejores parámetros de HGB: {grid_hgb.best_params_}")

# -------------------------------------------------------------
# Paso 5: Evaluar Modelos Optimizados
# -------------------------------------------------------------
optimized_models = {
    'Lasso (Optimized)': best_lasso,
    'Ridge (Optimized)': best_ridge,
    'Random Forest (Optimized)': best_rf,
    'HistGradientBoosting (Optimized)': best_hgb
}

new_results = []
n_train, p_train = X_train.shape
n_test, p_test = X_test.shape

for name, model in optimized_models.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics_train = calculate_metrics(y_train, y_train_pred, n_train, p_train)
    metrics_test = calculate_metrics(y_test, y_test_pred, n_test, p_test)
    
    new_results.append({
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

# Guardar nuevos resultados en las comparaciones JSON
json_path = os.path.join(output_dir, 'metrics_comparison.json')
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    except Exception:
        existing_results = []
else:
    existing_results = []

merged_dict = {item['Model']: item for item in existing_results}
for item in new_results:
    merged_dict[item['Model']] = item

all_results = list(merged_dict.values())
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

df_all = pd.DataFrame(all_results)
# Ordenar por Test R2
df_all_sorted = df_all.sort_values(by='Test R²', ascending=False)

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

existing_models = [m for m in CREATION_ORDER if m in df_all['Model'].values]
for m in df_all['Model'].values:
    if m not in existing_models:
        existing_models.append(m)

df_sorted = df_all.copy()
df_sorted['Model'] = pd.Categorical(df_sorted['Model'], categories=existing_models, ordered=True)
df_sorted = df_sorted.sort_values(by='Model')

sorted_by_performance = df_all.sort_values(by='Test R²', ascending=False)['Model'].tolist()
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
print("               COMPARATIVA GLOBAL DE RENDIMIENTO DE MODELOS (INCLUYENDO OPTIMIZADOS)".center(width))
print("="*width)
print(lines[0])

# Encontrar índices de columnas para el recorte de colores
idx_train_mse = lines[0].find("Train MSE")
idx_test_r2 = lines[0].find("Test R²")
idx_test_adj_r2 = lines[0].find("Test Adj R²")

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

# -------------------------------------------------------------
# Paso 6: Identificar el Mejor Modelo Optimizado y Analizar Importancia de Variables
# -------------------------------------------------------------
# Obtener el mejor modelo general (primero en la lista ordenada)
best_overall_row = df_all_sorted.iloc[0]
best_model_name = best_overall_row['Model']

print(f"\n🏆 Mejor Modelo General: {best_model_name}")

# Asignar el nombre del modelo al objeto real para el análisis de errores
if 'Lasso' in best_model_name:
    if 'Optimized' in best_model_name:
        best_model_obj = best_lasso
        print(f"Alpha Óptimo: {grid_lasso.best_params_['alpha']:.4f}")
    else:
        from sklearn.linear_model import Lasso
        best_model_obj = Lasso(alpha=1.0, random_state=42).fit(X_train, y_train)
        print("Alpha: 1.0")
    # Extraer coeficientes
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model_obj.coef_
    })
    # Ordenar por magnitud absoluta del coeficiente
    importances['Abs_Importance'] = importances['Importance'].abs()
    importances = importances.sort_values(by='Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])
elif 'Ridge' in best_model_name:
    if 'Optimized' in best_model_name:
        best_model_obj = best_ridge
        print(f"Alpha Óptimo: {grid_ridge.best_params_['alpha']:.4f}")
    else:
        from sklearn.linear_model import Ridge
        best_model_obj = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
        print("Alpha: 1.0")
    # Extraer coeficientes
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model_obj.coef_
    })
    # Ordenar por magnitud absoluta del coeficiente
    importances['Abs_Importance'] = importances['Importance'].abs()
    importances = importances.sort_values(by='Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])
elif 'HistGradientBoosting' in best_model_name:
    if 'Optimized' in best_model_name:
        best_model_obj = best_hgb
        print(f"Parámetros Óptimos: {grid_hgb.best_params_}")
    else:
        best_model_obj = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
        print("Parámetros: Por defecto")
    # HistGradientBoosting no tiene feature_importances_ incorporado. Usamos permutation_importance
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_model_obj, X_train, y_train, n_repeats=10, random_state=42)
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': r.importances_mean
    }).sort_values(by='Importance', ascending=False)
else:
    if 'Optimized' in best_model_name:
        best_model_obj = best_rf
        print(f"Parámetros Óptimos: {grid_rf.best_params_}")
    else:
        best_model_obj = RandomForestRegressor(random_state=42).fit(X_train, y_train)
        print("Parámetros: Por defecto")
    # Extraer importancia de características
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model_obj.feature_importances_
    }).sort_values(by='Importance', ascending=False)

print("\n--- Importancia de Variables ---")
print(importances.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Guardar importancias a CSV
importances.to_csv(os.path.join(output_dir, 'best_model_feature_importances.csv'), index=False)

# -------------------------------------------------------------
# Paso 7: Gráfico de Residuos y Análisis de Errores
# -------------------------------------------------------------
print("Generando gráficos de residuos para el mejor modelo...")
y_test_pred = best_model_obj.predict(X_test)
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Reales vs Predichos
sns.scatterplot(x=y_test, y=y_test_pred, ax=axes[0], color='dodgerblue', alpha=0.8, edgecolor='w', s=60)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ajuste Perfecto (Y = X)')
axes[0].set_title('Reales vs. Predichos (Set de Test)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Grasa Corporal Real (%)', fontsize=11)
axes[0].set_ylabel('Grasa Corporal Predicha (%)', fontsize=11)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Gráfico 2: Residuos vs Predichos
sns.scatterplot(x=y_test_pred, y=residuals, ax=axes[1], color='coral', alpha=0.8, edgecolor='w', s=60)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_title('Residuos vs. Predichos', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Valores Predichos (%)', fontsize=11)
axes[1].set_ylabel('Residuos (Real - Predicho)', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle(f"Análisis de Residuos - {best_model_name}", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

plot_path = os.path.join(output_dir, '03_residuos_mejor_modelo.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Gráfico de residuos guardado exitosamente en: {plot_path}")
