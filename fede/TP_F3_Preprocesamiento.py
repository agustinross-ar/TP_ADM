import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolver rutas absolutas basándose en la ubicación de este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Asegurar que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Carga de Datos Base y Limpieza de Anomalías
# -------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'bodyfat.csv')
if not os.path.exists(csv_path):
    csv_path = 'bodyfat.csv'
if not os.path.exists(csv_path):
    csv_path = '../bodyfat.csv'
if not os.path.exists(csv_path):
    csv_path = 'TP_ADM/bodyfat.csv'
if not os.path.exists(csv_path):
    csv_path = '/home/fzacchigna/ceia/2-amq/TP_ADM/bodyfat.csv'

print(f"Cargando dataset desde: {csv_path}")
df = pd.read_csv(csv_path)

# Limpieza de Anomalías de la Fase 2:
# (Toda la limpieza de anomalías biológicas y outliers se delega al Paso 3 dentro de remove_outliers)
df_clean = df.copy()

# -------------------------------------------------------------
# Paso 0: Eliminar la columna Density
# -------------------------------------------------------------
print("\n--- Paso 0: Eliminando la columna Density ---")
df_prep = df_clean.drop(columns="Density")
print(f"Columnas del dataset luego de eliminar Density: {list(df_prep.columns)}")

# -------------------------------------------------------------
# Paso 1: División 70/30 (Train/Test)
# -------------------------------------------------------------
print("\n--- Paso 1: Dividiendo los datos (70/30) ---")
X = df_prep.drop(columns="BodyFat")
y = df_prep["BodyFat"]

# Uso de random_state=42 para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"Forma del conjunto de entrenamiento: Características: {X_train.shape}, Target: {y_train.shape}")
print(f"Forma del conjunto de prueba:        Características: {X_test.shape}, Target: {y_test.shape}")

# -------------------------------------------------------------
# Paso 2: Imprimir boxplots para todas las columnas de train
# -------------------------------------------------------------
print("\n--- Paso 2: Generando boxplots iniciales ---")
# Combinar y_train y X_train para visualizar todas las columnas del conjunto de entrenamiento
df_train = pd.concat([y_train, X_train], axis=1)

# Graficar boxplots en una grilla
cols = df_train.columns
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df_train[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('')
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Eliminar ejes no utilizados
for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Boxplots - Datos de Entrenamiento (Con Outliers)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path_1 = os.path.join(OUTPUT_DIR, "01_boxplots_con_outliers.png")
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Boxplots iniciales guardados en {plot_path_1}")

# -------------------------------------------------------------
# Paso 3: Eliminación de Outliers
# -------------------------------------------------------------
print("\n--- Paso 3: Eliminación de outliers ---")
def remove_outliers(X_df, y_df):
    """
    Elimina anomalías biológicas y outliers del dataset:
    - Remueve observaciones con BodyFat < 2%
    - Remueve observaciones con Height < 30 pulgadas
    """
    # Combinar target y características temporalmente para filtrar de forma consistente
    df_temp = pd.concat([y_df, X_df], axis=1)
    
    before_size = len(df_temp)
    # 1. Eliminar filas con BodyFat < 2%
    df_temp = df_temp[df_temp['BodyFat'] >= 2]
    after_bf = len(df_temp)
    
    # 2. Eliminar outliers de altura (Height < 30 pulgadas)
    df_temp = df_temp[df_temp['Height'] >= 30]
    after_height = len(df_temp)
    
    print(f"Resumen de eliminación de outliers:")
    print(f"  - Tamaño original: {before_size}")
    print(f"  - Tamaño luego del filtro de BodyFat >= 2%: {after_bf} (eliminadas {before_size - after_bf} filas)")
    print(f"  - Tamaño luego del filtro de Height >= 30\": {after_height} (eliminadas {after_bf - after_height} filas)")
    
    # Separar nuevamente características y target
    X_clean = df_temp.drop(columns=[y_df.name])
    y_clean = df_temp[y_df.name]
    
    return X_clean, y_clean

X_train_clean, y_train_clean = remove_outliers(X_train, y_train)

# -------------------------------------------------------------
# Paso 4: Imprimir boxplots sin outliers (comparativa)
# -------------------------------------------------------------
print("\n--- Paso 4: Generando boxplots comparativos ---")
fig, axes = plt.subplots(4, 4, figsize=(18, 18))
axes = axes.flatten()

# Recombinar target y características para datos limpios
df_train_clean = pd.concat([y_train_clean, X_train_clean], axis=1)

for i, col in enumerate(cols):
    # Crear dataframe comparativo para esta columna
    comp_df = pd.DataFrame({
        'Con Outliers': df_train[col],
        'Sin Outliers': df_train_clean[col]
    })
    sns.boxplot(data=comp_df, ax=axes[i], palette="Set2")
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].grid(True, linestyle='--', alpha=0.5)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Comparativa de Boxplots: Con vs. Sin Outliers", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path_2 = os.path.join(OUTPUT_DIR, "02_boxplots_comparacion.png")
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Boxplots comparativos guardados en {plot_path_2}")

# -------------------------------------------------------------
# Paso 5: Ajustar escalador (StandardScaler - solo con train)
# -------------------------------------------------------------
print("\n--- Paso 5: Ajustando StandardScaler en características de entrenamiento ---")
scaler = StandardScaler()
# Nota: solo ajustamos el escalador en las características (excluyendo el target y)
scaler.fit(X_train_clean)
print("StandardScaler ajustado sobre las características limpias de entrenamiento.")

# -------------------------------------------------------------
# Paso 6: Aplicar escalador a los datos de entrenamiento (train)
# -------------------------------------------------------------
print("\n--- Paso 6: Aplicando escalador a datos de entrenamiento ---")
X_train_scaled = scaler.transform(X_train_clean)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_clean.columns, index=X_train_clean.index)
print("Características de entrenamiento escaladas.")

# -------------------------------------------------------------
# Paso 7: Aplicar escalador a los datos de prueba (test)
# -------------------------------------------------------------
print("\n--- Paso 7: Aplicando escalador de entrenamiento a datos de prueba ---")
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
print("Características de prueba escaladas.")

# Mostrar estadísticas del escalado para verificar corrección
print("\n--- Verificación de Preprocesamiento de la Fase 3 ---")
print("\nPrimeras 3 filas de características escaladas de entrenamiento:")
print(X_train_scaled_df.head(3))
print("\nEstadísticas de escala (Train - Media debería ser aprox 0, Desv. Est. aprox 1):")
stats_train = pd.DataFrame({
    'Media': X_train_scaled_df.mean(),
    'Desv. Est.': X_train_scaled_df.std()
})
print(stats_train.round(4))

print("\nEstadísticas de escala (Test - Media y Desv. Est. variarán levemente):")
stats_test = pd.DataFrame({
    'Media': X_test_scaled_df.mean(),
    'Desv. Est.': X_test_scaled_df.std()
})
print(stats_test.round(4))

# -------------------------------------------------------------
# Paso 8: Guardar datasets preprocesados
# -------------------------------------------------------------
print("\n--- Paso 8: Guardando datasets preprocesados ---")

# Combinar target y características
train_clean_unscaled = pd.concat([y_train_clean, X_train_clean], axis=1)
test_clean_unscaled = pd.concat([y_test, X_test], axis=1)

train_clean_scaled = pd.concat([y_train_clean, X_train_scaled_df], axis=1)
test_clean_scaled = pd.concat([y_test, X_test_scaled_df], axis=1)

# Rutas de salida
train_unscaled_path = os.path.join(OUTPUT_DIR, "train_clean_unscaled.csv")
test_unscaled_path = os.path.join(OUTPUT_DIR, "test_clean_unscaled.csv")
train_scaled_path = os.path.join(OUTPUT_DIR, "train_clean_scaled.csv")
test_scaled_path = os.path.join(OUTPUT_DIR, "test_clean_scaled.csv")

# Guardar a CSV
train_clean_unscaled.to_csv(train_unscaled_path, index=False)
test_clean_unscaled.to_csv(test_unscaled_path, index=False)
train_clean_scaled.to_csv(train_scaled_path, index=False)
test_clean_scaled.to_csv(test_scaled_path, index=False)

print(f"Datasets no escalados guardados en:\n - {train_unscaled_path}\n - {test_unscaled_path}")
print(f"Datasets escalados guardados en:\n - {train_scaled_path}\n - {test_scaled_path}")
