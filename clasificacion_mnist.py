import sys
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_orig_re_search = re.search
def _safe_re_search(pattern, string, flags=0):
    if not isinstance(string, (str, bytes, bytearray)):
        string = str(string)
    return _orig_re_search(pattern, string, flags)
re.search = _safe_re_search

from pandas.core.arrays.categorical import Categorical
_orig_rename_categories = Categorical.rename_categories
def _safe_rename_categories(self, new_categories, *args, **kwargs):
    if callable(new_categories):
        orig_func = new_categories
        def _wrapped(val):
            try:
                return orig_func(val)
            except (TypeError, AttributeError):
                return str(val)
        return _orig_rename_categories(self, _wrapped, *args, **kwargs)
    return _orig_rename_categories(self, new_categories, *args, **kwargs)
Categorical.rename_categories = _safe_rename_categories

import sklearn
import sklearn.datasets._arff_parser
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("=" * 80)
print("1. ENTORNO DE EJECUCIÓN Y VERSIONES REGISTRADAS")
print("=" * 80)
print(f"Versión de Python      : {sys.version.split()[0]}")
print(f"Versión de scikit-learn: {sklearn.__version__}")
print("=" * 80 + "\n")

print("Descargando / Procesando MNIST (OpenML ID: 554)... (espera un momento)")
X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)

X = X.astype("float32") / 255.0
y = y.astype("int64")

def tomar_muestra(X_base, y_base, n):
    X_muestra, _, y_muestra, _ = train_test_split(
        X_base, y_base, train_size=n, stratify=y_base, random_state=42
    )
    return X_muestra, y_muestra

X_ent, y_ent = tomar_muestra(X[:60000], y[:60000], 10000)
X_pru, y_pru = tomar_muestra(X[60000:], y[60000:], 2000)

print(f"Dimensiones de X_ent: {X_ent.shape} | y_ent: {y_ent.shape}")
print(f"Dimensiones de X_pru: {X_pru.shape}  | y_pru: {y_pru.shape}\n")

distribucion = pd.DataFrame({
    "Dígito": list(range(10)),
    "Frecuencia Entrenamiento": np.bincount(y_ent),
    "Proporción Ent. (%)": (np.bincount(y_ent) / len(y_ent)) * 100.0,
    "Frecuencia Prueba": np.bincount(y_pru),
    "Proporción Pru. (%)": (np.bincount(y_pru) / len(y_pru)) * 100.0,
})
print("Distribución de clases por conjunto (Muestreo Estratificado):")
print(distribucion.to_string(index=False))
print("-" * 80)

fig1, axes1 = plt.subplots(1, 5, figsize=(10, 2.5))
fig1.canvas.manager.set_window_title("Muestra de 5 dígitos de MNIST")
for i in range(5):
    axes1[i].imshow(X_ent[i].reshape(28, 28), cmap="gray")
    axes1[i].set_title(f"Dígito: {y_ent[i]}", fontsize=11, fontweight="bold")
    axes1[i].axis("off")
plt.tight_layout()
fig1.savefig("figura1_muestras_mnist.png", dpi=300)
print("-> Figura 1 guardada como 'figura1_muestras_mnist.png'")

modelos = {
    "Árbol de decisión": DecisionTreeClassifier(
        max_depth=15, 
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    ),
    "k-NN": KNeighborsClassifier(
        n_neighbors=5, 
        metric="euclidean", 
        weights="uniform", 
        n_jobs=-1
    )
}

tiempos_entrenamiento = {}
predicciones_ent = {}
predicciones_pru = {}

print("\n" + "=" * 80)
print("2. ENTRENAMIENTO DE LOS MODELOS (fit)")
print("=" * 80)

for nombre, modelo in modelos.items():
    print(f"Ajustando {nombre} sobre {X_ent.shape[0]} muestras...")
    t0 = time.time()
    modelo.fit(X_ent, y_ent)
    t1 = time.time()
    duracion = t1 - t0
    tiempos_entrenamiento[nombre] = duracion
    print(f" -> {nombre} ajustado en {duracion:.2f} segundos.")

filas_tabla_2 = []

for nombre, modelo in modelos.items():
    y_pred_ent = modelo.predict(X_ent)
    y_pred_pru = modelo.predict(X_pru)
    
    predicciones_ent[nombre] = y_pred_ent
    predicciones_pru[nombre] = y_pred_pru
    
    acc_ent = accuracy_score(y_ent, y_pred_ent) * 100.0
    err_ent = 100.0 - acc_ent
    acc_pru = accuracy_score(y_pru, y_pred_pru) * 100.0
    err_pru = 100.0 - acc_pru
    
    brecha = acc_ent - acc_pru
    
    filas_tabla_2.append({
        "Modelo": nombre,
        "Exactitud ent. (%)": round(acc_ent, 2),
        "Error ent. (%)": round(err_ent, 2),
        "Exactitud prueba (%)": round(acc_pru, 2),
        "Error prueba (%)": round(err_pru, 2),
        "Brecha (p. p.)": round(brecha, 2)
    })

df_tabla_2 = pd.DataFrame(filas_tabla_2)

print("\n" + "=" * 80)
print("TABLA 2. RESULTADOS DE ENTRENAMIENTO Y PRUEBA")
print("=" * 80)
print(df_tabla_2.to_string(index=False))
print("=" * 80)

for _, r in df_tabla_2.iterrows():
    assert np.isclose(r["Exactitud ent. (%)"] + r["Error ent. (%)"], 100.0), "Error de suma en entrenamiento"
    assert np.isclose(r["Exactitud prueba (%)"] + r["Error prueba (%)"], 100.0), "Error de suma en prueba"
print("Verificación formal superada: Exactitud + Error = 100.00% en todos los clasificadores.\n")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.canvas.manager.set_window_title("Matrices de Confusión de Prueba")

matrices_cm = {}

for idx, (nombre, y_pred) in enumerate(predicciones_pru.items()):
    cm = confusion_matrix(y_pru, y_pred, labels=list(range(10)))
    matrices_cm[nombre] = cm
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=axes2[idx], cmap="Blues", colorbar=False, values_format="d")
    axes2[idx].set_title(f"{nombre}", fontsize=12, fontweight="bold")
    axes2[idx].set_xlabel("Predicción")
    axes2[idx].set_ylabel("Etiqueta Real" if idx == 0 else "")

plt.tight_layout()
fig2.savefig("figura2_matrices_confusion.png", dpi=300)
print("-> Figura 2 guardada como 'figura2_matrices_confusion.png'")

print("\n" + "=" * 80)
print("CONFUSIONES MÁS FRECUENTES POR MODELO (Sección 2.7)")
print("=" * 80)

for nombre, cm in matrices_cm.items():
    cm_sin_diagonal = cm.copy()
    np.fill_diagonal(cm_sin_diagonal, 0)
    
    idx_max = np.argmax(cm_sin_diagonal)
    real_f, pred_f = np.unravel_index(idx_max, cm_sin_diagonal.shape)
    conteo_fallos = cm_sin_diagonal[real_f, pred_f]
    
    print(f"[{nombre}] Confusión más frecuente: Dígito {real_f} clasificado como {pred_f} -> ({conteo_fallos} casos)")

indices_fallos = []
for i in range(len(y_pru)):
    fallo = (
        predicciones_pru["Árbol de decisión"][i] != y_pru[i] or
        predicciones_pru["Random Forest"][i] != y_pru[i] or
        predicciones_pru["k-NN"][i] != y_pru[i]
    )
    if fallo:
        indices_fallos.append(i)

idx_ejemplo = indices_fallos[0]
img_ejemplo = X_pru[idx_ejemplo].reshape(28, 28)
real_val = y_pru[idx_ejemplo]
pred_dt = predicciones_pru["Árbol de decisión"][idx_ejemplo]
pred_rf = predicciones_pru["Random Forest"][idx_ejemplo]
pred_knn = predicciones_pru["k-NN"][idx_ejemplo]

fig3, ax3 = plt.subplots(figsize=(4, 4))
fig3.canvas.manager.set_window_title("Inspección de Imagen de Prueba Mal Clasificada")
ax3.imshow(img_ejemplo, cmap="gray")
titulo_error = (
    f"Etiqueta Real: {real_val}\n"
    f"Pred. Árbol: {pred_dt} | Pred. RF: {pred_rf} | Pred. k-NN: {pred_knn}"
)
ax3.set_title(titulo_error, fontsize=10, fontweight="bold", pad=10)
ax3.axis("off")
plt.tight_layout()
fig3.savefig("figura3_caso_mal_clasificado.png", dpi=300)
print("-> Figura 3 guardada como 'figura3_caso_mal_clasificado.png'\n")

print("=" * 80)
print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
print("=" * 80)

plt.show()