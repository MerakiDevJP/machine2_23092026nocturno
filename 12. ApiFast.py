from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = FastAPI(title="API Neurona MP - Cáncer de Mama")

# 1. Cargar datos y seleccionar las 5 características más importantes
cancer_mama = load_breast_cancer()
df_full = pd.DataFrame(cancer_mama.data, columns=cancer_mama.feature_names)

# Seleccionamos explícitamente las 5 características clave
top_features = [
    'worst area',
    'worst concave points',
    'mean concave points',
    'worst radius', 
    'worst perimeter'
]
df = df_full[top_features]
Y = cancer_mama.target

# 2. Dividir y binarizar los datos para la Neurona de McCulloch-Pitts
X_train, X_test, y_train, y_test = train_test_split(df, Y, stratify=Y, random_state=42)

X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1, 0])
X_test_bin = X_test.apply(pd.cut, bins=2, labels=[1, 0])

# 3. Implementación de la Neurona MP
class MpNeurona:
    def __init__(self):
        self.threshold = None
        
    def model(self, x):
        z = sum(x)
        return (z >= self.threshold)
    
    def predict(self, X):
        Y_preds = []
        for x in X:
            result = self.model(x)
            Y_preds.append(result)
        return np.array(Y_preds)
    
    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        self.threshold = max(accuracy, key=accuracy.get)

# Entrenar el modelo automáticamente al iniciar la API
np_neurona = MpNeurona()
np_neurona.fit(X_train_bin.to_numpy(), y_train)

# Esquema de datos de entrada para la API
class PatientData(BaseModel):
    worst_radius: float
    worst_perimeter: float
    worst_concave_points: float
    worst_area: float
    mean_concave_points: float

# Ruta principal que sirve la interfaz HTML
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Predicción de Cáncer de Mama - Neurona MP</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-7">
                    <div class="card shadow">
                        <div class="card-header bg-primary text-white">
                            <h4 class="mb-0">Evaluador de Cáncer de Mama (5 Características Clave)</h4>
                        </div>
                        <div class="card-body">
                            <form id="predictionForm">
                                <div class="mb-3">
                                    <label class="form-label">Worst Radius (Radio Máximo):</label>
                                    <input type="number" step="any" class="form-control" id="worst_radius" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Worst Perimeter (Perímetro Máximo):</label>
                                    <input type="number" step="any" class="form-control" id="worst_perimeter" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Worst Concave Points (Puntos Cóncavos Máximos):</label>
                                    <input type="number" step="any" class="form-control" id="worst_concave_points" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Worst Area (Área Máxima):</label>
                                    <input type="number" step="any" class="form-control" id="worst_area" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Mean Concave Points (Media de Puntos Cóncavos):</label>
                                    <input type="number" step="any" class="form-control" id="mean_concave_points" required>
                                </div>
                                <button type="submit" class="btn btn-success w-100">Predecir Diagnóstico</button>
                            </form>
                            
                            <div id="resultado" class="mt-4" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const data = {
                    worst_radius: parseFloat(document.getElementById('worst_radius').value),
                    worst_perimeter: parseFloat(document.getElementById('worst_perimeter').value),
                    worst_concave_points: parseFloat(document.getElementById('worst_concave_points').value),
                    worst_area: parseFloat(document.getElementById('worst_area').value),
                    mean_concave_points: parseFloat(document.getElementById('mean_concave_points').value)
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const resDiv = document.getElementById('resultado');
                resDiv.style.display = 'block';
                
                if (result.prediction === "Benigno") {
                    resDiv.className = "alert alert-success";
                    resDiv.innerHTML = `<strong>Resultado:</strong> ${result.prediction} (Umbral superado favorablemente).`;
                } else {
                    resDiv.className = "alert alert-danger";
                    resDiv.innerHTML = `<strong>Resultado:</strong> ${result.prediction} (Requiere revisión clínica).`;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

# Ruta de la API para procesar la predicción
@app.post("/predict")
def predict(patient: PatientData):
    # Convertir los 5 valores ingresados en un DataFrame de 1 fila
    input_data = pd.DataFrame([[
        patient.worst_radius,
        patient.worst_perimeter,
        patient.worst_concave_points,
        patient.worst_area,
        patient.mean_concave_points
    ]], columns=top_features)
    
    # Binarizar usando los mismos criterios (bins=2)
    input_bin = input_data.apply(pd.cut, bins=2, labels=[1, 0])
    
    # Realizar predicción con la neurona MP
    pred = np_neurona.predict(input_bin.to_numpy())[0]
    
    # En el dataset de sklearn: 1 suele ser Benigno y 0 Maligno (o viceversa según el mapeo target)
    diagnosis = "Benigno" if pred else "Maligno"
    
    return {"prediction": diagnosis, "raw_output": bool(pred)}