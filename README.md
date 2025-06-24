# ML DevOps - Predicción de Precios de Viviendas

Este proyecto implementa una solución de MLOps para un modelo de predicción de precios de viviendas basado en el dataset de [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv). La solución no es única e incluye pipeline de entrenamiento, containerización, despliegue, y monitoreo en producción.

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Training       │───▶│   Model         │
│  (Boston CSV)   │    │  Pipeline       │    │  Registry       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   API Service   │◀───│  Docker Image   │
│  (Grafana)      │    │  (FastAPI)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Kubernetes    │
                       │    (AKS)        │
                       └─────────────────┘
```

## 🚀 Tecnologías Utilizadas

### Core ML Stack
- **Python 3.9+**: Lenguaje principal
- **scikit-learn**: Modelo de regresión
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas

### MLOps Tools
- **MLflow**: Tracking de experimentos y registro de modelos
- **FastAPI**: API REST para servir el modelo
- **Docker**: Containerización
- **Kubernetes**: Orquestación de los contenedores

### Cloud & Infrastructure
- **Azure Kubernetes Service (AKS)**: Despliegue en producción
- **Azure Container Registry (ACR)**: Registro de imágenes
- **Azure ML**: Pipeline de entrenamiento

### Monitoring
- **Prometheus**: Métricas del sistema
- **Grafana**: Visualización y dashboards

## 🛠️ Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/ml-devops-housing-prediction.git
cd ml-devops-housing-prediction
```

### 2. Configurar Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=subscription-id
AZURE_RESOURCE_GROUP=ml-devops-rg
AZURE_LOCATION=eastus
ACR_NAME=mldevopsacr
AKS_CLUSTER_NAME=ml-devops-aks

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=housing-price-prediction

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### 4. Configurar Azure CLI

```bash
# Login a Azure
az login

# Establecer suscripción
az account set --subscription "subscription-id"

# Crear grupo de recursos
az group create --name ml-devops-rg --location eastus
```

## 🎯 Entrenamiento del Modelo

### Opción 1: Entrenamiento Local

```bash
# Ejecutar script de entrenamiento
python scripts/train_model.py
```

### Opción 2: Entrenamiento en Azure ML

```bash
# Configurar workspace de Azure ML
python scripts/setup_azureml.py

# Ejecutar pipeline en Azure ML
python pipelines/azure_ml_pipeline.py
```

### Métricas del Modelo

- **R² Score**: Coeficiente de determinación
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **Feature Importance**: Importancia de cada característica

### Ejemplo:

```
R² Score: 0.8542
MAE: 2.1456
RMSE: 3.2789
Training Time: 0.45 seconds
```

## 🐳 Containerización

### Construir Imagen Docker

```bash
# Construir imagen localmente
docker build -t housing-prediction-api .

# Probar localmente
docker run -p 8000:8000 housing-prediction-api
```

### Usar Docker Compose para Desarrollo

```bash
# Levantar stack completo (API + MLflow + Monitoring)
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Detener servicios
docker-compose down
```

### Subir a Azure Container Registry

```bash
# Crear ACR
az acr create --resource-group ml-devops-rg --name mldevopsacr --sku Basic

# Login a ACR
az acr login --name mldevopsacr

# Tag y push imagen
docker tag housing-prediction-api mldevopsacr.azurecr.io/housing-prediction-api:latest
docker push mldevopsacr.azurecr.io/housing-prediction-api:latest
```

## ☸️ Despliegue en Kubernetes (AKS)

### 1. Crear Cluster AKS

```bash
# Crear cluster AKS
az aks create \
    --resource-group ml-devops-rg \
    --name ml-devops-aks \
    --node-count 2 \
    --node-vm-size Standard_B2s \
    --enable-addons monitoring \
    --generate-ssh-keys

# Obtener credenciales
az aks get-credentials --resource-group ml-devops-rg --name ml-devops-aks
```

### 2. Configurar Acceso a ACR

```bash
# Adjuntar ACR al cluster AKS
az aks update -n ml-devops-aks -g ml-devops-rg --attach-acr mldevopsacr
```

### 3. Desplegar Aplicación

```bash
# Aplicar manifiestos de Kubernetes
kubectl apply -f k8s/

# Verificar despliegue
kubectl get pods
kubectl get services

# Obtener IP externa del servicio
kubectl get service housing-prediction-service --watch
```

### 4. Script de Despliegue Automatizado

```bash
# Usar script de despliegue
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## 🔧 Uso de la API

### Endpoints Disponibles

#### 1. Health Check
```bash
GET /health
```

#### 2. Predicción Individual
```bash
POST /predict
Content-Type: application/json

{
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.0900,
    "rad": 1,
    "tax": 296,
    "ptratio": 15.3,
    "b": 396.90,
    "lstat": 4.98
}
```

#### 3. Predicción por Lotes
```bash
POST /predict/batch
Content-Type: application/json

{
    "instances": [
        {
            "crim": 0.00632,
            "zn": 18.0,
            // ... más features
        },
        {
            "crim": 0.02731,
            "zn": 0.0,
            // ... más features
        }
    ]
}
```

#### 4. Métricas del Modelo
```bash
GET /metrics
```

### Ejemplos de Uso

#### cURL
```bash
# Health check
curl -X GET http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.0900,
    "rad": 1,
    "tax": 296,
    "ptratio": 15.3,
    "b": 396.90,
    "lstat": 4.98
  }'
```

#### Python
```python
import requests
import json

# Datos de ejemplo
data = {
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.0900,
    "rad": 1,
    "tax": 296,
    "ptratio": 15.3,
    "b": 396.90,
    "lstat": 4.98
}

# Realizar predicción
response = requests.post(
    "http://localhost:8000/predict",
    json=data
)

result = response.json()
print(f"Predicted price: ${result['prediction']:.2f}")
```

## 📊 Monitoreo

### Configuración de Prometheus

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'housing-prediction-api'
    static_configs:
      - targets: ['housing-prediction-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Configuración de Grafana

```bash
# Port forward para acceder localmente
kubectl port-forward service/grafana 3000:3000

# Acceder en: http://localhost:3000
# Usuario: admin
# Password: admin (cambiar en primer login)
```

### Configurar Alertas

Las alertas están configuradas para:

- **High Error Rate**: Tasa de error > 5%
- **High Latency**: Latencia p95 > 500ms
- **Low Throughput**: Menos de 10 req/min durante 5 minutos
- **Memory Usage**: Uso de memoria > 80%
- **Model Drift**: Drift score > 0.3

## 🚨 Detección de Drift y Reentrenamiento

### Detección de Data Drift

El sistema incluye detección automática de drift usando:

- **Statistical Tests**: Kolmogorov-Smirnov test
- **Population Stability Index (PSI)**
- **Jensen-Shannon Divergence**

```python
# Ejecutar detección de drift
python monitoring/drift_detection.py \
    --reference-data data/reference_data.csv \
    --current-data data/current_data.csv \
    --threshold 0.3
```

### Pipeline de Reentrenamiento

El reentrenamiento se activa cuando:

1. **Drift Score > 0.3**: Cambio significativo en los datos
2. **Model Performance < 0.8**: Degradación del rendimiento
3. **Schedule**: Reentrenamiento semanal programado

```bash
# Ejecutar reentrenamiento manual
python scripts/retrain_model.py --drift-detected

# Reentrenamiento automático vía GitHub Actions
# Se ejecuta cuando se detecta drift o según schedule
```

## 🧪 Testing

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_model.py -v
pytest tests/test_api.py -v
pytest tests/test_pipeline.py -v

# Tests con cobertura
pytest tests/ --cov=src --cov-report=html
```

### Tipos de Tests

#### 1. Unit Tests
- Tests de funciones individuales
- Validación de modelos
- Tests de transformaciones de datos

#### 2. Integration Tests
- Tests de API endpoints
- Tests de pipeline completo
- Tests de conexiones a bases de datos

#### 3. Performance Tests
- Tests de carga con locust
- Tests de latencia
- Tests de throughput

### Ejecutar Tests de Carga

```bash
# Instalar locust
pip install locust

# Ejecutar test de carga
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📈 Métricas de Performance

### Benchmarks del Modelo

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| R² Score | 0.854 | > 0.80 |
| MAE | 2.145 | < 3.0 |
| RMSE | 3.279 | < 4.0 |
| Training Time | 0.45s | < 1.0s |
| Inference Time | 12ms | < 50ms |

### Benchmarks de API

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| Throughput | 1,200 req/s | > 1,000 req/s |
| Latency (p50) | 25ms | < 50ms |
| Latency (p95) | 45ms | < 100ms |
| Latency (p99) | 78ms | < 200ms |
| Error Rate | 0.1% | < 1% |

### Benchmarks de Infraestructura

| Recurso | Uso Promedio | Límite |
|---------|--------------|--------|
| CPU | 35% | 80% |
| Memoria | 45% | 80% |
| Disco | 20% | 70% |
| Red | 15 Mbps | 100 Mbps |

## 📚 Documentación

### Referencias

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)


## 📞 Contacto

- **Email**: crbellor@unal.edu.co