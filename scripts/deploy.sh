# scripts/deploy.sh - Script principal de despliegue

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

RESOURCE_GROUP="ml-housing-rg"
LOCATION="eastus"
AKS_CLUSTER_NAME="ml-housing-aks"
ACR_NAME="housingpriceregistry"
APP_NAME="housing-price-api"

echo -e "${GREEN}=== ML Housing Price Deployment Script ===${NC}"

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Azure CLI
if ! command -v az &> /dev/null; then
    error "Azure CLI no está instalado. Instálación: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
fi

# Kubectl
if ! command -v kubectl &> /dev/null; then
    error "kubectl no está instalado. Instálación: https://kubernetes.io/docs/tasks/tools/"
fi

# Docker
if ! command -v docker &> /dev/null; then
    error "Docker no está instalado. Instálación: https://docs.docker.com/get-docker/"
fi

# Login Azure
log "Verificando login de Azure..."
if ! az account show &> /dev/null; then
    log "Haciendo login a Azure..."
    az login
fi

log "Creando grupo de recursos..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Azure Container Registry
log "Creando Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic

# Admin user para ACR
log "Habilitando admin user para ACR..."
az acr update -n $ACR_NAME --admin-enabled true

# AKS cluster
log "Creando AKS cluster (esto puede tomar varios minutos)..."
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_CLUSTER_NAME \
    --node-count 3 \
    --node-vm-size Standard_B2s \
    --enable-addons monitoring \
    --attach-acr $ACR_NAME \
    --generate-ssh-keys

# AKS
log "Obteniendo credenciales de AKS..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME

# Construir y push de la imagen Docker
log "Construyendo imagen Docker..."
docker build -t $APP_NAME:latest .

# Hacer login a ACR
log "Haciendo login a ACR..."
az acr login --name $ACR_NAME

# Tag y push de la imagen
log "Etiquetando y subiendo imagen a ACR..."
docker tag $APP_NAME:latest $ACR_NAME.azurecr.io/$APP_NAME:latest
docker push $ACR_NAME.azurecr.io/$APP_NAME:latest

# Crear secret para ACR
log "Creando secret para ACR..."
kubectl create secret docker-registry acr-secret \
    --docker-server=$ACR_NAME.azurecr.io \
    --docker-username=$ACR_NAME \
    --docker-password=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv) \
    --namespace=ml-housing-app || true

# Kubernetes
log "Aplicando manifiestos de Kubernetes..."
kubectl apply -f k8s/

log "Esperando a que los pods estén listos..."
kubectl wait --for=condition=available --timeout=300s deployment/housing-price-api -n ml-housing-app

# Mostrar información del despliegue
log "Obteniendo información del despliegue..."
kubectl get pods -n ml-housing-app
kubectl get services -n ml-housing-app
kubectl get ingress -n ml-housing-app

# Obtener IP externa del servicio
log "Obteniendo IP del servicio..."
SERVICE_IP=$(kubectl get service housing-price-service -n ml-housing-app -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$SERVICE_IP" ]; then
    warn "El servicio aún no tiene IP externa asignada. Esto puede tomar unos minutos."
    log "Ejecuta 'kubectl get service housing-price-service -n ml-housing-app' para verificar el estado."
else
    log "Servicio disponible en: http://$SERVICE_IP"
fi

log "¡Despliegue completado exitosamente!"
log "Para probar la API, ejecuta: curl http://$SERVICE_IP/health"