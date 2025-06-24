#!/bin/bash
# scripts/setup-monitoring.sh - Configurar monitoreo completo

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log "=== Configurando Monitoreo Completo ==="

# Crear namespace de monitoreo
log "Creando namespace de monitoreo..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Aplicar configuraciones de monitoreo
log "Aplicando configuraciones de Prometheus y Grafana..."
kubectl apply -f monitoring/

# Esperar a que Prometheus esté listo
log "Esperando a que Prometheus esté listo..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n monitoring

# Esperar a que Grafana esté listo
log "Esperando a que Grafana esté listo..."
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n monitoring

# Obtener IPs de servicios
log "Obteniendo información de servicios..."
PROMETHEUS_IP=$(kubectl get service prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
GRAFANA_IP=$(kubectl get service grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

log "Configuración de monitoreo completada!"
log "Prometheus: http://$PROMETHEUS_IP:9090"
log "Grafana: http://$GRAFANA_IP:3000 (admin/admin123)"

# Crear dashboard de Grafana
log "Configurando dashboard de Grafana..."
cat > /tmp/housing-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Housing Price API Dashboard",
    "tags": ["ml", "housing", "api"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, http_request_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "title": "Prediction Drift",
        "type": "graph",
        "targets": [
          {
            "expr": "prediction_drift_score",
            "legendFormat": "Drift Score"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "30s"
  }
}
EOF

log "Dashboard JSON creado en /tmp/housing-dashboard.json"
log "Importa este dashboard manualmente en Grafana"

# Crear script de test de carga
log "Creando script de test de carga..."
cat > scripts/load-test.sh << 'EOF'
#!/bin/bash
# Script de test de carga para la API

API_URL=${1:-"http://localhost:8000"}
CONCURRENT_USERS=${2:-10}
DURATION=${3:-60}

echo "Ejecutando test de carga..."
echo "URL: $API_URL"
echo "Usuarios concurrentes: $CONCURRENT_USERS"
echo "Duración: $DURATION segundos"

# Instalar wrk si no está disponible
if ! command -v wrk &> /dev/null; then
    echo "wrk no está instalado. Instalando..."
    sudo apt-get update && sudo apt-get install -y wrk
fi

# Test de carga con datos de ejemplo
wrk -t$CONCURRENT_USERS -c$CONCURRENT_USERS -d${DURATION}s \
    -s scripts/wrk-script.lua \
    $API_URL/predict

echo "Test de carga completado"
EOF

# Crear script Lua para wrk
cat > scripts/wrk-script.lua << 'EOF'
-- wrk script para test de carga
wrk.method = "POST"
wrk.body = '{"crim": 0.1, "zn": 0.0, "indus": 10.0, "chas": 0, "nox": 0.5, "rm": 6.0, "age": 50.0, "dis": 4.0, "rad": 5, "tax": 300.0, "ptratio": 15.0, "b": 350.0, "lstat": 10.0}'
wrk.headers["Content-Type"] = "application/json"

function response(status, headers, body)
    if status ~= 200 then
        print("Error response: " .. status)
    end
end
EOF

chmod +x scripts/load-test.sh
chmod +x scripts/wrk-script.lua

log "Script de test de carga creado: scripts/load-test.sh"

# Crear script de monitoreo de salud
cat > scripts/health-monitor.sh << 'EOF'
#!/bin/bash
# Monitor de salud de la aplicación

API_URL=${1:-"http://localhost:8000"}
CHECK_INTERVAL=${2:-30}

echo "Iniciando monitor de salud..."
echo "URL: $API_URL"
echo "Intervalo: $CHECK_INTERVAL segundos"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Health check
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
    
    # Ready check
    READY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/ready)
    
    # Metrics check
    METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/metrics)
    
    if [ "$HEALTH_STATUS" = "200" ] && [ "$READY_STATUS" = "200" ]; then
        echo "[$TIMESTAMP] ✅ Servicio saludable (Health: $HEALTH_STATUS, Ready: $READY_STATUS, Metrics: $METRICS_STATUS)"
    else
        echo "[$TIMESTAMP] ❌ Servicio con problemas (Health: $HEALTH_STATUS, Ready: $READY_STATUS, Metrics: $METRICS_STATUS)"
        
        # Opcional: enviar alerta
        # curl -X POST "https://hooks.slack.com/services/YOUR/WEBHOOK/URL" \
        #      -H 'Content-type: application/json' \
        #      --data '{"text":"🚨 API Health Check Failed!"}'
    fi
    
    sleep $CHECK_INTERVAL
done
EOF

chmod +x scripts/health-monitor.sh

log "Script de monitoreo de salud creado: scripts/health-monitor.sh"

# Crear script de backup del modelo
cat > scripts/backup-model.sh << 'EOF'
#!/bin/bash
# Script de backup del modelo

set -e

BACKUP_DIR=${1:-"backups"}
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MODEL_DIR="models"

echo "Creando backup del modelo..."

# Crear directorio de backup
mkdir -p $BACKUP_DIR

# Crear archivo de backup
BACKUP_FILE="$BACKUP_DIR/model_backup_$TIMESTAMP.tar.gz"

tar -czf $BACKUP_FILE -C $MODEL_DIR .

echo "Backup creado: $BACKUP_FILE"

# Limpiar backups antiguos (mantener últimos 10)
cd $BACKUP_DIR
ls -t model_backup_*.tar.gz | tail -n +11 | xargs -r rm

echo "Backup completado y limpieza realizada"
EOF

chmod +x scripts/backup-model.sh

log "Script de backup creado: scripts/backup-model.sh"

log "¡Configuración de monitoreo completada exitosamente!"
log ""
log "Comandos útiles:"
log "- Test de carga: ./scripts/load-test.sh http://your-api-url"
log "- Monitor de salud: ./scripts/health-monitor.sh http://your-api-url"
log "- Backup del modelo: ./scripts/backup-model.sh"
log ""
log "Acceso a servicios:"
log "- Prometheus: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
log "- Grafana: kubectl port-forward -n monitoring svc/grafana 3000:3000"