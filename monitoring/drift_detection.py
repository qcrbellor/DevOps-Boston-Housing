# drift_detection/drift_monitor.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDrifts

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detector de drift para modelos ML"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_data = None
        self.column_mapping = None
        
    def set_reference_data(self, data: pd.DataFrame, target_column: str = None):
        """Establecer datos de referencia"""
        self.reference_data = data
        self.column_mapping = ColumnMapping()
        if target_column:
            self.column_mapping.target = target_column
            
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detectar drift en los datos"""
        if self.reference_data is None:
            raise ValueError("Datos de referencia no establecidos")
            
        # Usar Evidently para detección de drift
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extraer resultados
        report_dict = data_drift_report.as_dict()
        drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
        
        return {
            'drift_detected': drift_detected,
            'drift_score': report_dict['metrics'][0]['result']['drift_share'],
            'drifted_features': [
                feature for feature, result in 
                report_dict['metrics'][0]['result']['drift_by_columns'].items()
                if result['drift_detected']
            ],
            'report': report_dict,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_prediction_drift(self, 
                              reference_predictions: np.ndarray,
                              current_predictions: np.ndarray) -> Dict:
        """Detectar drift en predicciones"""
        
        # Test de Kolmogorov-Smirnov
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)
        
        # Test de Mann-Whitney U
        mw_statistic, mw_pvalue = stats.mannwhitneyu(
            reference_predictions, current_predictions, alternative='two-sided'
        )
        
        # Comparación de distribuciones
        ref_mean, ref_std = np.mean(reference_predictions), np.std(reference_predictions)
        cur_mean, cur_std = np.mean(current_predictions), np.std(current_predictions)
        
        # Drift score basado en diferencias normalizadas
        mean_drift = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0
        std_drift = abs(cur_std - ref_std) / ref_std if ref_std > 0 else 0
        
        drift_score = (mean_drift + std_drift) / 2
        
        return {
            'drift_detected': ks_pvalue < self.threshold or mw_pvalue < self.threshold,
            'drift_score': drift_score,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_statistic,
            'mw_pvalue': mw_pvalue,
            'reference_stats': {'mean': ref_mean, 'std': ref_std},
            'current_stats': {'mean': cur_mean, 'std': cur_std},
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_drift_report(self, 
                            current_data: pd.DataFrame,
                            current_predictions: np.ndarray = None) -> str:
        """Generar reporte completo de drift"""
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': None,
            'prediction_drift': None
        }
        
        # Detectar drift en datos
        try:
            report_data['data_drift'] = self.detect_data_drift(current_data)
        except Exception as e:
            logger.error(f"Error detectando drift en datos: {e}")
            
        # Detectar drift en predicciones si están disponibles
        if current_predictions is not None and hasattr(self, 'reference_predictions'):
            try:
                report_data['prediction_drift'] = self.detect_prediction_drift(
                    self.reference_predictions, current_predictions
                )
            except Exception as e:
                logger.error(f"Error detectando drift en predicciones: {e}")
        
        # Guardar reporte
        report_filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        return report_filename

class ContinuousMonitor:
    """Monitor continuo de drift"""
    
    def __init__(self, 
                 drift_detector: DriftDetector,
                 check_interval: int = 3600,  # 1 hora
                 alert_threshold: float = 0.1):
        self.drift_detector = drift_detector
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.monitoring_data = []
        self.last_check = datetime.now()
        
    def add_data_point(self, data: pd.DataFrame, prediction: float = None):
        """Agregar punto de datos para monitoreo"""
        self.monitoring_data.append({
            'timestamp': datetime.now(),
            'data': data,
            'prediction': prediction
        })
        
        # Mantener solo datos de las últimas 24 horas
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.monitoring_data = [
            item for item in self.monitoring_data 
            if item['timestamp'] > cutoff_time
        ]
        
    def should_check_drift(self) -> bool:
        """Verificar si es tiempo de revisar drift"""
        return (datetime.now() - self.last_check).seconds >= self.check_interval
    
    def check_and_alert(self) -> Optional[Dict]:
        """Verificar drift y generar alertas si es necesario"""
        if not self.should_check_drift() or len(self.monitoring_data) < 10:
            return None
            
        # Compilar datos para análisis
        current_data = pd.concat([item['data'] for item in self.monitoring_data])
        current_predictions = np.array([
            item['prediction'] for item in self.monitoring_data 
            if item['prediction'] is not None
        ])
        
        # Detectar drift
        drift_result = None
        if len(current_predictions) > 0:
            drift_result = self.drift_detector.detect_prediction_drift(
                self.drift_detector.reference_predictions, current_predictions
            )
            
        # Generar alerta si es necesario
        if drift_result and drift_result['drift_score'] > self.alert_threshold:
            alert = {
                'type': 'drift_alert',
                'severity': 'high' if drift_result['drift_score'] > 0.5 else 'medium',
                'message': f"Drift detectado con score: {drift_result['drift_score']:.3f}",
                'timestamp': datetime.now().isoformat(),
                'details': drift_result
            }
            
            self.send_alert(alert)
            self.last_check = datetime.now()
            return alert
            
        self.last_check = datetime.now()
        return None
    
    def send_alert(self, alert: Dict):
        """Enviar alerta (implementar según necesidades)"""
        logger.warning(f"DRIFT ALERT: {alert['message']}")
        
        # Aquí se puede implementar:
        # - Envío de emails
        # - Webhooks
        # - Notificaciones a Slack/Teams
        # - Escritura a sistemas de monitoreo
        
        # Ejemplo de guardado en archivo
        alert_filename = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"alerts/{alert_filename}", 'w') as f:
            json.dump(alert, f, indent=2)

# drift_detection/retrain_trigger.py
class RetrainTrigger:
    """Trigger para reentrenamiento automático"""
    
    def __init__(self, 
                 drift_threshold: float = 0.3,
                 performance_threshold: float = 0.1,
                 min_samples: int = 100):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.min_samples = min_samples
        self.should_retrain = False
        
    def evaluate_retrain_need(self, 
                            drift_score: float,
                            performance_degradation: float,
                            sample_count: int) -> Dict:
        """Evaluar si es necesario reentrenar"""
        
        reasons = []
        
        if drift_score > self.drift_threshold:
            reasons.append(f"Drift score ({drift_score:.3f}) excede threshold ({self.drift_threshold})")
            
        if performance_degradation > self.performance_threshold:
            reasons.append(f"Degradación de performance ({performance_degradation:.3f}) excede threshold ({self.performance_threshold})")
            
        if sample_count < self.min_samples:
            reasons.append(f"Muestras insuficientes ({sample_count}) para evaluación confiable")
            
        should_retrain = len(reasons) > 0 and sample_count >= self.min_samples
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'drift_score': drift_score,
            'performance_degradation': performance_degradation,
            'sample_count': sample_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def trigger_retrain(self, retrain_config: Dict) -> bool:
        """Disparar proceso de reentrenamiento"""
        try:
            # Aquí se implementaría la lógica para:
            # 1. Preparar datos de reentrenamiento
            # 2. Disparar pipeline de ML
            # 3. Validar nuevo modelo
            # 4. Desplegar si es mejor
            
            logger.info("Iniciando proceso de reentrenamiento...")
            
            # Ejemplo de comando para disparar pipeline
            # subprocess.run(['python', 'train_pipeline.py', '--config', json.dumps(retrain_config)])
            
            return True
            
        except Exception as e:
            logger.error(f"Error disparando reentrenamiento: {e}")
            return False

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar detector de drift
    drift_detector = DriftDetector(threshold=0.05)
    
    # Cargar datos de referencia
    reference_data = pd.read_csv('data/reference_data.csv')
    drift_detector.set_reference_data(reference_data, target_column='medv')
    
    # Configurar monitor continuo
    monitor = ContinuousMonitor(drift_detector)
    
    # Configurar trigger de reentrenamiento
    retrain_trigger = RetrainTrigger()
    
    # Simular monitoreo
    for i in range(100):
        # Simular nuevos datos
        new_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10),
            'feature2': np.random.normal(0, 1, 10)
        })
        
        monitor.add_data_point(new_data, np.random.normal(25, 5))
        
        # Verificar drift
        alert = monitor.check_and_alert()
        if alert:
            print(f"Alerta generada: {alert}")
            
            # Evaluar necesidad de reentrenamiento
            retrain_eval = retrain_trigger.evaluate_retrain_need(
                drift_score=alert['details']['drift_score'],
                performance_degradation=0.15,  # Ejemplo
                sample_count=len(monitor.monitoring_data)
            )
            
            if retrain_eval['should_retrain']:
                print("Disparando reentrenamiento...")
                retrain_trigger.trigger_retrain({'model_type': 'random_forest'})