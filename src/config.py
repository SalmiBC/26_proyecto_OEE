"""
Configuración global del Sistema Híbrido de Predicción de OEE,
Estimación de Riesgo de Plan y Diagnóstico de Fallas (LSTM + LLM).
Grupo PiSA / Laboratorios PiSA S.A. de C.V.
"""

import os
from pathlib import Path

# Directorios base (configurables vía variables de entorno para despliegues en AWS / Cloud)
BASE_DIR = Path(os.environ.get("APP_BASE_DIR", os.getcwd()))
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.environ.get("APP_MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.environ.get("APP_REPORTS_DIR", BASE_DIR / "reports"))

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Archivos de entrada principales (con fallback a copias temporales si están bloqueados por Excel)
FILE_OEE = BASE_DIR / "OEE 2024-2026.xlsx"
FILE_AVISOS = BASE_DIR / "avisos.xlsx"
FILE_PLAN = BASE_DIR / "Plan_nov.xlsx"
FILE_VELOCIDAD = BASE_DIR / "Velocidad y OEE meta.xlsx"

FILE_OEE_TEMP = BASE_DIR / "temp_oee.xlsx"
FILE_AVISOS_TEMP = BASE_DIR / "temp_avisos.xlsx"
FILE_PLAN_TEMP = BASE_DIR / "temp_plan.xlsx"
FILE_VELOCIDAD_TEMP = BASE_DIR / "temp_velocidad.xlsx"

# Archivos procesados intermedios
FILE_PROCESSED_AVISOS = DATA_DIR / "processed_avisos.parquet"
FILE_DAILY_MAINTENANCE = DATA_DIR / "daily_maintenance_features.parquet"
FILE_MASTER_OEE = DATA_DIR / "master_oee_dataset.parquet"
FILE_VELOCIDAD_PARQUET = DATA_DIR / "velocidad_oee_meta.parquet"
FILE_RISK_RESULTS = DATA_DIR / "risk_assessment_results.json"
FILE_PRESCRIPTIVE = DATA_DIR / "prescriptive_diagnostics.json"
FILE_TEST_METRICS = DATA_DIR / "test_evaluation_metrics.json"
FILE_TEST_COMPARISON = DATA_DIR / "test_comparison_2026.parquet"

# Partición Temporal Oficial (Train / Test split)
TRAIN_END_DATE = "2026-01-31"      # Datos de entrenamiento: 2024 a Enero 2026
TEST_START_DATE = "2026-02-01"     # Datos de prueba / comparativa: Febrero a Septiembre 2026
TEST_END_DATE = "2026-09-30"

# Dashboard de salida
FILE_DASHBOARD_HTML = BASE_DIR / "dashboard_oee_predictivo.html"

# Mapeo y Normalización de Líneas entre OEE, Plan_nov y Avisos
# Las 19 líneas oficiales de Plan_nov son el objetivo principal
LINE_MAPPING = {
    "A4": "A4",
    "A5": "A5",
    "ANES": "ANES",
    "AV1": "AV1",
    "AV2": "AV2",
    "AV3": "AV3",
    "AV6": "AV6",
    "AV7": "AV7",
    "BP": "BP",
    "HM": "HM",
    "JERINGAS": "JERINGAS",
    "Jeringas": "JERINGAS",
    "jeringas": "JERINGAS",
    "OFT": "OFT",
    "OFTALMICOS": "OFT",
    "OFTÁLMICOS": "OFT",
    "POLVOS": "POLVOS",
    "SOV": "SOV",
    "SV1": "SV1",
    "VA1": "VA1",
    "VA2": "VA2",
    "VA3": "VA3",
    "VA4": "VA4",
    # Variantes comunes en datos históricos
    "LINEA 1": "LINEA 1",
    "LINEA 2": "LINEA 2",
    "LINEA 4": "LINEA 4",
    "CALCIO": "CALCIO"
}

# Columnas de Paros No Planeados en OEE 2024-2026 (Cols O a U)
UNPLANNED_STOPPAGE_COLS = [
    'Ineficiencia',
    'Cambios y limpiezas',
    'Exceso Cambios y limpiezas',
    'Falta de material',
    'Falta de personal',
    'Falta de servicios',
    'Comedor'
]

# Columnas de Paros Planeados en OEE 2024-2026 (Cols V a AK)
PLANNED_STOPPAGE_COLS = [
    'SL por falta de material',
    'Contingencia',
    'Da festivo',
    'Sin turno',
    'Sin Programa',
    'Esperas',
    'Mtto Preventivo',
    'Exceso Mtto Preventivo',
    'Administrativo',
    'Proyectos (Paros mayores)',
    'Desarrollo',
    'Habilitacin',
    'Transferencia',
    'Validacin',
    'Llenado Simulado',
    'Cumplimiento normativo'
]

# Parámetros del Modelo LSTM
LSTM_CONFIG = {
    "sequence_length": 14,      # Ventana temporal en días
    "forecast_horizon": 30,     # Días proyectados (Noviembre)
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.003,
    "epochs": 45,
    "batch_size": 32,
    "mc_samples": 200,          # Muestras Monte Carlo Dropout para inferencia Bayesiana
    "random_seed": 42
}

# Umbrales de Semáforo de Riesgo (según PRD)
RISK_THRESHOLDS = {
    "verde_max": 0.20,          # P < 20%
    "amarillo_max": 0.60,       # 20% <= P <= 60%
    "rojo_min": 0.60            # P > 60%
}
