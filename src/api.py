"""
API REST para el Sistema Híbrido de Predicción de OEE y Evaluación de Riesgo de Fabricación.
Laboratorios PiSA S.A. de C.V. - Planeación de la Producción.
Cumple con el Criterio 3 de Publicación del Modelo (FastAPI, Swagger/OpenAPI, Pydantic)
y el Criterio 4 de Integración de Aplicaciones con Manejo de Errores y Validaciones.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import norm
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Asegurar path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FILE_MASTER_OEE, MODELS_DIR, RISK_THRESHOLDS,
    UNPLANNED_STOPPAGE_COLS, LINE_MAPPING
)
from src.data_pipeline import get_line_specs_dict
from src.risk_engine import calculate_line_saturation
from src.lstm_model import (
    MultiOutputMCDropoutLSTM, predict_line_probabilistic,
    FEATURE_COLS, TARGET_STOPPAGE_COLS
)

# Estado global para modelos y metadatos en memoria
ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos, scaler y especificaciones de líneas al arrancar el servicio."""
    model_path = MODELS_DIR / "best_lstm_model.pth"
    scaler_path = MODELS_DIR / "scaler.joblib"
    
    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(f"Modelos no encontrados en {MODELS_DIR}. Entrena el modelo previamente.")
    
    # Cargar checkpoint PyTorch
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    config = checkpoint['config']
    
    model = MultiOutputMCDropoutLSTM(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_p=config['dropout'],
        stoppage_dim=len(checkpoint['stoppage_cols'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = joblib.load(scaler_path)
    df_master = pd.read_parquet(FILE_MASTER_OEE)
    line_specs = get_line_specs_dict()
    
    ml_assets["model"] = model
    ml_assets["scaler"] = scaler
    ml_assets["df_master"] = df_master
    ml_assets["line_specs"] = line_specs
    ml_assets["available_lines"] = sorted(list(line_specs.keys()))
    
    print(f"[API] Modelo LSTM, Scaler y {len(ml_assets['available_lines'])} Líneas cargados exitosamente.")
    yield
    ml_assets.clear()

app = FastAPI(
    title="API de Predicción de OEE, Saturación y Riesgo Operativo",
    description="Servicio REST para inferencia estocástica de OEE (LSTM + Monte Carlo Dropout) y evaluación de saturación e incumplimiento de plan de fabricación en Laboratorios PiSA.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Habilitar CORS para clientes web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Esquemas Pydantic con Validación de Entradas -----------------

VALID_LINES = list(LINE_MAPPING.keys())

class LineSpecItem(BaseModel):
    linea: str
    velocidad_nominal: float = Field(description="Velocidad nominal en piezas (PVOs) por hora")
    oee_meta_pct: float = Field(description="OEE Meta estándar de ingeniería en %")

class PredictionRequest(BaseModel):
    linea: str = Field(
        ...,
        description="Código oficial de la línea de producción (ej. 'A4', 'ANES', 'AV1', 'AV3', 'BP')",
        examples=["AV3"]
    )
    mc_samples: int = Field(
        default=200,
        ge=20,
        le=500,
        description="Número de simulaciones estocásticas Monte Carlo Dropout (entre 20 y 500)"
    )

    @field_validator("linea")
    @classmethod
    def validate_linea(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in LINE_MAPPING:
            raise ValueError(f"Línea '{v}' no reconocida. Opciones válidas: {sorted(list(set(LINE_MAPPING.values())))}")
        return LINE_MAPPING[clean]

class RiskEvaluationRequest(BaseModel):
    linea: str = Field(..., description="Línea de producción", examples=["AV3"])
    oee_meta_pct: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=100.0,
        description="OEE Meta programado para el mes en porcentaje (ej. 65.0). Si se omite, se usa el estándar de ingeniería.",
        examples=[65.0]
    )
    horas_programadas: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=744.0,
        description="Horas disponibles/programadas de operación (máximo 744.0 h para 31 días naturales del mes)",
        examples=[600.0]
    )
    horas_disponibles: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=744.0,
        description="Alias de horas_programadas (máximo 744.0 h para 31 días naturales del mes)",
        examples=[600.0]
    )
    volumen_programado_piezas: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Volumen programado de piezas de producto terminado",
        examples=[50000.0]
    )
    pvos_a_fabricar: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Alias de volumen_programado_piezas",
        examples=[50000.0]
    )
    mc_samples: int = Field(
        default=200,
        ge=20,
        le=500,
        description="Número de muestras de Monte Carlo Dropout"
    )

    @field_validator("linea")
    @classmethod
    def validate_linea(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in LINE_MAPPING:
            raise ValueError(f"Línea '{v}' no reconocida. Opciones válidas: {sorted(list(set(LINE_MAPPING.values())))}")
        return LINE_MAPPING[clean]

class PredictionResponse(BaseModel):
    linea: str
    mu_oee_pct: float = Field(description="Media esperada del OEE en porcentaje")
    sigma_oee_pct: float = Field(description="Desviación estándar de la incertidumbre en %")
    q10_oee_pct: float = Field(description="Cuantil 10 (Escenario pesimista)")
    q50_oee_pct: float = Field(description="Cuantil 50 (Mediana)")
    q90_oee_pct: float = Field(description="Cuantil 90 (Escenario optimista)")
    desglose_paros_horas: Dict[str, float] = Field(description="Horas estimadas de paro por categoría no planeada")

class RiskResponse(BaseModel):
    linea: str
    velocidad_nominal: float = Field(description="Velocidad nominal en piezas/hora")
    horas_disponibles: float = Field(description="Horas netas disponibles para fabricar")
    pvos_a_fabricar: float = Field(description="Volumen total de piezas (PVOs) a fabricar")
    oee_meta_pct: float = Field(description="OEE Meta comprometido en %")
    oee_proyectado_mu_pct: float = Field(description="OEE proyectado por red LSTM (Media μ) en %")
    oee_proyectado_sigma_pct: float = Field(description="Desviación estándar de incertidumbre en %")
    q10_oee_pct: float = Field(description="Cuantil P10 de OEE (Escenario Pesimista)")
    q50_oee_pct: float = Field(description="Cuantil P50 de OEE (Mediana)")
    q90_oee_pct: float = Field(description="Cuantil P90 de OEE (Escenario Optimista)")
    
    # Horas y Saturación
    horas_requeridas_teoricas: float = Field(description="Horas requeridas teóricas con OEE Meta")
    saturacion_teorica_pct: float = Field(description="Saturación teórica de la línea con OEE Meta en %")
    horas_requeridas_proyectadas: float = Field(description="Horas requeridas proyectadas con OEE esperado LSTM")
    saturacion_proyectada_pct: float = Field(description="Saturación proyectada con OEE esperado en %")
    saturacion_q10_pct: float = Field(description="Saturación Cuantil P10 en %")
    saturacion_q50_pct: float = Field(description="Saturación Cuantil P50 en %")
    saturacion_q90_pct: float = Field(description="Saturación Cuantil P90 en %")
    
    # Decisión Oficial de Incumplimiento
    incumple_plan: bool = Field(description="True si Saturación Proyectada > 100% (Incumplimiento), False si cumple")
    pct_incumplimiento_piezas: float = Field(description="% de piezas del plan que no podrán fabricarse")
    pct_sobresaturacion_capacidad: float = Field(description="% de sobre-saturación de capacidad por encima del 100%")
    piezas_faltantes: float = Field(description="Cantidad de piezas estimadas que no se podrán entregar")
    horas_faltantes: float = Field(description="Horas adicionales de máquina requeridas para cumplir el plan")
    probabilidad_saturacion_pct: float = Field(description="Probabilidad Monte Carlo P(Saturación > 100%)")
    
    # Riesgo y Paros
    probabilidad_incumplimiento_pct: float = Field(description="Probabilidad de que OEE real sea inferior al OEE Meta")
    nivel_riesgo: str = Field(description="'Estable' (Verde), 'Alerta' (Amarillo) o 'Crítico' (Rojo)")
    semaforo: str = Field(description="'green', 'yellow' o 'red'")
    horas_en_riesgo: float
    volumen_piezas_en_riesgo: float
    detalle_paros_no_planeados: Dict[str, float] = Field(description="Horas estimadas de paro por día típico de 24h")
    detalle_paros_periodo: Optional[Dict[str, float]] = Field(default=None, description="Horas estimadas de paro acumuladas en el total de horas programadas")

# ----------------- Endpoints REST y Vistas Web -----------------

@app.get("/", tags=["UI"], include_in_schema=False)
def get_root_ui():
    """Sirve la aplicación de consulta web directamente en la raíz para despliegues en AWS / Cloud."""
    base_dir = Path(__file__).resolve().parent.parent
    html_path = base_dir / "app_consulta.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return {"service": "OEE & Risk Prediction API", "status": "running", "docs": "/docs"}

@app.get("/app", tags=["UI"], include_in_schema=False)
def get_app_ui():
    """Ruta explícita para la aplicación de consulta interactiva."""
    base_dir = Path(__file__).resolve().parent.parent
    html_path = base_dir / "app_consulta.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="app_consulta.html no encontrado")

@app.get("/presentacion", tags=["UI"], include_in_schema=False)
def get_presentation_ui():
    """Sirve la presentación ejecutiva e interactiva de la rúbrica MLOps."""
    base_dir = Path(__file__).resolve().parent.parent
    html_path = base_dir / "presentacion_rubrica_oee.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="presentacion_rubrica_oee.html no encontrado")

@app.get("/health", tags=["Monitoreo"])
def health_check():
    """Verifica que el servicio esté operativo y los modelos cargados."""
    is_ready = "model" in ml_assets and "scaler" in ml_assets
    return {
        "status": "healthy" if is_ready else "initializing",
        "service": "OEE & Risk Prediction API",
        "framework": "FastAPI + PyTorch",
        "models_loaded": is_ready,
        "available_lines_count": len(ml_assets.get("available_lines", []))
    }

@app.get("/lines", response_model=List[str], tags=["Metadatos"])
def list_available_lines():
    """Retorna la lista de líneas de producción con soporte de inferencia."""
    return ml_assets.get("available_lines", [])

@app.get("/line-specs", response_model=Dict[str, LineSpecItem], tags=["Metadatos"])
def get_all_line_specs():
    """Retorna las velocidades nominales (pvos/h) y OEE meta de ingeniería para cada línea."""
    specs = ml_assets.get("line_specs", {})
    res = {}
    for line, s in specs.items():
        res[line] = LineSpecItem(
            linea=line,
            velocidad_nominal=round(s.get("velocidad_nominal", 0.0), 1),
            oee_meta_pct=round(s.get("oee_meta", 0.60) * 100.0, 2)
        )
    return res

@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
def predict_oee(request: PredictionRequest):
    """
    Genera la predicción probabilística de OEE para una línea de producción
    utilizando la red LSTM y simulación Monte Carlo Dropout.
    """
    linea = request.linea
    model = ml_assets.get("model")
    scaler = ml_assets.get("scaler")
    df_master = ml_assets.get("df_master")
    
    if model is None or scaler is None or df_master is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos aún no se han inicializado."
        )
    
    try:
        pred = predict_line_probabilistic(
            line=linea,
            model=model,
            scaler=scaler,
            df_master=df_master,
            n_samples=request.mc_samples
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia estocástica de la línea {linea}: {str(e)}"
        )
        
    return PredictionResponse(
        linea=linea,
        mu_oee_pct=round(pred["mu_oee"] * 100, 2),
        sigma_oee_pct=round(pred["sigma_oee"] * 100, 2),
        q10_oee_pct=round(pred["q10_oee"] * 100, 2),
        q50_oee_pct=round(pred["q50_oee"] * 100, 2),
        q90_oee_pct=round(pred["q90_oee"] * 100, 2),
        desglose_paros_horas={k: round(v, 2) for k, v in pred["stoppage_breakdown"].items()}
    )

@app.post("/evaluate-risk", response_model=RiskResponse, tags=["Inferencia"])
def evaluate_risk(request: RiskEvaluationRequest):
    """
    Evalúa el riesgo de incumplimiento del plan de fabricación y la saturación de línea
    confrontando la predicción del modelo LSTM contra la meta comprometida.
    """
    linea = request.linea
    model = ml_assets.get("model")
    scaler = ml_assets.get("scaler")
    df_master = ml_assets.get("df_master")
    line_specs = ml_assets.get("line_specs", {})
    
    if model is None or scaler is None or df_master is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos aún no se han inicializado."
        )
        
    specs = line_specs.get(linea, {'velocidad_nominal': 5000.0, 'oee_meta': 0.60})
    vel_nominal = float(specs.get('velocidad_nominal', 5000.0))
    
    # Determinar OEE Meta, Horas Disponibles y PVOs
    default_meta_pct = round(float(specs.get('oee_meta', 0.60)) * 100.0, 2)
    oee_meta_pct = request.oee_meta_pct if request.oee_meta_pct is not None else default_meta_pct
    
    horas_disp = request.horas_disponibles or request.horas_programadas or 480.0
    pvos_val = request.pvos_a_fabricar or request.volumen_programado_piezas or 1000000.0
    
    # 1. Inferencia estocástica con Monte Carlo Dropout
    try:
        pred = predict_line_probabilistic(
            line=linea,
            model=model,
            scaler=scaler,
            df_master=df_master,
            n_samples=request.mc_samples
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al evaluar modelo estocástico para línea {linea}: {str(e)}"
        )
        
    mu_oee = pred['mu_oee']
    sigma_oee = pred['sigma_oee']
    samples = pred['oee_samples']
    
    # 2. Cálculo integral de saturación de línea
    sat = calculate_line_saturation(
        line=linea,
        pvos=pvos_val,
        horas_disponibles=horas_disp,
        velocidad_nominal=vel_nominal,
        oee_meta=oee_meta_pct,
        mu_oee=mu_oee,
        sigma_oee=sigma_oee,
        oee_samples=samples
    )
    
    # 3. Probabilidad de que OEE real < OEE Meta
    oee_meta_ratio = oee_meta_pct / 100.0
    emp_p = sum(1 for s in samples if s < oee_meta_ratio) / len(samples)
    if sigma_oee > 1e-4:
        z_score = (oee_meta_ratio - mu_oee) / sigma_oee
        norm_p = float(norm.cdf(z_score))
    else:
        norm_p = 1.0 if oee_meta_ratio > mu_oee else 0.0
    p_incumplimiento = round(0.5 * emp_p + 0.5 * norm_p, 4)
    
    # Horas y volumen en riesgo
    horas_en_riesgo = sat['horas_faltantes'] if sat['incumple_plan'] else max(0.0, (oee_meta_ratio - mu_oee) * horas_disp)
    volumen_en_riesgo = sat['piezas_faltantes'] if sat['incumple_plan'] else (
        max(0.0, (1.0 - (mu_oee / max(1e-4, oee_meta_ratio))) * pvos_val) if oee_meta_ratio > 0 else 0.0
    )
    
    return RiskResponse(
        linea=linea,
        velocidad_nominal=round(vel_nominal, 1),
        horas_disponibles=round(horas_disp, 1),
        pvos_a_fabricar=round(pvos_val, 0),
        oee_meta_pct=round(oee_meta_pct, 2),
        oee_proyectado_mu_pct=round(mu_oee * 100, 2),
        oee_proyectado_sigma_pct=round(sigma_oee * 100, 2),
        q10_oee_pct=round(pred['q10_oee'] * 100, 2),
        q50_oee_pct=round(pred['q50_oee'] * 100, 2),
        q90_oee_pct=round(pred['q90_oee'] * 100, 2),
        horas_requeridas_teoricas=sat['horas_requeridas_teoricas'],
        saturacion_teorica_pct=sat['saturacion_teorica_pct'],
        horas_requeridas_proyectadas=sat['horas_requeridas_proyectadas'],
        saturacion_proyectada_pct=sat['saturacion_proyectada_pct'],
        saturacion_q10_pct=sat['saturacion_q10_pct'],
        saturacion_q50_pct=sat['saturacion_q50_pct'],
        saturacion_q90_pct=sat['saturacion_q90_pct'],
        incumple_plan=sat['incumple_plan'],
        pct_incumplimiento_piezas=sat['pct_incumplimiento_piezas'],
        pct_sobresaturacion_capacidad=sat['pct_sobresaturacion_capacidad'],
        piezas_faltantes=sat['piezas_faltantes'],
        horas_faltantes=sat['horas_faltantes'],
        probabilidad_saturacion_pct=sat['probabilidad_saturacion_pct'],
        probabilidad_incumplimiento_pct=round(p_incumplimiento * 100, 1),
        nivel_riesgo=sat['nivel_riesgo'],
        semaforo=sat['semaforo'],
        horas_en_riesgo=round(horas_en_riesgo, 1),
        volumen_piezas_en_riesgo=round(volumen_en_riesgo, 0),
        detalle_paros_no_planeados={k: round(v, 2) for k, v in pred['stoppage_breakdown'].items()},
        detalle_paros_periodo={k: round(v * (horas_disp / 24.0), 1) for k, v in pred['stoppage_breakdown'].items()}
    )

@app.post("/evaluate-saturation", response_model=RiskResponse, tags=["Inferencia"])
def evaluate_saturation(request: RiskEvaluationRequest):
    """Alias específico para evaluar la saturación y riesgo de incumplimiento de plan."""
    return evaluate_risk(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
