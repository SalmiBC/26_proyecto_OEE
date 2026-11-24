"""
Módulo 3: Motor de Cuantificación de Riesgo de Fabricación.
Cruza las predicciones probabilísticas del modelo LSTM con las metas del Plan_nov,
calculando la probabilidad de incumplimiento P(OEE_real < OEE_planeado),
horas en riesgo, volumen faltante, semaforización ejecutiva y descomposición en cascada (Waterfall).
"""

import os
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import joblib

from src.config import (
    FILE_MASTER_OEE, FILE_RISK_RESULTS, MODELS_DIR, RISK_THRESHOLDS,
    UNPLANNED_STOPPAGE_COLS
)
from src.data_pipeline import load_and_aggregate_plan, get_line_specs_dict
from src.lstm_model import MultiOutputMCDropoutLSTM, predict_line_probabilistic, FEATURE_COLS, TARGET_STOPPAGE_COLS

def calculate_line_saturation(
    line: str,
    pvos: float,
    horas_disponibles: float,
    velocidad_nominal: float,
    oee_meta: float,
    mu_oee: float,
    sigma_oee: float = 0.0,
    oee_samples: list = None
) -> dict:
    """
    Calcula la saturación y riesgo de incumplimiento del plan de producción
    con base en la velocidad nominal y el OEE (teórico vs real proyectado por LSTM).
    
    Fórmula oficial:
      Horas Requeridas = PVOs / (Velocidad Nominal * OEE)
      Saturación (%)   = (Horas Requeridas / Horas Disponibles) * 100
      
    Regla de Incumplimiento:
      Si Saturación Proyectada > 100%, la línea incumple el plan de fabricación.
      
    Métricas reportadas:
      - % Piezas Faltantes (Volumen no entregable): (Piezas Faltantes / PVOs) * 100
      - % Sobre-Saturación de Capacidad: max(0, Saturación - 100%)
      - Piezas Faltantes y Horas Faltantes
      - Probabilidad Monte Carlo de saturación P(Saturación > 100%)
    """
    oee_meta_ratio = (oee_meta / 100.0) if oee_meta > 1.0 else max(0.01, float(oee_meta))
    oee_proj_ratio = (mu_oee / 100.0) if mu_oee > 1.0 else max(0.01, float(mu_oee))
    h_disp = max(0.1, float(horas_disponibles))
    vel_nom = max(1.0, float(velocidad_nominal))
    pvos_val = max(0.0, float(pvos))
    
    # 1. Horas Requeridas y Saturación Teórica (con OEE Meta)
    vel_eff_meta = vel_nom * oee_meta_ratio
    h_req_teorica = pvos_val / vel_eff_meta if vel_eff_meta > 0 else 0.0
    sat_teorica_pct = (h_req_teorica / h_disp) * 100.0
    
    # 2. Horas Requeridas y Saturación Proyectada (con OEE LSTM esperado)
    vel_eff_proj = vel_nom * oee_proj_ratio
    h_req_proj = pvos_val / vel_eff_proj if vel_eff_proj > 0 else 0.0
    sat_proj_pct = (h_req_proj / h_disp) * 100.0
    
    # Condición de incumplimiento
    incumple_plan = sat_proj_pct > 100.0
    
    # Capacidad máxima alcanzable en las horas disponibles
    piezas_posibles = h_disp * vel_eff_proj
    piezas_faltantes = max(0.0, pvos_val - piezas_posibles)
    horas_faltantes = max(0.0, h_req_proj - h_disp)
    
    # Métricas de incumplimiento (Respuesta A1 del usuario)
    pct_incumplimiento_piezas = (piezas_faltantes / pvos_val * 100.0) if pvos_val > 0 else 0.0
    pct_sobresaturacion_capacidad = max(0.0, sat_proj_pct - 100.0)
    
    # Inferencia estocástica Monte Carlo sobre la saturación
    prob_saturacion = 0.0
    q10_sat, q50_sat, q90_sat = sat_proj_pct, sat_proj_pct, sat_proj_pct
    if oee_samples and len(oee_samples) > 0 and vel_nom > 0:
        sat_samples = []
        for s in oee_samples:
            s_val = (s / 100.0) if s > 1.0 else max(0.01, float(s))
            sat_s = (pvos_val / (vel_nom * s_val * h_disp)) * 100.0
            sat_samples.append(sat_s)
        if len(sat_samples) > 0:
            prob_saturacion = (sum(1 for s in sat_samples if s > 100.0) / len(sat_samples)) * 100.0
            q10_sat = float(np.percentile(sat_samples, 10))
            q50_sat = float(np.percentile(sat_samples, 50))
            q90_sat = float(np.percentile(sat_samples, 90))
            
    # Semáforo de Riesgo
    if sat_proj_pct > 100.0 or prob_saturacion > 60.0:
        nivel_riesgo = "Crítico"
        semaforo = "red"
    elif sat_proj_pct > 90.0 or prob_saturacion > 20.0:
        nivel_riesgo = "Alerta"
        semaforo = "yellow"
    else:
        nivel_riesgo = "Estable"
        semaforo = "green"
        
    return {
        'line': line,
        'pvos_a_fabricar': pvos_val,
        'horas_disponibles': round(h_disp, 1),
        'velocidad_nominal': round(vel_nom, 1),
        'oee_meta_pct': round(oee_meta_ratio * 100, 2),
        'oee_proyectado_mu_pct': round(oee_proj_ratio * 100, 2),
        'horas_requeridas_teoricas': round(h_req_teorica, 1),
        'saturacion_teorica_pct': round(sat_teorica_pct, 1),
        'horas_requeridas_proyectadas': round(h_req_proj, 1),
        'saturacion_proyectada_pct': round(sat_proj_pct, 1),
        'saturacion_q10_pct': round(q10_sat, 1),
        'saturacion_q50_pct': round(q50_sat, 1),
        'saturacion_q90_pct': round(q90_sat, 1),
        'incumple_plan': bool(incumple_plan),
        'pct_incumplimiento_piezas': round(pct_incumplimiento_piezas, 2),
        'pct_sobresaturacion_capacidad': round(pct_sobresaturacion_capacidad, 2),
        'piezas_faltantes': round(piezas_faltantes, 0),
        'horas_faltantes': round(horas_faltantes, 1),
        'probabilidad_saturacion_pct': round(prob_saturacion, 1),
        'nivel_riesgo': nivel_riesgo,
        'semaforo': semaforo
    }

def calculate_monthly_scheduled_hours(df_master: pd.DataFrame) -> dict:
    """
    Calcula las horas programadas típicas de operación mensual por línea
    (basado en la columna AL 'Tiempo para fabricación' en meses recientes de 2025/2026).
    """
    df_recent = df_master[df_master['Fecha'] >= '2025-01-01'].copy()
    df_recent['Mes_Period'] = df_recent['Fecha'].dt.to_period('M')
    
    monthly_line_hours = df_recent.groupby(['Linea', 'Mes_Period'])['Horas_Para_Fabricar'].sum().groupby('Linea').mean()
    
    hours_dict = {}
    for line, hrs in monthly_line_hours.items():
        # Acotar a valores realistas de planta (mínimo 150h, máximo 720h)
        val = float(hrs)
        if val < 100 or np.isnan(val):
            val = 480.0  # 20 días x 24h o 2 esquemas por defecto
        hours_dict[line] = round(val, 1)
        
    return hours_dict

def run_risk_engine():
    """Ejecuta la evaluación integral de riesgo para todas las líneas de Plan_nov."""
    print("[*] Iniciando Motor de Cuantificación de Riesgo de Fabricación...")
    
    # Cargar plan maestro consolidado
    df_plan = load_and_aggregate_plan()
    
    # Cargar master dataset
    df_master = pd.read_parquet(FILE_MASTER_OEE)
    
    # Cargar modelo LSTM entrenado
    model_path = MODELS_DIR / "best_lstm_model.pth"
    scaler_path = MODELS_DIR / "scaler.joblib"
    
    checkpoint = torch.load(model_path, weights_only=False)
    config = checkpoint['config']
    
    model = MultiOutputMCDropoutLSTM(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout_p=config["dropout"],
        stoppage_dim=len(checkpoint['stoppage_cols'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = joblib.load(scaler_path)
    
    # Horas programadas mensuales por línea
    scheduled_hours = calculate_monthly_scheduled_hours(df_master)
    
    # Especificaciones de ingeniería: Velocidad Nominal y OEE Meta por línea
    line_specs = get_line_specs_dict()
    
    risk_results = []
    
    for _, row in df_plan.iterrows():
        line = row['Linea']
        oee_plan = float(row['OEE_Planeado'])
        vol_plan = float(row['Volumen_Programado'])
        skus = int(row['SKUs_Count'])
        planta = row['Planta']
        familia = row['Familia']
        
        # Horas programadas para noviembre
        h_prog = scheduled_hours.get(line, 500.0)
        
        # Especificaciones nominales
        specs = line_specs.get(line, {'velocidad_nominal': 5000.0, 'oee_meta': oee_plan})
        vel_nominal = specs['velocidad_nominal']
        oee_meta_nominal = specs['oee_meta']
        
        # Inferencia probabilística con Monte Carlo Dropout (N=200)
        pred = predict_line_probabilistic(line, model, scaler, df_master, n_samples=config.get("mc_samples", 200))
        
        mu_oee = pred['mu_oee']
        sigma_oee = pred['sigma_oee']
        q10_oee = pred['q10_oee']
        q50_oee = pred['q50_oee']
        q90_oee = pred['q90_oee']
        min_oee = pred['min_oee']
        max_oee = pred['max_oee']
        samples = pred['oee_samples']
        
        # Cálculo de Saturación y Riesgo de Incumplimiento de Plan
        sat_data = calculate_line_saturation(
            line=line,
            pvos=vol_plan,
            horas_disponibles=h_prog,
            velocidad_nominal=vel_nominal,
            oee_meta=oee_meta_nominal,
            mu_oee=mu_oee,
            sigma_oee=sigma_oee,
            oee_samples=samples
        )
        
        # 1. Brecha de OEE (ΔOEE)
        delta_oee = round(oee_plan - mu_oee, 4)
        
        # 2. Probabilidad de Incumplimiento: P(OEE_real < OEE_planeado)
        emp_p = sum(1 for s in samples if s < oee_plan) / len(samples)
        if sigma_oee > 1e-4:
            z_score = (oee_plan - mu_oee) / sigma_oee
            norm_p = float(norm.cdf(z_score))
        else:
            norm_p = 1.0 if oee_plan > mu_oee else 0.0
            
        p_incumplimiento = round(0.5 * emp_p + 0.5 * norm_p, 4)
        
        # 3. Horas y Volumen en Riesgo (alineados a saturación e históricos)
        horas_en_riesgo = sat_data['horas_faltantes'] if sat_data['incumple_plan'] else round(max(0.0, delta_oee * h_prog), 1)
        volumen_en_riesgo = sat_data['piezas_faltantes'] if sat_data['incumple_plan'] else (
            round(max(0.0, (1.0 - (mu_oee / max(1e-4, oee_plan))) * vol_plan), 0) if oee_plan > 0 else 0.0
        )
        
        # 4. Semáforo de Severidad (determinado por saturación y probabilidad)
        estado = sat_data['nivel_riesgo']
        semaforo = sat_data['semaforo']
            
        # 5. Desglose de Paros Waterfall (Impacto en la caída del OEE)
        stoppages = pred['stoppage_breakdown']
        total_stoppage_hrs = sum(stoppages.values())
        
        waterfall_impacts = {}
        for cat, hrs in stoppages.items():
            if total_stoppage_hrs > 0:
                waterfall_impacts[cat] = round((hrs / max(1.0, h_prog)) * 100, 2)
            else:
                waterfall_impacts[cat] = 0.0
                
        risk_results.append({
            'line': line,
            'planta': planta,
            'familia': familia,
            'skus_count': skus,
            'volumen_programado': vol_plan,
            'horas_programadas': h_prog,
            'velocidad_nominal': vel_nominal,
            'oee_planeado': oee_plan,
            'oee_meta_nominal': oee_meta_nominal,
            'oee_proyectado_mu': mu_oee,
            'oee_proyectado_sigma': sigma_oee,
            'oee_proyectado_q10': q10_oee,
            'oee_proyectado_q50': q50_oee,
            'oee_proyectado_q90': q90_oee,
            'oee_min': min_oee,
            'oee_max': max_oee,
            'delta_oee': delta_oee,
            # Métricas oficiales de Saturación
            'saturacion_teorica_pct': sat_data['saturacion_teorica_pct'],
            'saturacion_proyectada_pct': sat_data['saturacion_proyectada_pct'],
            'saturacion_q10_pct': sat_data['saturacion_q10_pct'],
            'saturacion_q50_pct': sat_data['saturacion_q50_pct'],
            'saturacion_q90_pct': sat_data['saturacion_q90_pct'],
            'incumple_plan': sat_data['incumple_plan'],
            'pct_incumplimiento_piezas': sat_data['pct_incumplimiento_piezas'],
            'pct_sobresaturacion_capacidad': sat_data['pct_sobresaturacion_capacidad'],
            'probabilidad_saturacion_pct': sat_data['probabilidad_saturacion_pct'],
            'prob_incumplimiento': p_incumplimiento,
            'prob_incumplimiento_pct': round(p_incumplimiento * 100, 1),
            'horas_en_riesgo': horas_en_riesgo,
            'volumen_en_riesgo': volumen_en_riesgo,
            'estado': estado,
            'semaforo': semaforo,
            'stoppages_hours': stoppages,
            'waterfall_impacts': waterfall_impacts
        })
        
    # KPIs Globales de Planta
    tot_vol_plan = sum(r['volumen_programado'] for r in risk_results)
    tot_vol_risk = sum(r['volumen_en_riesgo'] for r in risk_results)
    tot_hrs_risk = sum(r['horas_en_riesgo'] for r in risk_results)
    tot_hrs_prog = sum(r['horas_programadas'] for r in risk_results)
    
    oee_prom_planeado = sum(r['oee_planeado'] * r['horas_programadas'] for r in risk_results) / max(1.0, tot_hrs_prog)
    oee_prom_proyectado = sum(r['oee_proyectado_mu'] * r['horas_programadas'] for r in risk_results) / max(1.0, tot_hrs_prog)
    saturacion_prom_proyectada = sum(r['saturacion_proyectada_pct'] * r['horas_programadas'] for r in risk_results) / max(1.0, tot_hrs_prog)
    
    cumplimiento_global = max(0.0, min(100.0, (1.0 - (tot_vol_risk / max(1.0, tot_vol_plan))) * 100))
    
    lineas_criticas = [r for r in risk_results if r['estado'] == 'Crítico']
    lineas_alerta = [r for r in risk_results if r['estado'] == 'Alerta']
    lineas_estables = [r for r in risk_results if r['estado'] == 'Estable']
    lineas_incumple = [r for r in risk_results if r['incumple_plan']]
    
    summary = {
        'global_kpis': {
            'oee_promedio_planeado': round(oee_prom_planeado * 100, 1),
            'oee_promedio_proyectado': round(oee_prom_proyectado * 100, 1),
            'saturacion_promedio_proyectada': round(saturacion_prom_proyectada, 1),
            'horas_totales_en_riesgo': round(tot_hrs_risk, 1),
            'volumen_total_en_riesgo': round(tot_vol_risk, 0),
            'cumplimiento_global_pct': round(cumplimiento_global, 1),
            'total_lineas': len(risk_results),
            'lineas_incumplimiento_count': len(lineas_incumple),
            'lineas_criticas_count': len(lineas_criticas),
            'lineas_alerta_count': len(lineas_alerta),
            'lineas_estables_count': len(lineas_estables)
        },
        'lines': risk_results
    }
    
    with open(FILE_RISK_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Resultados del motor de riesgo guardados en: {FILE_RISK_RESULTS}")
    print("\n================ RESUMEN EJECUTIVO DE RIESGO DE PLAN NOV ================")
    print(f"OEE Promedio Planeado:   {summary['global_kpis']['oee_promedio_planeado']}%")
    print(f"OEE Promedio Proyectado: {summary['global_kpis']['oee_promedio_proyectado']}%")
    print(f"Horas Totales en Riesgo: {summary['global_kpis']['horas_totales_en_riesgo']:,.1f} hrs")
    print(f"Volumen Total en Riesgo: {summary['global_kpis']['volumen_total_en_riesgo']:,.0f} unidades")
    print(f"Cumplimiento Esperado:   {summary['global_kpis']['cumplimiento_global_pct']}%")
    print(f"Semaforo de Lineas:      [CRITICO]: {len(lineas_criticas)} | [ALERTA]: {len(lineas_alerta)} | [ESTABLE]: {len(lineas_estables)}")
    print("=========================================================================\n")
    
    return summary

if __name__ == "__main__":
    run_risk_engine()
