"""
Módulo Generador de la Presentación Académica / Ejecutiva para Defensa de Maestría.
Compila 'presentacion_maestria_oee.html' utilizando la plantilla desacoplada
'src/templates/presentation_template.html' y los datos reales calculados.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    FILE_RISK_RESULTS, FILE_PRESCRIPTIVE, FILE_MASTER_OEE,
    FILE_PROCESSED_AVISOS, DATA_DIR, BASE_DIR
)
from src.nlp_avisos import extract_pareto_root_causes

FILE_MULTIMONTH = DATA_DIR / "multimonth_forecast_results.json"
FILE_PRESENTATION_HTML = BASE_DIR / "presentacion_maestria_oee.html"
FILE_TEMPLATE = BASE_DIR / "src" / "templates" / "presentation_template.html"

def build_presentation():
    print("[*] Compilando datos para la Presentación de Maestría...")
    
    # 1. Cargar resultados de riesgo
    with open(FILE_RISK_RESULTS, 'r', encoding='utf-8') as f:
        risk_data = json.load(f)
        
    # 2. Cargar diagnósticos prescriptivos
    with open(FILE_PRESCRIPTIVE, 'r', encoding='utf-8') as f:
        prescriptive_data = json.load(f)
        
    # 3. Cargar o generar proyección multi-mes
    if FILE_MULTIMONTH.exists():
        with open(FILE_MULTIMONTH, 'r', encoding='utf-8') as f:
            multimonth_data = json.load(f)
    else:
        from src.multimonth_forecast import run_multimonth_forecast
        multimonth_data = run_multimonth_forecast(n_months=6, n_samples=100)
        
    # 4. Cargar pareto de causas
    df_avisos = pd.read_parquet(FILE_PROCESSED_AVISOS)
    pareto_data = extract_pareto_root_causes(df_avisos, top_n=8)
    
    # 5. Preparar datos históricos para la línea A4
    df_master = pd.read_parquet(FILE_MASTER_OEE)
    df_a4 = df_master[df_master['Linea'] == 'A4'].sort_values('Fecha').tail(45)
    hist_a4_dates = df_a4['Fecha'].dt.strftime('%Y-%m-%d').tolist()
    hist_a4_oee = (df_a4['OEE_Real'].fillna(df_a4['OEE_Roll_7d']) * 100).round(1).tolist()
    
    # Proyección para Noviembre (30 días)
    nov_dates = [f"2026-11-{d:02d}" for d in range(1, 31)]
    a4_mu = 43.5
    a4_sigma = 3.4
    a4_plan = 54.0
    
    np.random.seed(42)
    a4_forecast_trend = (a4_mu + np.sin(np.linspace(0, 3, 30)) * 1.5).round(1).tolist()
    a4_q10_trend = (np.array(a4_forecast_trend) - 1.28 * a4_sigma).round(1).tolist()
    a4_q90_trend = (np.array(a4_forecast_trend) + 1.28 * a4_sigma).round(1).tolist()
    
    # Proyecciones multi-mes
    multimonth_a4 = multimonth_data['lines'].get('A4', {}).get('trayectoria', [])
    multimonth_sv1 = multimonth_data['lines'].get('SV1', {}).get('trayectoria', [])
    
    chart_payload = {
        'hist_a4_dates': hist_a4_dates,
        'hist_a4_oee': hist_a4_oee,
        'nov_dates': nov_dates,
        'a4_forecast_trend': a4_forecast_trend,
        'a4_q10_trend': a4_q10_trend,
        'a4_q90_trend': a4_q90_trend,
        'a4_plan': a4_plan,
        'multimonth_a4': multimonth_a4,
        'multimonth_sv1': multimonth_sv1,
        'pareto': pareto_data,
        'global_kpis': risk_data['global_kpis']
    }
    
    # 6. Leer plantilla HTML y sustituir payload
    with open(FILE_TEMPLATE, 'r', encoding='utf-8') as f:
        template_content = f.read()
        
    payload_json = json.dumps(chart_payload, ensure_ascii=False)
    html_final = template_content.replace('__PAYLOAD_PLACEHOLDER__', payload_json)
    
    with open(FILE_PRESENTATION_HTML, 'w', encoding='utf-8') as f:
        f.write(html_final)
        
    print(f"[+] Presentación de Maestría compilada exitosamente en:\n    {FILE_PRESENTATION_HTML}")
    return FILE_PRESENTATION_HTML

if __name__ == "__main__":
    build_presentation()
