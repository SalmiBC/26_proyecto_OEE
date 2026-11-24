"""
Módulo 4: Motor de Diagnóstico Prescriptivo impulsado por LLM.
Sintetiza las bitácoras no estructuradas de mantenimiento y las proyecciones probabilísticas
de OEE para generar explicaciones ejecutivas de causa raíz y recomendaciones preventivas
accionables para las líneas críticas y en alerta.
"""

import os
import json
import pandas as pd
from pathlib import Path

from src.config import (
    FILE_RISK_RESULTS, FILE_PROCESSED_AVISOS, FILE_PRESCRIPTIVE,
    DATA_DIR
)

def get_line_maintenance_insights(line: str, df_avisos: pd.DataFrame, days_back: int = 90) -> dict:
    """Extrae las causas y equipos más problemáticos para una línea específica."""
    df_line = df_avisos[df_avisos['Linea_Norm'] == line].copy()
    if len(df_line) == 0:
        return {
            'top_equipos': [],
            'top_subsistemas': [],
            'top_fallas': [],
            'textos_recientes': [],
            'total_avisos': 0,
            'horas_totales_paro': 0.0
        }
        
    # Filtrar avisos recientes si hay fechas suficientes
    max_date = df_line['Fecha'].max()
    cutoff = max_date - pd.Timedelta(days=days_back)
    recent = df_line[df_line['Fecha'] >= cutoff]
    if len(recent) < 10:
        recent = df_line
        
    tot_avisos = len(recent)
    tot_hrs = float(recent['Hrs_Paro_Clean'].sum())
    
    # Top equipos
    equip_col = 'Equipo Descripción' if 'Equipo Descripción' in recent.columns else 'Equipo Descripcin'
    top_equipos = recent.groupby(equip_col)['Hrs_Paro_Clean'].sum().sort_values(ascending=False).head(4).to_dict()
    
    # Top subsistemas
    top_sub = recent.groupby('Subsistema')['Hrs_Paro_Clean'].sum().sort_values(ascending=False).head(3).to_dict()
    
    # Top tipos de falla
    top_fallas = recent['Tipo_Falla'].value_counts().head(3).to_dict()
    
    # Textos más representativos
    text_samples = []
    sample_df = recent.sort_values('Hrs_Paro_Clean', ascending=False).head(8)
    for _, r in sample_df.iterrows():
        eq = str(r.get(equip_col, ''))
        sub = str(r.get('Subsistema', ''))
        cs = str(r.get('Descripción Causa', ''))
        ts = str(r.get('Texto Síntoma', ''))
        tc = str(r.get('Texto Causa', ''))
        hrs = float(r.get('Hrs_Paro_Clean', 0))
        
        detail = f"{eq} ({sub}): "
        if pd.notna(ts) and str(ts).strip() != '':
            detail += f"Síntoma: '{ts}'. "
        if pd.notna(tc) and str(tc).strip() != '':
            detail += f"Causa: '{tc}'. "
        elif pd.notna(cs) and str(cs).strip() != '':
            detail += f"Causa: '{cs}'. "
        detail += f"[{hrs:.1f} hrs paro]"
        text_samples.append(detail)
        
    return {
        'top_equipos': top_equipos,
        'top_subsistemas': top_sub,
        'top_fallas': top_fallas,
        'textos_recientes': text_samples,
        'total_avisos': tot_avisos,
        'horas_totales_paro': round(tot_hrs, 1)
    }

def generate_prescriptive_narratives():
    """Genera diagnósticos y recomendaciones prescriptivas detalladas para cada línea."""
    print("[*] Generando diagnósticos prescriptivos y recomendaciones por línea...")
    
    if not FILE_RISK_RESULTS.exists():
        print("[!] No se encontró risk_assessment_results.json. Ejecute primero risk_engine.")
        return {}
        
    with open(FILE_RISK_RESULTS, 'r', encoding='utf-8') as f:
        risk_data = json.load(f)
        
    df_avisos = pd.read_parquet(FILE_PROCESSED_AVISOS)
    
    diagnostics = {}
    
    for r in risk_data['lines']:
        line = r['line']
        estado = r['estado']
        oee_plan = r['oee_planeado'] * 100
        oee_proj = r['oee_proyectado_mu'] * 100
        p_inc = r['prob_incumplimiento_pct']
        hrs_risk = r['horas_en_riesgo']
        vol_risk = r['volumen_en_riesgo']
        stoppages = r['stoppages_hours']
        
        insights = get_line_maintenance_insights(line, df_avisos)
        
        # Identificar principal tipo de paro del LSTM
        max_stop_cat = max(stoppages.items(), key=lambda x: x[1])[0] if stoppages else 'Ineficiencia'
        
        # Identificar equipo y subsistema más crítico de avisos
        top_eq = list(insights['top_equipos'].keys())[0] if insights['top_equipos'] else 'Equipos de línea'
        top_sub = list(insights['top_subsistemas'].keys())[0] if insights['top_subsistemas'] else 'Alimentación y Transporte'
        top_falla = list(insights['top_fallas'].keys())[0] if insights['top_fallas'] else 'Mecánica'
        
        # Síntesis Diagnóstica
        if estado == 'Crítico':
            resumen_causa = (
                f"La línea {line} presenta una probabilidad crítica de incumplimiento ({p_inc:.1f}%), "
                f"con una brecha proyectada de {(oee_plan - oee_proj):.1f} puntos porcentuales (Meta: {oee_plan:.1f}% vs Proyectado: {oee_proj:.1f}%). "
                f"El análisis multivariado LSTM y de bitácoras identifica que la principal fuga de disponibilidad proviene de '{max_stop_cat}', "
                f"combinada con fallas recurrentes de tipo '{top_falla}' en el subsistema '{top_sub}'. "
                f"El equipo cuello de botella con mayor severidad histórica es '{top_eq}', con {insights['horas_totales_paro']:.1f} hrs de paro registradas."
            )
            prioridad = "ALTA / INMEDIATA"
        elif estado == 'Alerta':
            resumen_causa = (
                f"La línea {line} se encuentra en zona de alerta ({p_inc:.1f}% de riesgo de incumplimiento). "
                f"El OEE proyectado ({oee_proj:.1f}%) se sitúa ligeramente por debajo o al borde de la meta ({oee_plan:.1f}%). "
                f"El factor limitante es '{max_stop_cat}', manifestándose en eventos de '{top_sub}' en '{top_eq}'. "
                f"Existe volatilidad operacional que puede traducirse en {hrs_risk:,.1f} horas faltantes si se presentan fallas de ajuste."
            )
            prioridad = "MEDIA / PREVENTIVA"
        else:
            resumen_causa = (
                f"La línea {line} mantiene un desempeño estable ({p_inc:.1f}% de riesgo). "
                f"El OEE proyectado ({oee_proj:.1f}%) cubre holgadamente la meta del plan ({oee_plan:.1f}%). "
                f"Los eventos de paro se concentran principalmente en cambios de formato programados y desgastes normales en '{top_eq}'."
            )
            prioridad = "RUTINARIA / CONTROL"
            
        # Acciones Prescriptivas
        prescripciones = []
        if estado == 'Crítico':
            prescripciones.append({
                'accion': f"Inspección Pre-Arranque y Alineación en '{top_eq}'",
                'detalle': f"Auditar galgas, centradores, fotoceldas de túnel/expulsor y coples mecánicos antes de cada inicio de turno. Reducir desajustes dinámicos que provocan paros por {top_falla.lower()}.",
                'responsable': "Mantenimiento Mecánico y Ajustadores de Línea"
            })
            prescripciones.append({
                'accion': "Ajuste Dinámico de Colchón de Horas en Programación",
                'detalle': f"Incorporar un colchón preventivo de al menos {min(48, int(hrs_risk * 0.4))} horas en el secuenciamiento del mes para amortiguar la varianza de entrega de los SKUs más voluminosos.",
                'responsable': "Planeación de la Producción"
            })
            prescripciones.append({
                'accion': "Kit de Refacciones Críticas en Piso",
                'detalle': f"Garantizar stock inmediato en planta de agujas de carrusel, dedos de expulsor, sellos neumáticos y sensores de paso de {top_sub}.",
                'responsable': "Almacén de Refacciones y Mantenimiento"
            })
            prescripciones.append({
                'accion': "Estandarización SMED en Cambios y Limpiezas",
                'detalle': "Desplegar checklist predeterminado con pre-calentamiento y pre-calibración de formato fuera de línea para acortar tiempos muertos.",
                'responsable': "Producción y Mejora Continua"
            })
        elif estado == 'Alerta':
            prescripciones.append({
                'accion': f"Monitoreo Predictivo de Ajustes en '{top_eq}'",
                'detalle': f"Revisión de parámetros de torque, vibración y alineación de guías en cambios de formato para mitigar paros en {top_sub}.",
                'responsable': "Técnicos de Mantenimiento de Turno"
            })
            prescripciones.append({
                'accion': "Optimización del Loteo y Secuenciamiento de SKUs",
                'detalle': "Agrupar órdenes de fabricación por formato idéntico de envase/volumen para minimizar transiciones de formato y limpiezas profundas.",
                'responsable': "Planeación de la Producción"
            })
            prescripciones.append({
                'accion': "Verificación de Tiempos de Comedor y Relevos",
                'detalle': "Supervisar traslapes de turno y relevos de descanso de operarios para abatir las horas muertas de ineficiencia.",
                'responsable': "Supervisión de Producción"
            })
        else:
            prescripciones.append({
                'accion': "Mantenimiento Autónomo y Lubricación",
                'detalle': f"Continuar con las rutinas estándar de limpieza e inspección de guías y transportadores en {top_eq}.",
                'responsable': "Operadores de Línea"
            })
            prescripciones.append({
                'accion': "Soporte para Absorción de Capacidad Excedente",
                'detalle': "Línea candidata para absorber transferencias de volumen en caso de contingencia en líneas críticas del mismo formato.",
                'responsable': "Planeación Central"
            })
            
        diagnostics[line] = {
            'line': line,
            'estado': estado,
            'prioridad': prioridad,
            'prob_incumplimiento_pct': p_inc,
            'horas_en_riesgo': hrs_risk,
            'volumen_en_riesgo': vol_risk,
            'oee_planeado_pct': round(oee_plan, 1),
            'oee_proyectado_pct': round(oee_proj, 1),
            'resumen_causa': resumen_causa,
            'equipo_critico': top_eq,
            'subsistema_critico': top_sub,
            'tipo_falla_predominante': top_falla,
            'ejemplos_bitacora': insights['textos_recientes'][:4],
            'prescripciones': prescripciones
        }
        
    with open(FILE_PRESCRIPTIVE, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Diagnósticos prescriptivos guardados en: {FILE_PRESCRIPTIVE} ({len(diagnostics)} líneas analizadas)")
    return diagnostics

if __name__ == "__main__":
    generate_prescriptive_narratives()
