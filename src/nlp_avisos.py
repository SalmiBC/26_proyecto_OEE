"""
Módulo 1: Normalización de Bitácoras con LLM (avisos).
Procesa registros no estructurados de mantenimiento, clasifica mediante taxonomía
industrial (Subsistema, Tipo de Falla, Severidad) y genera vectores diarios por línea.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata

from src.config import (
    FILE_AVISOS, FILE_AVISOS_TEMP, DATA_DIR,
    FILE_PROCESSED_AVISOS, FILE_DAILY_MAINTENANCE, LINE_MAPPING
)

def strip_accents(text: str) -> str:
    """Normaliza texto removiendo acentos y caracteres especiales."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    return "".join(c for c in text if unicodedata.category(c) != 'Mn').lower().strip()

def normalize_line_name(val) -> str:
    """Normaliza el nombre de la línea para alinearlo con Plan_nov."""
    if pd.isna(val):
        return "DESCONOCIDA"
    s = str(val).strip().upper()
    s = strip_accents(s).upper()
    
    # Reemplazos comunes
    if "OFTALM" in s or s == "OFT":
        return "OFT"
    if "POLVO" in s:
        return "POLVOS"
    if "JERINGA" in s:
        return "JERINGAS"
    if s in ["SV1", "L-SV1", "LINEA SV1"]:
        return "SV1"
    if s in ["A-4", "A 4"]:
        return "A4"
    if s in ["A-5", "A 5"]:
        return "A5"
    if s.startswith("LINEA "):
        s = s.replace("LINEA ", "LÍNEA ")
        
    return LINE_MAPPING.get(s, s)

def classify_subsystem(text: str) -> str:
    """Identifica el subsistema afectado a partir del texto completo del aviso."""
    t = strip_accents(text)
    
    if any(k in t for k in ['lavad', 'tunel', 'aguja de lavado', 'esteril', 'despirogen', 'despiro']):
        return 'Lavado y Esterilización'
    elif any(k in t for k in ['dosific', 'llenad', 'bomba dos', 'piston', 'valvula de dos', 'aguja de llen', 'pesaje en lin']):
        return 'Dosificación y Llenado'
    elif any(k in t for k in ['cierre', 'sellad', 'engargol', 'tapon', 'tapa', 'boca ampolleta', 'sopletes', 'quemador']):
        return 'Sellado y Engargolado'
    elif any(k in t for k in ['etiquet', 'impres', 'fechador', 'ribbon', 'hot stamping', 'codificador']):
        return 'Etiquetado e Impresión'
    elif any(k in t for k in ['vision', 'camara', 'revisad', 'inspecc', 'particul', 'hermet', 'cmp']):
        return 'Inspección y Visión'
    elif any(k in t for k in ['encarton', 'bliste', 'caja', 'formado', 'instruct', 'estuche', 'fajill', 'paletiz', 'sleever']):
        return 'Empaque y Encartonado'
    elif any(k in t for k in ['transport', 'cadena', 'banda', 'estrella', 'empujador', 'tornillo sin fin', 'carrusel', 'expulsor', 'alimentad']):
        return 'Alimentación y Transporte'
    elif any(k in t for k in ['panel', 'hmi', 'plc', 'variador', 'sensor', 'fotoceld', 'fibra', 'encoder', 'senal', 'alarma']):
        return 'Control y Sensores'
    elif any(k in t for k in ['neumat', 'vacio', 'aire', 'vapor', 'chiller', 'hidraul', 'filtro']):
        return 'Servicios y Neumática'
    else:
        return 'Mecanismo General'

def classify_fault_type(text: str, causa_raw: str = "") -> str:
    """
    Clasifica el tipo de falla en una de las 5 categorías requeridas:
    ['Mecánica', 'Eléctrica', 'Ajuste', 'Desgaste', 'Operación']
    """
    t = strip_accents(text)
    c = strip_accents(causa_raw)
    
    # 1. Ajuste
    if any(k in t for k in ['desalinea', 'calibr', 'ajuste', 'parametr', 'formato', 'posicion', 'centrad', 'holgura', 'desfas', 'sincron']):
        return 'Ajuste'
    
    # 2. Desgaste
    if any(k in t for k in ['desgaste', 'fatiga', 'vida util', 'friccion', 'holgura por uso', 'degradac', 'oxidac', 'corros']):
        return 'Desgaste'
        
    # 3. Operación
    if any(k in t for k in ['operacion', 'operador', 'error humano', 'falta de material', 'limpieza', 'rechazo por lote', 'atasco por ampolleta']):
        return 'Operación'
        
    # 4. Eléctrica / Electrónica
    if c in ['electronica', 'electrica', 'causas electricas'] or any(k in t for k in ['electr', 'motor', 'servomotor', 'corto', 'fusible', 'cable', 'plc', 'senal', 'software', 'tarjeta']):
        return 'Eléctrica'
        
    # 5. Mecánica por defecto / explícita
    return 'Mecánica'

def calculate_severity(hrs_paro: float, subsystem: str, fault_type: str, text: str) -> int:
    """
    Calcula severidad de 1 a 5 según duración del paro y criticidad del subsistema.
    """
    t = strip_accents(text)
    hrs = float(hrs_paro) if pd.notna(hrs_paro) and hrs_paro > 0 else 0.5
    
    # Outliers extremos en horas (avisos no cerrados a tiempo) se limitan a 24h
    hrs_eff = min(hrs, 24.0)
    
    # Puntuación base por tiempo
    if hrs_eff < 0.5:
        base = 1
    elif hrs_eff < 1.5:
        base = 2
    elif hrs_eff < 4.0:
        base = 3
    elif hrs_eff < 8.0:
        base = 4
    else:
        base = 5
        
    # Factores agravantes: subsistema crítico (Lavado, Esterilización, Dosificación, Visión)
    if subsystem in ['Lavado y Esterilización', 'Dosificación y Llenado', 'Inspección y Visión']:
        if any(w in t for w in ['rotura', 'dano mayor', 'reemplazo', 'cople danado', 'quiebra', 'bloqueo total']):
            base = min(5, base + 1)
            
    return base

def is_critical_equipment(equip_desc: str) -> bool:
    """Determina si el equipo es cuello de botella crítico en la línea."""
    t = strip_accents(equip_desc)
    return any(k in t for k in ['lavadora', 'dosificadora', 'tunel', 'horno', 'esterilizador', 'revisadora', 'cmp', 'encartonadora'])

def load_and_process_avisos() -> pd.DataFrame:
    """Lee y procesa completamente el archivo avisos.xlsx."""
    source_file = FILE_AVISOS if FILE_AVISOS.exists() else FILE_AVISOS_TEMP
    try:
        df = pd.read_excel(source_file, header=1)
    except Exception:
        df = pd.read_excel(FILE_AVISOS_TEMP, header=1)
        
    print(f"[*] Registros leídos de Avisos: {len(df)}")
    
    # Normalizar línea: preferir 'Línea Oficial' si existe, de lo contrario 'Linea'
    line_col = 'Línea Oficial' if 'Línea Oficial' in df.columns else 'Lnea Oficial'
    if line_col not in df.columns:
        line_col = [c for c in df.columns if 'linea' in c.lower()][0]
        
    df['Linea_Norm'] = df[line_col].apply(normalize_line_name)
    
    # Normalizar fecha
    date_col = 'Dia' if 'Dia' in df.columns else 'Fecha Inicio'
    df['Fecha'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    df = df.dropna(subset=['Fecha']).copy()
    
    # Horas de paro
    hrs_col = 'Hrs Paro' if 'Hrs Paro' in df.columns else [c for c in df.columns if 'hrs' in c.lower()][0]
    df['Hrs_Paro'] = pd.to_numeric(df[hrs_col], errors='coerce').fillna(0.5)
    # Truncar tiempos absurdos por avisos abiertos meses
    df['Hrs_Paro_Clean'] = df['Hrs_Paro'].clip(lower=0.01, upper=24.0)
    
    # Concatenación de texto descriptivo completo
    text_fields = ['Equipo Descripción', 'Síntoma', 'Descripción Síntoma', 
                   'Parte Objeto', 'Descripción Objeto', 'Causa', 
                   'Descripción Causa', 'Texto Síntoma', 'Texto Causa']
    
    def combine_text(row):
        parts = []
        for col in text_fields:
            val = row.get(col, '')
            if pd.notna(val) and str(val).strip() != '':
                parts.append(str(val).strip())
        return " | ".join(parts)
        
    df['Full_Text'] = df.apply(combine_text, axis=1)
    
    # Clasificación NLP / Taxonomía
    print("[*] Aplicando taxonomía industrial y NLP a bitácoras...")
    df['Subsistema'] = df['Full_Text'].apply(classify_subsystem)
    
    causa_col = 'Causa' if 'Causa' in df.columns else ''
    df['Tipo_Falla'] = df.apply(lambda r: classify_fault_type(r['Full_Text'], r.get(causa_col, '')), axis=1)
    
    df['Severidad'] = df.apply(
        lambda r: calculate_severity(r['Hrs_Paro_Clean'], r['Subsistema'], r['Tipo_Falla'], r['Full_Text']),
        axis=1
    )
    
    equip_col = 'Equipo Descripción' if 'Equipo Descripción' in df.columns else 'Equipo Descripcin'
    df['Es_Equipo_Critico'] = df[equip_col].apply(is_critical_equipment)
    
    # Guardar bitácoras procesadas
    df.to_parquet(FILE_PROCESSED_AVISOS, index=False)
    print(f"[+] Bitácoras procesadas guardadas en: {FILE_PROCESSED_AVISOS}")
    
    return df

def generate_daily_maintenance_features(df_avisos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa avisos temporalmente generando vectores diarios por línea con variables cuantitativas:
    - Frecuencia_Falla_Mecanica_7d, Horas_Paro_Equipo_Critico_7d, Severidad_Promedio, etc.
    """
    print("[*] Generando agregaciones temporales diarias por línea...")
    
    # Agregación diaria básica por Línea y Fecha
    daily_list = []
    
    for (line, dt), grp in df_avisos.groupby(['Linea_Norm', 'Fecha']):
        tot_avisos = len(grp)
        tot_hrs = grp['Hrs_Paro_Clean'].sum()
        crit_hrs = grp[grp['Es_Equipo_Critico']]['Hrs_Paro_Clean'].sum()
        avg_sev = grp['Severidad'].mean()
        
        # Conteo por tipo de falla
        fallas_mec = (grp['Tipo_Falla'] == 'Mecánica').sum()
        fallas_ele = (grp['Tipo_Falla'] == 'Eléctrica').sum()
        fallas_aju = (grp['Tipo_Falla'] == 'Ajuste').sum()
        fallas_des = (grp['Tipo_Falla'] == 'Desgaste').sum()
        fallas_ope = (grp['Tipo_Falla'] == 'Operación').sum()
        
        daily_list.append({
            'Linea': line,
            'Fecha': dt,
            'Avisos_Total_Dia': tot_avisos,
            'Hrs_Paro_Avisos_Dia': tot_hrs,
            'Hrs_Equipo_Critico_Dia': crit_hrs,
            'Severidad_Media_Dia': avg_sev,
            'Falla_Mecanica_Dia': fallas_mec,
            'Falla_Electrica_Dia': fallas_ele,
            'Falla_Ajuste_Dia': fallas_aju,
            'Falla_Desgaste_Dia': fallas_des,
            'Falla_Operacion_Dia': fallas_ope
        })
        
    df_daily = pd.DataFrame(daily_list)
    
    # Crear rejilla completa de fechas por línea para calcular rolling sin saltos
    all_lines = df_daily['Linea'].unique()
    min_date = df_daily['Fecha'].min()
    max_date = df_daily['Fecha'].max()
    full_dates = pd.date_range(min_date, max_date, freq='D')
    
    grid = pd.MultiIndex.from_product([all_lines, full_dates], names=['Linea', 'Fecha']).to_frame().reset_index(drop=True)
    df_merged = pd.merge(grid, df_daily, on=['Linea', 'Fecha'], how='left').fillna(0)
    
    # Calcular métricas móviles (Rolling 7d, 14d, 30d)
    feature_dfs = []
    for line, grp in df_merged.groupby('Linea'):
        grp = grp.sort_values('Fecha').copy()
        
        # Rolling 7 días
        grp['Frecuencia_Falla_Mecanica_7d'] = grp['Falla_Mecanica_Dia'].rolling(7, min_periods=1).sum()
        grp['Frecuencia_Falla_Electrica_7d'] = grp['Falla_Electrica_Dia'].rolling(7, min_periods=1).sum()
        grp['Frecuencia_Falla_Ajuste_7d'] = grp['Falla_Ajuste_Dia'].rolling(7, min_periods=1).sum()
        grp['Frecuencia_Falla_Desgaste_7d'] = grp['Falla_Desgaste_Dia'].rolling(7, min_periods=1).sum()
        grp['Frecuencia_Falla_Operacion_7d'] = grp['Falla_Operacion_Dia'].rolling(7, min_periods=1).sum()
        
        grp['Horas_Paro_Equipo_Critico_7d'] = grp['Hrs_Equipo_Critico_Dia'].rolling(7, min_periods=1).sum()
        grp['Hrs_Paro_Avisos_7d'] = grp['Hrs_Paro_Avisos_Dia'].rolling(7, min_periods=1).sum()
        grp['Avisos_Total_7d'] = grp['Avisos_Total_Dia'].rolling(7, min_periods=1).sum()
        grp['Severidad_Promedio_7d'] = grp['Severidad_Media_Dia'].replace(0, np.nan).rolling(7, min_periods=1).mean().fillna(1.0)
        
        # Rolling 30 días
        grp['Avisos_Total_30d'] = grp['Avisos_Total_Dia'].rolling(30, min_periods=1).sum()
        grp['Hrs_Paro_Avisos_30d'] = grp['Hrs_Paro_Avisos_Dia'].rolling(30, min_periods=1).sum()
        grp['Horas_Paro_Equipo_Critico_30d'] = grp['Hrs_Equipo_Critico_Dia'].rolling(30, min_periods=1).sum()
        
        feature_dfs.append(grp)
        
    df_features = pd.concat(feature_dfs, ignore_index=True)
    df_features.to_parquet(FILE_DAILY_MAINTENANCE, index=False)
    print(f"[+] Vectores cuantitativos diarios guardados en: {FILE_DAILY_MAINTENANCE} ({len(df_features)} filas)")
    
    return df_features

def extract_pareto_root_causes(df_avisos: pd.DataFrame, top_n: int = 15) -> list:
    """Genera datos de causas raíz para el diagrama de Pareto."""
    # Combinar equipo + descripción de causa o síntoma
    equip_col = 'Equipo Descripción' if 'Equipo Descripción' in df_avisos.columns else 'Equipo Descripcin'
    causa_desc_col = 'Descripción Causa' if 'Descripción Causa' in df_avisos.columns else 'Descripcin Causa'
    sint_desc_col = 'Descripción Síntoma' if 'Descripción Síntoma' in df_avisos.columns else 'Descripcin Sntoma'
    
    df_valid = df_avisos.copy()
    df_valid['Causa_Detallada'] = df_valid[equip_col].fillna('Equipo N/E').astype(str) + " - " + \
                                  df_valid[causa_desc_col].fillna(df_valid[sint_desc_col].fillna('Falla mecánica')).astype(str)
                                  
    pareto = df_valid.groupby('Causa_Detallada').agg({
        'Aviso': 'count',
        'Hrs_Paro_Clean': 'sum'
    }).rename(columns={'Aviso': 'Frecuencia', 'Hrs_Paro_Clean': 'Horas_Totales'}).reset_index()
    
    pareto = pareto.sort_values('Horas_Totales', ascending=False).reset_index(drop=True)
    tot_hrs = pareto['Horas_Totales'].sum()
    pareto['Pct_Individual'] = (pareto['Horas_Totales'] / tot_hrs) * 100
    pareto['Pct_Acumulado'] = pareto['Pct_Individual'].cumsum()
    
    return pareto.head(top_n).to_dict(orient='records')

if __name__ == "__main__":
    df_av = load_and_process_avisos()
    df_feat = generate_daily_maintenance_features(df_av)
    pareto = extract_pareto_root_causes(df_av)
    print("\n--- Top 5 Causas Pareto ---")
    for p in pareto[:5]:
        print(f"Causa: {p['Causa_Detallada']} | Hrs: {p['Horas_Totales']:.1f} | Acum: {p['Pct_Acumulado']:.1f}%")
