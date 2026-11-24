"""
Pipeline de Datos y Feature Engineering para OEE 2024-2026 y Plan_nov.
Integra series temporales cuantitativas con métricas NLP de mantenimiento.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import shutil

from src.config import (
    FILE_OEE, FILE_OEE_TEMP, FILE_PLAN, FILE_PLAN_TEMP,
    FILE_VELOCIDAD, FILE_VELOCIDAD_TEMP, FILE_VELOCIDAD_PARQUET,
    FILE_DAILY_MAINTENANCE, FILE_MASTER_OEE, BASE_DIR,
    UNPLANNED_STOPPAGE_COLS, PLANNED_STOPPAGE_COLS,
    LINE_MAPPING
)
from src.nlp_avisos import normalize_line_name

def load_and_clean_oee() -> pd.DataFrame:
    """Carga y limpia la base de datos diaria de OEE 2024-2026."""
    source_file = FILE_OEE if FILE_OEE.exists() else FILE_OEE_TEMP
    try:
        df = pd.read_excel(source_file, sheet_name='BD')
    except Exception:
        df = pd.read_excel(FILE_OEE_TEMP, sheet_name='BD')
        
    print(f"[*] Registros leídos de OEE BD: {len(df)}")
    
    # Identificar columnas clave
    cols = list(df.columns)
    date_col = cols[4]   # 'Fecha'
    line_col = cols[10]  # 'Línea'
    fab_col = cols[13]   # 'Fabricación' (Col N)
    al_col = cols[37]    # 'Tiempo para fabricación' (Col AL)
    
    # Renombrar y seleccionar columnas
    df['Fecha'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Fecha']).copy()
    df['Linea'] = df[line_col].apply(normalize_line_name)
    
    # Conversión numérica de producción y tiempo para fabricar
    df['Fabricacion'] = pd.to_numeric(df[fab_col], errors='coerce').fillna(0.0)
    df['Tiempo_Para_Fabricar'] = pd.to_numeric(df[al_col], errors='coerce').fillna(0.0)
    
    # Conversión numérica de paros no planeados (Cols O a U)
    available_unplanned = [c for c in UNPLANNED_STOPPAGE_COLS if c in df.columns]
    for c in available_unplanned:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    df['Paros_No_Planeados'] = df[available_unplanned].sum(axis=1)
    
    # Conversión numérica de paros planeados (Cols V a AK)
    available_planned = [c for c in PLANNED_STOPPAGE_COLS if c in df.columns]
    for c in available_planned:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    df['Paros_Planeados'] = df[available_planned].sum(axis=1)
    
    # REGLA DE NEGOCIO DEL PRD:
    # Horas para Fabricar = 24 - sum(Paros Planeados) o Tiempo_Para_Fabricar
    # Si Col AL está presente y > 0, es exactamente el tiempo neto programado.
    df['Horas_Para_Fabricar'] = df['Tiempo_Para_Fabricar']
    # En caso de discrepancia menor o cero con paros registrados:
    mask_recalc = (df['Horas_Para_Fabricar'] == 0) & (df['Fabricacion'] > 0)
    df.loc[mask_recalc, 'Horas_Para_Fabricar'] = df.loc[mask_recalc, 'Fabricacion'] + df.loc[mask_recalc, 'Paros_No_Planeados']
    
    # OEE Real Calculado: (Horas de Fabricación [Col N]) / (Horas para Fabricar [Col AL])
    # Cuando Horas_Para_Fabricar > 0, calculamos ratio (0.0 a 1.0)
    df['OEE_Real'] = np.where(
        df['Horas_Para_Fabricar'] > 0.05,
        (df['Fabricacion'] / df['Horas_Para_Fabricar']).clip(0.0, 1.0),
        np.nan  # Día sin programación o parada técnica total
    )
    
    return df

def feature_engineering_oee(df_oee: pd.DataFrame) -> pd.DataFrame:
    """Calcula lags, medias móviles, desviaciones estándar y proporciones de paro."""
    print("[*] Generando características temporales y lags de OEE...")
    
    dfs = []
    for line, grp in df_oee.groupby('Linea'):
        grp = grp.sort_values('Fecha').copy()
        
        # Interpolar suavemente días sin producción para cálculo continuo de tendencia
        oee_series = grp['OEE_Real'].ffill().bfill().fillna(0.50)
        
        # Lags
        grp['OEE_Lag_1'] = oee_series.shift(1).bfill()
        grp['OEE_Lag_7'] = oee_series.shift(7).bfill()
        grp['OEE_Lag_14'] = oee_series.shift(14).bfill()
        grp['OEE_Lag_30'] = oee_series.shift(30).bfill()
        
        # Medias móviles y volatilidad
        grp['OEE_Roll_7d'] = oee_series.rolling(7, min_periods=1).mean()
        grp['OEE_Roll_14d'] = oee_series.rolling(14, min_periods=1).mean()
        grp['OEE_Roll_30d'] = oee_series.rolling(30, min_periods=1).mean()
        grp['OEE_Std_7d'] = oee_series.rolling(7, min_periods=2).std().fillna(0.05)
        grp['OEE_Std_30d'] = oee_series.rolling(30, min_periods=5).std().fillna(0.08)
        
        # Medias móviles de paros no planeados
        grp['Paros_No_Plan_Roll_7d'] = grp['Paros_No_Planeados'].rolling(7, min_periods=1).mean()
        grp['Paros_No_Plan_Roll_30d'] = grp['Paros_No_Planeados'].rolling(30, min_periods=1).mean()
        
        if 'Ineficiencia' in grp.columns:
            grp['Ineficiencia_Roll_7d'] = grp['Ineficiencia'].rolling(7, min_periods=1).mean()
        if 'Cambios y limpiezas' in grp.columns:
            grp['Cambios_Roll_7d'] = grp['Cambios y limpiezas'].rolling(7, min_periods=1).mean()
        if 'Falta de material' in grp.columns:
            grp['Falta_Material_Roll_7d'] = grp['Falta de material'].rolling(7, min_periods=1).mean()
            
        dfs.append(grp)
        
    df_feat = pd.concat(dfs, ignore_index=True)
    
    # Características de calendario
    df_feat['Dia_Semana'] = df_feat['Fecha'].dt.dayofweek
    df_feat['Mes'] = df_feat['Fecha'].dt.month
    df_feat['Sin_Dia_Semana'] = np.sin(2 * np.pi * df_feat['Dia_Semana'] / 7)
    df_feat['Cos_Dia_Semana'] = np.cos(2 * np.pi * df_feat['Dia_Semana'] / 7)
    
    return df_feat

def merge_oee_and_maintenance() -> pd.DataFrame:
    """Combina la serie temporal de OEE con las variables extraídas de avisos."""
    print("[*] Cruzando series de OEE con vectores de mantenimiento (Avisos)...")
    
    df_oee = load_and_clean_oee()
    df_oee_feat = feature_engineering_oee(df_oee)
    
    # Cargar features de mantenimiento
    if not FILE_DAILY_MAINTENANCE.exists():
        from src.nlp_avisos import load_and_process_avisos, generate_daily_maintenance_features
        df_av = load_and_process_avisos()
        df_maint = generate_daily_maintenance_features(df_av)
    else:
        df_maint = pd.read_parquet(FILE_DAILY_MAINTENANCE)
        
    df_maint['Fecha'] = pd.to_datetime(df_maint['Fecha'])
    
    # Merge left con OEE
    master_df = pd.merge(df_oee_feat, df_maint, on=['Linea', 'Fecha'], how='left')
    
    # Imputar ceros para días donde no hubo avisos de mantenimiento
    maint_cols = [c for c in df_maint.columns if c not in ['Linea', 'Fecha']]
    master_df[maint_cols] = master_df[maint_cols].fillna(0.0)
    
    # Asegurar tipos limpios para parquet/csv evitando strings de guiones en columnas numéricas
    for c in master_df.columns:
        if c in ['Linea', 'Fecha']:
            continue
        # Intentar convertir columnas de objeto que son numéricas con ' -   '
        if master_df[c].dtype == 'object':
            converted = pd.to_numeric(master_df[c].astype(str).str.replace('-', '').str.strip(), errors='coerce')
            if converted.notna().sum() > 0:
                master_df[c] = converted.fillna(0.0)
            else:
                master_df[c] = master_df[c].astype(str)
                
    master_df.to_parquet(FILE_MASTER_OEE, index=False)
    print(f"[+] Master dataset generado en: {FILE_MASTER_OEE} ({len(master_df)} filas, {len(master_df.columns)} columnas)")
    
    return master_df

def load_and_aggregate_plan() -> pd.DataFrame:
    """Carga y procesa Plan_nov.xlsx agregando por línea."""
    source_file = FILE_PLAN if FILE_PLAN.exists() else FILE_PLAN_TEMP
    try:
        df = pd.read_excel(source_file)
    except Exception:
        df = pd.read_excel(FILE_PLAN_TEMP)
        
    print(f"[*] Registros leídos de Plan_nov: {len(df)}")
    
    # Normalizar columnas
    line_col = [c for c in df.columns if 'linea' in c.lower()][0]
    vol_col = [c for c in df.columns if 'plan de liberac' in c.lower() or 'pvo' in c.lower()][0]
    oee_col = [c for c in df.columns if 'oee planeado' in c.lower() or 'meta' in c.lower()][0]
    
    df['Linea_Norm'] = df[line_col].apply(normalize_line_name)
    df['Volumen_PVO'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0.0)
    df['OEE_Planeado'] = pd.to_numeric(df[oee_col], errors='coerce').fillna(0.50)
    
    # Agrupación por línea con OEE planeado ponderado por volumen
    plan_summary = []
    for line, grp in df.groupby('Linea_Norm'):
        tot_vol = grp['Volumen_PVO'].sum()
        if tot_vol > 0:
            weighted_oee = (grp['OEE_Planeado'] * grp['Volumen_PVO']).sum() / tot_vol
        else:
            weighted_oee = grp['OEE_Planeado'].mean()
            
        skus = grp['Material Padre'].nunique() if 'Material Padre' in grp.columns else len(grp)
        planta = grp['Planta'].iloc[0] if 'Planta' in grp.columns else 'PiSA'
        familia = grp['Familia-Subfamilia'].iloc[0] if 'Familia-Subfamilia' in grp.columns else 'Farmacéutica'
        
        plan_summary.append({
            'Linea': line,
            'Planta': planta,
            'Familia': familia,
            'SKUs_Count': skus,
            'Volumen_Programado': tot_vol,
            'OEE_Planeado': round(weighted_oee, 4),
            'OEE_Planeado_Min': round(grp['OEE_Planeado'].min(), 4),
            'OEE_Planeado_Max': round(grp['OEE_Planeado'].max(), 4),
        })
        
    df_plan_agg = pd.DataFrame(plan_summary)
    return df_plan_agg

def load_and_process_velocidad_oee_meta() -> pd.DataFrame:
    """Carga y estandariza la tabla de Velocidad Nominal y OEE Meta por línea."""
    print("[*] Procesando archivo de Velocidades Nominales y OEE Meta...")
    df = None
    for target in [FILE_VELOCIDAD, FILE_VELOCIDAD_TEMP]:
        if target.exists():
            try:
                temp_read = BASE_DIR / "temp_read_vel.xlsx"
                try:
                    shutil.copyfile(target, temp_read)
                    df = pd.read_excel(temp_read)
                except Exception:
                    df = pd.read_excel(target)
                break
            except Exception:
                continue
                
    if df is None:
        raise FileNotFoundError("No se pudo leer 'Velocidad y OEE meta.xlsx' ni sus copias temporales.")
        
    df.columns = [c.strip().upper() for c in df.columns]
    subfam_col = [c for c in df.columns if 'SUBFAMILIA' in c or 'LINEA' in c][0]
    vel_col = [c for c in df.columns if 'VELOCIDAD' in c][0]
    oee_col = [c for c in df.columns if 'OEE' in c][0]
    
    df['Linea'] = df[subfam_col].astype(str).str.strip().apply(normalize_line_name)
    df['Velocidad_Nominal'] = pd.to_numeric(df[vel_col], errors='coerce').fillna(0.0)
    df['OEE_Meta'] = pd.to_numeric(df[oee_col], errors='coerce').fillna(0.60)
    
    clean_df = df[['Linea', 'Velocidad_Nominal', 'OEE_Meta']].drop_duplicates(subset=['Linea']).reset_index(drop=True)
    clean_df.to_parquet(FILE_VELOCIDAD_PARQUET, index=False)
    print(f"[+] Velocidades y OEE Meta guardados en: {FILE_VELOCIDAD_PARQUET} ({len(clean_df)} líneas)")
    return clean_df

def get_line_specs_dict() -> dict:
    """Retorna un diccionario {linea: {'velocidad_nominal': float, 'oee_meta': float}}."""
    if not FILE_VELOCIDAD_PARQUET.exists():
        df = load_and_process_velocidad_oee_meta()
    else:
        df = pd.read_parquet(FILE_VELOCIDAD_PARQUET)
        
    specs = {}
    for _, row in df.iterrows():
        specs[row['Linea']] = {
            'velocidad_nominal': float(row['Velocidad_Nominal']),
            'oee_meta': float(row['OEE_Meta'])
        }
    return specs

if __name__ == "__main__":
    master = merge_oee_and_maintenance()
    plan = load_and_aggregate_plan()
    print("\n--- Resumen de Plan_nov por Línea ---")
    print(plan[['Linea', 'SKUs_Count', 'Volumen_Programado', 'OEE_Planeado']])
