"""
Módulo de Predicción Multi-Mes (Horizonte Extendido: 3 a 6 meses).
Implementa un esquema de proyección autorregresivo iterado con propagación estocástica
de incertidumbre mediante Monte Carlo Dropout (Gal & Ghahramani, 2016).
Permite evaluar la evolución del OEE y la dilatación de las bandas de confianza (fan chart).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

from src.config import (
    FILE_MASTER_OEE, MODELS_DIR, DATA_DIR, LSTM_CONFIG,
    UNPLANNED_STOPPAGE_COLS
)
from src.lstm_model import MultiOutputMCDropoutLSTM, FEATURE_COLS

FILE_MULTIMONTH_RESULTS = DATA_DIR / "multimonth_forecast_results.json"

def run_multimonth_forecast(n_months: int = 6, n_samples: int = 100):
    """
    Ejecuta la proyección recurrente multi-mes para las líneas principales de PiSA.
    Retorna la trayectoria de medias, cuantiles e incertidumbre acumulada.
    """
    print(f"[*] Iniciando proyección multi-mes a horizonte de {n_months} meses (N={n_samples} MC passes)...")
    
    # 1. Cargar dataset maestro
    df_master = pd.read_parquet(FILE_MASTER_OEE)
    
    # 2. Cargar modelo y scaler
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
    
    model.train()  # Activar Dropout para Monte Carlo
    
    # Meses a proyectar
    month_names = [
        "Nov 2026 (M1)", "Dic 2026 (M2)", "Ene 2027 (M3)",
        "Feb 2027 (M4)", "Mar 2027 (M5)", "Abr 2027 (M6)"
    ][:n_months]
    
    seq_len = config["sequence_length"]
    lines_to_project = ['A4', 'A5', 'ANES', 'AV1', 'AV2', 'AV3', 'AV6', 'BP', 'HM', 'JERINGAS', 'OFT', 'POLVOS', 'SOV', 'SV1', 'VA1', 'VA2', 'VA3', 'VA4']
    
    multimonth_results = {}
    
    for line in lines_to_project:
        df_l = df_master[df_master['Linea'] == line].sort_values('Fecha').copy()
        if len(df_l) < seq_len:
            continue
            
        # Tomar la ventana más reciente
        recent_window = df_l[FEATURE_COLS].tail(seq_len).fillna(0.0).values.copy()
        current_oee_history = list(df_l['OEE_Real'].dropna().tail(30).values)
        if len(current_oee_history) == 0:
            current_oee_history = [0.5] * 30
            
        line_trajectory = []
        
        for m_idx, m_name in enumerate(month_names):
            # Escalar ventana actual
            scaled_seq = scaler.transform(recent_window)
            x_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
            
            # Muestreo Monte Carlo Dropout
            oee_draws = []
            with torch.no_grad():
                for _ in range(n_samples):
                    p_oee, _ = model(x_tensor)
                    # Añadir componente estocástica de difusión browniana acumulada por horizonte
                    noise = np.random.normal(0, 0.008 * np.sqrt(m_idx + 1))
                    val = np.clip(p_oee.item() + noise, 0.20, 0.85)
                    oee_draws.append(val)
                    
            oee_draws = np.array(oee_draws)
            mu_m = float(np.mean(oee_draws))
            sigma_m = float(np.std(oee_draws))
            q10_m = float(np.percentile(oee_draws, 10))
            q50_m = float(np.percentile(oee_draws, 50))
            q90_m = float(np.percentile(oee_draws, 90))
            
            line_trajectory.append({
                'mes_indice': m_idx + 1,
                'mes_nombre': m_name,
                'oee_mu': round(mu_m * 100, 2),
                'oee_sigma': round(sigma_m * 100, 2),
                'oee_q10': round(q10_m * 100, 2),
                'oee_q50': round(q50_m * 100, 2),
                'oee_q90': round(q90_m * 100, 2)
            })
            
            # Actualización autorregresiva de la ventana para el siguiente mes (t+1 -> t+2)
            # Desplazar la ventana 1 paso e insertar las nuevas features proyectadas
            current_oee_history.append(mu_m)
            if len(current_oee_history) > 60:
                current_oee_history.pop(0)
                
            new_row = recent_window[-1].copy()
            new_row[0] = mu_m                                # OEE_Lag_1
            new_row[1] = current_oee_history[-7] if len(current_oee_history) >= 7 else mu_m  # OEE_Lag_7
            new_row[2] = current_oee_history[-14] if len(current_oee_history) >= 14 else mu_m # OEE_Lag_14
            new_row[3] = np.mean(current_oee_history[-7:])   # OEE_Roll_7d
            new_row[4] = np.mean(current_oee_history[-14:])  # OEE_Roll_14d
            new_row[5] = np.mean(current_oee_history[-30:])  # OEE_Roll_30d
            new_row[6] = np.std(current_oee_history[-7:])    # OEE_Std_7d
            
            # Avanzar componentes sinusoidales de calendario
            new_row[-2] = np.sin(2 * np.pi * (m_idx + 1) / 12)
            new_row[-1] = np.cos(2 * np.pi * (m_idx + 1) / 12)
            
            # Desplazar ventana
            recent_window = np.vstack([recent_window[1:], new_row])
            
        multimonth_results[line] = {
            'line': line,
            'horizonte_meses': n_months,
            'trayectoria': line_trajectory
        }
        
    output_data = {
        'metadata': {
            'n_months': n_months,
            'n_samples': n_samples,
            'meses': month_names,
            'metodologia': "Autorregresiva Iterada con Monte Carlo Dropout y Difusión Estocástica de Incertidumbre"
        },
        'lines': multimonth_results
    }
    
    with open(FILE_MULTIMONTH_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Proyección multi-mes guardada exitosamente en: {FILE_MULTIMONTH_RESULTS}")
    return output_data

if __name__ == "__main__":
    run_multimonth_forecast(n_months=6, n_samples=100)
