"""
Módulo 2: Modelo de Series de Tiempo (LSTM) con Inferencia Bayesiana (Monte Carlo Dropout).
Entrena un modelo multivariado con métricas de OEE y variables extraídas de avisos,
generando proyecciones probabilísticas (media, desv. estándar, cuantiles q10, q50, q90)
y desglose proyectado de paros no planeados.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.config import (
    FILE_MASTER_OEE, MODELS_DIR, DATA_DIR, LSTM_CONFIG,
    UNPLANNED_STOPPAGE_COLS, TRAIN_END_DATE, TEST_START_DATE,
    TEST_END_DATE, FILE_TEST_METRICS, FILE_TEST_COMPARISON
)

# Fijar semillas para reproducibilidad
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(LSTM_CONFIG["random_seed"])

# Columnas de entrada (Features)
FEATURE_COLS = [
    'OEE_Lag_1', 'OEE_Lag_7', 'OEE_Lag_14',
    'OEE_Roll_7d', 'OEE_Roll_14d', 'OEE_Roll_30d', 'OEE_Std_7d',
    'Paros_No_Plan_Roll_7d', 'Paros_No_Plan_Roll_30d',
    'Frecuencia_Falla_Mecanica_7d', 'Frecuencia_Falla_Electrica_7d',
    'Frecuencia_Falla_Ajuste_7d', 'Frecuencia_Falla_Desgaste_7d',
    'Horas_Paro_Equipo_Critico_7d', 'Severidad_Promedio_7d', 'Avisos_Total_7d',
    'Sin_Dia_Semana', 'Cos_Dia_Semana'
]

# Columnas objetivo
TARGET_OEE_COL = 'OEE_Real'
TARGET_STOPPAGE_COLS = [c for c in UNPLANNED_STOPPAGE_COLS]

class TimeseriesDataset(Dataset):
    def __init__(self, X, y_oee, y_stoppage):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_oee = torch.tensor(y_oee, dtype=torch.float32).unsqueeze(-1)
        self.y_stoppage = torch.tensor(y_stoppage, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_oee[idx], self.y_stoppage[idx]

class MultiOutputMCDropoutLSTM(nn.Module):
    """
    Arquitectura LSTM Multicapa con Dropout Estocástico para Inferencia Bayesiana
    (Monte Carlo Dropout).
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_p=0.2, stoppage_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Cabeza para OEE proyectado (rango acotado 0-1)
        self.oee_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Cabeza para horas proyectadas de paros no planeados por categoría (>= 0)
        self.stoppage_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, stoppage_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Tomar la salida del último paso temporal de la ventana
        last_step = lstm_out[:, -1, :]
        last_step_drop = self.dropout(last_step)
        
        pred_oee = self.oee_head(last_step_drop)
        pred_stoppages = self.stoppage_head(last_step_drop)
        return pred_oee, pred_stoppages

def create_sequences(df: pd.DataFrame, seq_len: int = 14, scaler: StandardScaler = None, is_train: bool = True):
    """
    Crea ventanas deslizantes (X) y objetivos (y) por línea respetando la partición temporal estricta:
    - is_train=True: Datos de 2024 hasta Enero 2026 (<= 2026-01-31). Ajusta el StandardScaler.
    - is_train=False: Datos de Febrero a Septiembre 2026 (>= 2026-02-01). Aplica el scaler existente sin data leakage.
    """
    X_list, y_oee_list, y_stop_list = [], [], []
    dates_list, lines_list = [], []
    
    df_clean = df.copy()
    df_clean['OEE_Target'] = df_clean['OEE_Real'].fillna(df_clean['OEE_Roll_7d']).fillna(0.50).clip(0.05, 0.95)
    
    if is_train:
        df_clean = df_clean[df_clean['Fecha'] <= TRAIN_END_DATE].copy()
        scaler = StandardScaler()
        feature_data = df_clean[FEATURE_COLS].fillna(0.0).values
        scaled_features = scaler.fit_transform(feature_data)
        for idx, col in enumerate(FEATURE_COLS):
            df_clean[f"scaled_{col}"] = scaled_features[:, idx]
    else:
        if scaler is None:
            raise ValueError("Se requiere un scaler ajustado para generar secuencias de prueba.")
        # Incluir días previos a TEST_START_DATE para que los primeros días de febrero tengan ventana completa
        start_dt = pd.to_datetime(TEST_START_DATE)
        window_start = start_dt - pd.Timedelta(days=seq_len + 5)
        df_clean = df_clean[(df_clean['Fecha'] >= window_start) & (df_clean['Fecha'] <= TEST_END_DATE)].copy()
        feature_data = df_clean[FEATURE_COLS].fillna(0.0).values
        scaled_features = scaler.transform(feature_data)
        for idx, col in enumerate(FEATURE_COLS):
            df_clean[f"scaled_{col}"] = scaled_features[:, idx]
            
    scaled_col_names = [f"scaled_{col}" for col in FEATURE_COLS]
    test_start_ts = pd.to_datetime(TEST_START_DATE)
    
    for line, grp in df_clean.groupby('Linea'):
        grp = grp.sort_values('Fecha').reset_index(drop=True)
        if len(grp) <= seq_len:
            continue
            
        feat_vals = grp[scaled_col_names].values
        target_oee = grp['OEE_Target'].values
        target_stop = grp[TARGET_STOPPAGE_COLS].fillna(0.0).values
        dates = grp['Fecha'].values
        
        for i in range(len(grp) - seq_len):
            target_date = pd.to_datetime(dates[i + seq_len])
            if not is_train and target_date < test_start_ts:
                continue
            X_list.append(feat_vals[i : i + seq_len])
            y_oee_list.append(target_oee[i + seq_len])
            y_stop_list.append(target_stop[i + seq_len])
            dates_list.append(target_date)
            lines_list.append(line)
            
    X = np.array(X_list)
    y_oee = np.array(y_oee_list)
    y_stoppage = np.array(y_stop_list)
    
    if is_train:
        return X, y_oee, y_stoppage, scaler
    else:
        return X, y_oee, y_stoppage, dates_list, lines_list

def train_lstm_model():
    """
    Entrena la red LSTM únicamente con datos de 2024 a Enero 2026.
    Evalúa automáticamente contra el conjunto de prueba (Febrero a Septiembre 2026).
    """
    print(f"[*] Iniciando entrenamiento del modelo LSTM con datos hasta {TRAIN_END_DATE}...")
    df = pd.read_parquet(FILE_MASTER_OEE)
    
    seq_len = LSTM_CONFIG["sequence_length"]
    X, y_oee, y_stoppage, scaler = create_sequences(df, seq_len=seq_len, is_train=True)
    
    print(f"[*] Muestras de entrenamiento (2024 a Enero 2026): {len(X)} secuencias")
    
    # División Train / Val cronológica dentro del periodo de entrenamiento (85% train, 15% val)
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_oee_train, y_oee_val = y_oee[:split_idx], y_oee[split_idx:]
    y_stop_train, y_stop_val = y_stoppage[:split_idx], y_stoppage[split_idx:]
    
    train_ds = TimeseriesDataset(X_train, y_oee_train, y_stop_train)
    val_ds = TimeseriesDataset(X_val, y_oee_val, y_stop_val)
    
    train_loader = DataLoader(train_ds, batch_size=LSTM_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=LSTM_CONFIG["batch_size"], shuffle=False)
    
    # Inicializar modelo
    model = MultiOutputMCDropoutLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=LSTM_CONFIG["hidden_size"],
        num_layers=LSTM_CONFIG["num_layers"],
        dropout_p=LSTM_CONFIG["dropout"],
        stoppage_dim=len(TARGET_STOPPAGE_COLS)
    )
    
    criterion_oee = nn.MSELoss()
    criterion_stop = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    model_save_path = MODELS_DIR / "best_lstm_model.pth"
    
    print(f"[*] Entrenando durante {LSTM_CONFIG['epochs']} épocas...")
    for epoch in range(1, LSTM_CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y_oee, batch_y_stop in train_loader:
            optimizer.zero_grad()
            pred_oee, pred_stop = model(batch_x)
            loss_oee = criterion_oee(pred_oee, batch_y_oee)
            loss_stop = criterion_stop(pred_stop, batch_y_stop) * 0.1
            loss = loss_oee + loss_stop
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_oee, batch_y_stop in val_loader:
                pred_oee, pred_stop = model(batch_x)
                loss_oee = criterion_oee(pred_oee, batch_y_oee)
                loss_stop = criterion_stop(pred_stop, batch_y_stop) * 0.1
                val_loss += (loss_oee + loss_stop).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': LSTM_CONFIG,
                'feature_cols': FEATURE_COLS,
                'stoppage_cols': TARGET_STOPPAGE_COLS
            }, model_save_path)
            
        if epoch % 10 == 0 or epoch == LSTM_CONFIG["epochs"]:
            print(f"  Epoca {epoch:02d}/{LSTM_CONFIG['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    print(f"[+] Modelo entrenado y guardado en: {model_save_path} (Mejor Val Loss: {best_loss:.4f})")
    
    # Guardar escalador ajustado exclusivamente con datos de entrenamiento
    import joblib
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    
    # Cargar los mejores pesos para evaluar el conjunto de prueba
    best_checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Ejecutar evaluación comparativa fuera de muestra (Feb a Sep 2026)
    evaluate_test_comparative(model, scaler, df)
    
    return model, scaler

def evaluate_test_comparative(model=None, scaler=None, df_master: pd.DataFrame = None) -> dict:
    """
    Evalúa el modelo predictivo LSTM en los datos de prueba (Febrero a Septiembre 2026),
    comparando las predicciones contra los valores reales observados en planta.
    Calcula MAE, RMSE, MAPE global y por línea, y guarda resultados en JSON y Parquet.
    """
    print(f"[*] Evaluando modelo en conjunto de prueba / comparativa ({TEST_START_DATE} a {TEST_END_DATE})...")
    if df_master is None:
        df_master = pd.read_parquet(FILE_MASTER_OEE)
        
    if model is None or scaler is None:
        import joblib
        model_path = MODELS_DIR / "best_lstm_model.pth"
        scaler_path = MODELS_DIR / "scaler.joblib"
        checkpoint = torch.load(model_path, weights_only=False)
        model = MultiOutputMCDropoutLSTM(
            input_dim=len(checkpoint['feature_cols']),
            hidden_dim=checkpoint['config']["hidden_size"],
            num_layers=checkpoint['config']["num_layers"],
            dropout_p=checkpoint['config']["dropout"],
            stoppage_dim=len(checkpoint['stoppage_cols'])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = joblib.load(scaler_path)
        
    seq_len = LSTM_CONFIG["sequence_length"]
    X_test, y_oee_test, y_stop_test, dates_test, lines_test = create_sequences(
        df_master, seq_len=seq_len, scaler=scaler, is_train=False
    )
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds_oee, preds_stop = model(x_tensor)
        preds_oee = preds_oee.squeeze(-1).numpy()
        preds_stop = preds_stop.numpy()
        
    df_eval = pd.DataFrame({
        'Fecha': pd.to_datetime(dates_test),
        'Linea': lines_test,
        'OEE_Real': y_oee_test,
        'OEE_Predicho': preds_oee,
        'Error_Absoluto': np.abs(y_oee_test - preds_oee),
        'Error_Pct': np.abs(y_oee_test - preds_oee) / np.maximum(y_oee_test, 0.05) * 100.0
    })
    
    # Métricas Globales
    mae_global = float(df_eval['Error_Absoluto'].mean())
    rmse_global = float(np.sqrt(np.mean((y_oee_test - preds_oee) ** 2)))
    mape_global = float(df_eval['Error_Pct'].mean())
    
    # Métricas por Línea
    line_metrics = {}
    for line, grp in df_eval.groupby('Linea'):
        mae_line = float(grp['Error_Absoluto'].mean())
        rmse_line = float(np.sqrt(np.mean((grp['OEE_Real'] - grp['OEE_Predicho']) ** 2)))
        mape_line = float(grp['Error_Pct'].mean())
        corr = float(grp[['OEE_Real', 'OEE_Predicho']].corr().iloc[0, 1]) if len(grp) > 5 else 0.0
        line_metrics[line] = {
            'registros_test': len(grp),
            'oee_real_promedio': round(float(grp['OEE_Real'].mean() * 100), 2),
            'oee_predicho_promedio': round(float(grp['OEE_Predicho'].mean() * 100), 2),
            'mae_pct': round(mae_line * 100, 2),
            'rmse_pct': round(rmse_line * 100, 2),
            'mape_pct': round(mape_line, 2),
            'correlacion': round(0.0 if np.isnan(corr) else corr, 3)
        }
        
    # MAE en Paros No Planeados (Horas)
    stop_mae = {}
    for idx, col in enumerate(TARGET_STOPPAGE_COLS):
        stop_mae[col] = round(float(np.mean(np.abs(y_stop_test[:, idx] - preds_stop[:, idx]))), 3)
        
    metrics_summary = {
        'periodo_entrenamiento': f"Mayo 2024 a {TRAIN_END_DATE}",
        'periodo_prueba_comparativa': f"{TEST_START_DATE} a {TEST_END_DATE}",
        'muestras_evaluadas': len(df_eval),
        'metricas_globales': {
            'mae_oee_pct': round(mae_global * 100, 2),
            'rmse_oee_pct': round(rmse_global * 100, 2),
            'mape_oee_pct': round(mape_global, 2),
        },
        'mae_paros_no_planeados_horas': stop_mae,
        'metricas_por_linea': line_metrics
    }
    
    with open(FILE_TEST_METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
    df_eval.to_parquet(FILE_TEST_COMPARISON, index=False)
    
    print(f"[+] Evaluación comparativa completada y guardada en {FILE_TEST_METRICS}")
    print(f"    Muestras de prueba: {len(df_eval)} registros diarios (Feb-Sep 2026)")
    print(f"    MAE Global OEE:     {metrics_summary['metricas_globales']['mae_oee_pct']}%")
    print(f"    RMSE Global OEE:    {metrics_summary['metricas_globales']['rmse_oee_pct']}%")
    print(f"    MAPE Global OEE:    {metrics_summary['metricas_globales']['mape_oee_pct']}%")
    return metrics_summary

def predict_line_probabilistic(line: str, model, scaler, df_master: pd.DataFrame, n_samples: int = 200):
    """
    Realiza inferencia probabilística usando Monte Carlo Dropout para una línea específica.
    Retorna la distribución posterior de OEE proyectado y desglose de horas de paro.
    """
    seq_len = LSTM_CONFIG["sequence_length"]
    df_line = df_master[df_master['Linea'] == line].sort_values('Fecha').copy()
    
    # Si la línea no tiene historial suficiente (ej. AV7), usar el prior de su familia (AV)
    if len(df_line) < seq_len:
        print(f"[!] Línea {line} con historial limitado. Aplicando prior Bayesiano de familia...")
        family_lines = [l for l in ['AV1', 'AV2', 'AV3', 'AV6'] if l in df_master['Linea'].unique()]
        df_line = df_master[df_master['Linea'].isin(family_lines)].groupby('Fecha').mean(numeric_only=True).reset_index()
        
    recent_features = df_line[FEATURE_COLS].tail(seq_len).fillna(0.0).values
    if len(recent_features) < seq_len:
        # Pad si es necesario
        pad = np.zeros((seq_len - len(recent_features), len(FEATURE_COLS)))
        recent_features = np.vstack([pad, recent_features])
        
    scaled_seq = scaler.transform(recent_features)
    x_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)  # Shape (1, seq_len, num_feat)
    
    # Inferencia estocástica con Monte Carlo Dropout
    # Al activar model.train(), las capas de dropout permanecen activas
    model.train()
    
    oee_samples = []
    stoppage_samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred_oee, pred_stop = model(x_tensor)
            oee_val = pred_oee.item()
            stop_vals = pred_stop.squeeze(0).numpy()
            
            oee_samples.append(oee_val)
            stoppage_samples.append(stop_vals)
            
    oee_samples = np.array(oee_samples)
    stoppage_samples = np.array(stoppage_samples)  # Shape (n_samples, 7)
    
    # Estadísticas probabilísticas de OEE
    mu_oee = float(np.mean(oee_samples))
    sigma_oee = float(np.std(oee_samples))
    q10_oee = float(np.percentile(oee_samples, 10))
    q50_oee = float(np.percentile(oee_samples, 50))
    q90_oee = float(np.percentile(oee_samples, 90))
    min_oee = float(np.min(oee_samples))
    max_oee = float(np.max(oee_samples))
    
    # Desglose de horas de paros no planeados proyectados
    mean_stoppages = np.mean(stoppage_samples, axis=0)
    stoppage_breakdown = {}
    for idx, col in enumerate(TARGET_STOPPAGE_COLS):
        stoppage_breakdown[col] = float(mean_stoppages[idx])
        
    return {
        'line': line,
        'mu_oee': round(mu_oee, 4),
        'sigma_oee': round(sigma_oee, 4),
        'q10_oee': round(q10_oee, 4),
        'q50_oee': round(q50_oee, 4),
        'q90_oee': round(q90_oee, 4),
        'min_oee': round(min_oee, 4),
        'max_oee': round(max_oee, 4),
        'oee_samples': oee_samples.tolist(),
        'stoppage_breakdown': stoppage_breakdown
    }

if __name__ == "__main__":
    model, scaler = train_lstm_model()
    df_m = pd.read_parquet(FILE_MASTER_OEE)
    sample_pred = predict_line_probabilistic('A4', model, scaler, df_m)
    print("\n--- Predicción Probabilística A4 (Monte Carlo Dropout) ---")
    print(f"Media OEE: {sample_pred['mu_oee']*100:.1f}% | Desv Std: {sample_pred['sigma_oee']*100:.2f}% | Rango P10-P90: [{sample_pred['q10_oee']*100:.1f}% - {sample_pred['q90_oee']*100:.1f}%]")
    print("Desglose de paros proyectados:", sample_pred['stoppage_breakdown'])
