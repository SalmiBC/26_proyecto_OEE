"""
========================================================================================
SISTEMA HÍBRIDO DE PREDICCIÓN DE OEE, ESTIMACIÓN DE RIESGO DE PLAN Y DIAGNÓSTICO
(LSTM + LLM) - GRUPO PIÑA / LABORATORIOS PISA S.A. DE C.V.
========================================================================================
Punto de entrada principal (CLI / Pipeline Runner)
"""
import sys
import argparse
from pathlib import Path

# Agregar raíz al sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def run_nlp_pipeline():
    print("\n" + "="*80)
    print(">>> [FASE 1/5] PROCESAMIENTO NLP Y TAXONOMIA DE BITACORAS (avisos.xlsx)")
    print("="*80)
    from src.nlp_avisos import process_all_avisos, generate_daily_maintenance_features
    df_avisos = process_all_avisos()
    df_daily = generate_daily_maintenance_features(df_avisos)
    print(f"[OK] NLP Completado. Avisos: {len(df_avisos)} | Vectores Diarios: {len(df_daily)}")

def run_data_pipeline():
    print("\n" + "="*80)
    print(">>> [FASE 2/5] INGESTION, CALCULO DE OEE REAL Y FEATURE ENGINEERING")
    print("="*80)
    from src.data_pipeline import build_master_dataset
    df_master, df_plan_summary = build_master_dataset()
    print(f"[OK] Master Dataset consolidado: {df_master.shape[0]} registros x {df_master.shape[1]} columnas.")

def run_lstm_training(epochs=45):
    print("\n" + "="*80)
    print(f">>> [FASE 3/5] ENTRENAMIENTO DE RED NEURONAL LSTM MULTICAPA ({epochs} EPOCHS)")
    print("="*80)
    from src.lstm_model import train_lstm_pipeline
    model, scaler, results = train_lstm_pipeline(epochs=epochs)
    print(f"[OK] Modelo LSTM entrenado y guardado con exito. Val Loss: {results['best_val_loss']:.4f}")

def run_risk_engine(mc_samples=200):
    print("\n" + "="*80)
    print(f">>> [FASE 4/5] MOTOR DE RIESGO DE PLAN Y MONTE CARLO DROPOUT (N={mc_samples})")
    print("="*80)
    from src.risk_engine import run_risk_engine as execute_risk
    results = execute_risk()
    global_kpis = results['global_kpis']
    print(f"[OK] Evaluacion de riesgo completada para {len(results['lines'])} lineas.")
    print(f"     OEE Planeado: {global_kpis['oee_promedio_planeado']:.1f}% | OEE Proyectado: {global_kpis['oee_promedio_proyectado']:.1f}%")
    print(f"     Horas en Riesgo: {global_kpis['horas_totales_en_riesgo']:,.1f} hrs | Vol. Faltante: {global_kpis['volumen_total_en_riesgo']:,.0f} pzas")
    print(f"     Lineas Criticas (P > 60%): {global_kpis['lineas_criticas_count']} | Alerta (20-60%): {global_kpis['lineas_alerta_count']} | Estables (<20%): {global_kpis['lineas_estables_count']}")

def run_prescriptive_and_dashboard():
    print("\n" + "="*80)
    print(">>> [FASE 5/5] MOTOR PRESCRIPTIVO LLM Y GENERADOR DE DASHBOARD HTML")
    print("="*80)
    from src.prescriptive_engine import generate_prescriptive_narratives
    from src.generate_dashboard import build_html_dashboard
    
    print("[*] Generando diagnosticos prescriptivos por linea...")
    generate_prescriptive_narratives()
    
    print("[*] Compilando tablero interactivo HTML...")
    html_path = build_html_dashboard()
    print(f"[OK] Dashboard HTML final generado exitosamente en:\n     {html_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Sistema Hibrido de Prediccion de OEE, Riesgo y Diagnostico (LSTM + LLM) - Laboratorios PiSA"
    )
    parser.add_argument("--all", action="store_true", help="Ejecutar todo el pipeline de punta a punta (NLP + Datos + LSTM + Riesgo + Dashboard)")
    parser.add_argument("--skip-train", action="store_true", help="Omitir reentrenamiento del LSTM y usar el modelo ya entrenado en models/best_lstm_model.pth")
    parser.add_argument("--dashboard-only", action="store_true", help="Generar unicamente el dashboard HTML utilizando los resultados existentes")
    parser.add_argument("--epochs", type=int, default=45, help="Numero de epocas para el entrenamiento de la red LSTM (default: 45)")
    parser.add_argument("--mc-samples", type=int, default=200, help="Numero de muestras Monte Carlo Dropout para inferencia estocastica (default: 200)")
    
    args = parser.parse_args()
    
    if args.dashboard_only:
        run_prescriptive_and_dashboard()
        return

    if args.skip_train:
        run_risk_engine(mc_samples=args.mc_samples)
        run_prescriptive_and_dashboard()
        return

    # Si se pasa --all o se ejecuta sin argumentos por defecto:
    run_nlp_pipeline()
    run_data_pipeline()
    run_lstm_training(epochs=args.epochs)
    run_risk_engine(mc_samples=args.mc_samples)
    run_prescriptive_and_dashboard()

if __name__ == "__main__":
    main()
