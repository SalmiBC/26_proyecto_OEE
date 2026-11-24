"""
Cliente de Consulta del Modelo de OEE y Evaluación de Saturación / Riesgo de Plan.
Laboratorios PiSA S.A. de C.V. - Planeación de la Producción.
Cumple con el Criterio 4 de Integración:
- Consume el endpoint REST publicado (no el modelo local).
- Maneja excepciones de red, tiempos de espera (timeout) y respuestas de validación HTTP (422).
- Permite consulta interactiva y CLI solicitando PVOs y Horas Disponibles según requerimiento.
"""

import os
import sys
import argparse
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# URL por defecto configurable vía variable de entorno para entornos AWS / Cloud
DEFAULT_API_URL = os.environ.get("OEE_API_URL", "http://127.0.0.1:8000")

class OEEModelClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 12.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> dict:
        """Verifica conectividad y estado del servicio REST."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except ConnectionError:
            raise ConnectionError(
                f"[ERROR DE CONEXIÓN] No se pudo contactar al servidor en {self.base_url}. "
                "Verifica que la API esté corriendo (ejecuta: python -m uvicorn src.api:app --port 8000)."
            )
        except Timeout:
            raise Timeout(f"[TIMEOUT] La petición a {self.base_url}/health excedió los {self.timeout}s.")
        except HTTPError as e:
            raise RuntimeError(f"[ERROR HTTP {resp.status_code}] {resp.text}")

    def get_lines(self) -> list:
        """Consulta la lista de líneas disponibles en el servidor."""
        try:
            resp = requests.get(f"{self.base_url}/lines", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[!] No se pudo obtener la lista de líneas: {e}")
            return []

    def get_line_specs(self) -> dict:
        """Consulta las especificaciones nominales (velocidad y OEE meta) por línea."""
        try:
            resp = requests.get(f"{self.base_url}/line-specs", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[!] No se pudo obtener especificaciones de líneas: {e}")
            return {}

    def predict_oee(self, linea: str, mc_samples: int = 200) -> dict:
        """
        Envía solicitud POST al endpoint /predict para obtener la distribución
        probabilística del OEE y desglose de paros.
        """
        payload = {"linea": linea, "mc_samples": mc_samples}
        url = f"{self.base_url}/predict"
        
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 422:
                err_detail = resp.json().get("detail", resp.text)
                raise ValueError(f"[DATOS INVÁLIDOS HTTP 422]: {err_detail}")
            resp.raise_for_status()
            return resp.json()
            
        except ConnectionError:
            raise ConnectionError(f"[ERROR DE CONEXIÓN] Servicio en {self.base_url} inaccesible.")
        except Timeout:
            raise Timeout(f"[TIMEOUT] El cálculo estocástico (N={mc_samples}) superó el tiempo límite ({self.timeout}s).")
        except HTTPError as e:
            raise RuntimeError(f"[FALLA DEL SERVIDOR HTTP {resp.status_code}]: {resp.text}")

    def evaluate_risk(self, linea: str, pvos_a_fabricar: float, horas_disponibles: float,
                      oee_meta_pct: float = None, mc_samples: int = 200) -> dict:
        """
        Envía solicitud POST al endpoint /evaluate-risk para contrastar la predicción
        con la meta del plan y calcular saturación e incumplimiento de fabricación.
        """
        payload = {
            "linea": linea,
            "pvos_a_fabricar": pvos_a_fabricar,
            "horas_disponibles": horas_disponibles,
            "mc_samples": mc_samples
        }
        if oee_meta_pct is not None:
            payload["oee_meta_pct"] = oee_meta_pct
            
        url = f"{self.base_url}/evaluate-risk"
        
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 422:
                raise ValueError(f"[DATOS INVÁLIDOS HTTP 422]: {resp.json().get('detail', resp.text)}")
            resp.raise_for_status()
            return resp.json()
        except ConnectionError:
            raise ConnectionError(f"[ERROR DE CONEXIÓN] Servicio en {self.base_url} inaccesible.")
        except Timeout:
            raise Timeout(f"[TIMEOUT] La evaluación de riesgo superó el tiempo límite de {self.timeout}s.")
        except HTTPError as e:
            raise RuntimeError(f"[FALLA DEL SERVIDOR HTTP {resp.status_code}]: {resp.text}")

def format_prediction_report(pred: dict):
    print("\n" + "="*75)
    print(f" RESULTADO DE INFERENCIA ESTOCÁSTICA - LÍNEA: {pred['linea']}")
    print("="*75)
    print(f" • OEE Esperado (Media μ)   : {pred['mu_oee_pct']:.2f}%")
    print(f" • Incertidumbre (Desv Std σ): ±{pred['sigma_oee_pct']:.2f}%")
    print(f" • Intervalo Pesimista (P10) : {pred['q10_oee_pct']:.2f}%")
    print(f" • Mediana Proyectada (P50)  : {pred['q50_oee_pct']:.2f}%")
    print(f" • Intervalo Optimista (P90) : {pred['q90_oee_pct']:.2f}%")
    print("\nDesglose de Horas de Paros No Planeados Proyectados:")
    for paro, hrs in pred['desglose_paros_horas'].items():
        print(f"   - {paro:<28}: {hrs:>6.1f} horas")
    print("="*75)

def format_risk_report(risk: dict):
    incumple = risk['incumple_plan']
    if incumple:
        status_banner = "🔴 INCUMPLE PLAN DE FABRICACIÓN (Saturación Proyectada > 100%)"
        status_color = "🔴 CRÍTICO"
    elif risk['nivel_riesgo'] == "Alerta":
        status_banner = "🟡 CUMPLE PLAN CON ALERTA OPERATIVA (Saturación en Zona de Riesgo)"
        status_color = "🟡 ALERTA"
    else:
        status_banner = "🟢 CUMPLE PLAN DE FABRICACIÓN FACTIBLEMENTE"
        status_color = "🟢 ESTABLE"

    print("\n" + "="*80)
    print(f" {status_banner}")
    print("="*80)
    print(f" DATOS DE LA CONSULTA (LÍNEA: {risk['linea']}):")
    print(f" • Velocidad Nominal     : {risk['velocidad_nominal']:,.1f} PVOs/hora")
    print(f" • Horas Disponibles     : {risk['horas_disponibles']:,.1f} horas")
    print(f" • PVOs a Fabricar       : {risk['pvos_a_fabricar']:,.0f} unidades")
    print(f" • OEE Meta Comprometido : {risk['oee_meta_pct']:.2f}%")
    print(f" • OEE Proyectado (LSTM) : {risk['oee_proyectado_mu_pct']:.2f}% (±{risk['oee_proyectado_sigma_pct']:.2f}%)")
    print(f" • Rango OEE (P10 - P90) : [{risk['q10_oee_pct']:.2f}% - {risk['q90_oee_pct']:.2f}%]")
    print("-" * 80)
    print(" ANÁLISIS DE CAPACIDAD Y SATURACIÓN DE LÍNEA:")
    print(f" • Saturación Teórica (con Meta)  : {risk['saturacion_teorica_pct']:.1f}% ({risk['horas_requeridas_teoricas']:.1f} hrs req.)")
    print(f" • Saturación Proyectada (Real)   : {risk['saturacion_proyectada_pct']:.1f}% ({risk['horas_requeridas_proyectadas']:.1f} hrs req.)")
    print(f" • Banda de Saturación (P10-P90)  : [{risk['saturacion_q10_pct']:.1f}% - {risk['saturacion_q90_pct']:.1f}%]")
    print(f" • Probabilidad Sobresaturación   : {risk['probabilidad_saturacion_pct']:.1f}% P(Saturación > 100%)")
    print("-" * 80)
    if incumple:
        print(" 🚨 CUANTIFICACIÓN DEL INCUMPLIMIENTO:")
        print(f" • % Incumplimiento de Piezas    : {risk['pct_incumplimiento_piezas']:.2f}% de piezas no alcanzables")
        print(f" • % Sobre-Saturación Capacidad   : +{risk['pct_sobresaturacion_capacidad']:.2f}% excedente sobre el 100%")
        print(f" • Piezas Faltantes Estimadas    : {risk['piezas_faltantes']:,.0f} piezas")
        print(f" • Horas Máquina Faltantes       : {risk['horas_faltantes']:,.1f} horas adicionales necesarias")
    else:
        print(" ✅ EVALUACIÓN DE CAPACIDAD:")
        print(f" • La línea cuenta con holgura de capacidad (Holgura: {100.0 - risk['saturacion_proyectada_pct']:.1f}%).")
        print(f" • Piezas Faltantes: 0 unidades.")
    print("-" * 80)
    print(f" Semáforo de Riesgo Global: {status_color} | Prob. Incumplir OEE Meta: {risk['probabilidad_incumplimiento_pct']:.1f}%")
    print("\nDesglose de Horas de Paros No Planeados Proyectados (Causas Raíz):")
    paros_periodo = risk.get('detalle_paros_periodo') or {}
    h_disp = risk['horas_disponibles']
    print(f"   {'Categoría Oficial':<28} | {'Día Típico':<14} | {'Total en ' + str(int(round(h_disp))) + ' hrs':<18}")
    print(f"   {'-'*28}-+-{'-'*14}-+-{'-'*18}")
    for paro, hrs_dia in risk['detalle_paros_no_planeados'].items():
        hrs_tot = paros_periodo.get(paro, hrs_dia * (h_disp / 24.0))
        print(f"   - {paro:<26}: {hrs_dia:>5.1f} h/día    | {hrs_tot:>6.1f} h acumuladas")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente REST para consultar el modelo de OEE y saturación.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"URL base de la API REST (default: {DEFAULT_API_URL} o variable OEE_API_URL)")
    parser.add_argument("--linea", default=None, help="Código de línea a consultar (ej. 'AV3', 'A4')")
    parser.add_argument("--pvos", "--piezas", type=float, default=None, help="PVOs (piezas) a fabricar")
    parser.add_argument("--horas", type=float, default=None, help="Horas disponibles / programadas (máximo 744 h)")
    parser.add_argument("--meta", type=float, default=None, help="OEE Meta programado en porcentaje (opcional)")
    parser.add_argument("--samples", type=int, default=200, help="Muestras Monte Carlo")
    parser.add_argument("--only-predict", action="store_true", help="Solo generar predicción sin riesgo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo con preguntas al usuario")
    
    args = parser.parse_args()
    client = OEEModelClient(base_url=args.url)
    
    print(f"[*] Conectando a la API publicada en: {args.url}...")
    try:
        health = client.check_health()
        print(f"[OK] Servicio activo: {health['service']} (Status: {health['status']})")
    except Exception as e:
        print(f"[ERROR]: {e}")
        sys.exit(1)
        
    line_specs = client.get_line_specs()
    available_lines = sorted(list(line_specs.keys())) if line_specs else client.get_lines()
    
    # Modo interactivo o solicitud de datos si faltan parámetros
    interactive_mode = args.interactive or (args.linea is None and not args.only_predict)
    
    linea = args.linea
    pvos = args.pvos
    horas = args.horas
    meta = args.meta
    samples = args.samples
    
    if interactive_mode:
        print("\n" + "="*60)
        print(" CONSULTA INTERACTIVA DE SATURACIÓN Y RIESGO DE LÍNEA")
        print("="*60)
        print(f"Líneas disponibles ({len(available_lines)}):")
        print(", ".join(available_lines))
        print("-" * 60)
        
        while not linea or linea.upper() not in [l.upper() for l in available_lines]:
            line_in = input("\nIngrese la Línea de producción a consultar (ej. AV3): ").strip().upper()
            if line_in in [l.upper() for l in available_lines]:
                linea = line_in
            else:
                print(f"[!] Línea '{line_in}' no válida. Elija entre las disponibles.")
                
        spec = line_specs.get(linea, {})
        vel_nom = spec.get('velocidad_nominal', 5000.0)
        default_meta = spec.get('oee_meta_pct', 65.0)
        print(f" -> Línea seleccionada: {linea} (Velocidad nominal: {vel_nom:,.1f} pvos/h, Meta estándar: {default_meta:.1f}%)")
        
        if pvos is None:
            while True:
                try:
                    pvos_in = input("¿Cuántos PVOs (piezas) vas a fabricar?: ").strip().replace(',', '')
                    pvos = float(pvos_in)
                    if pvos > 0:
                        break
                    print("[!] Debe ingresar un volumen mayor a cero.")
                except ValueError:
                    print("[!] Ingrese un valor numérico válido.")
                    
        if horas is None:
            while True:
                try:
                    horas_in = input("¿Cuántas horas disponibles de operación vas a considerar? (máx 744 h): ").strip()
                    horas = float(horas_in)
                    if 0 < horas <= 744:
                        break
                    print("[!] Las horas deben estar entre 1 y 744 (máximo 31 días naturales del mes).")
                except ValueError:
                    print("[!] Ingrese un valor numérico válido.")
                    
        if meta is None:
            meta_in = input(f"OEE Meta (%) [Presione Enter para usar estándar {default_meta:.1f}%]: ").strip()
            meta = float(meta_in) if meta_in else default_meta
    else:
        linea = linea or "AV3"
        pvos = pvos or 50000.0
        horas = horas or 600.0
        if horas <= 0 or horas > 744:
            print(f"[!] Error: El parámetro --horas ({horas}) debe estar entre 1 y 744 (31 días naturales del mes).")
            sys.exit(1)
        if meta is None and linea in line_specs:
            meta = line_specs[linea].get('oee_meta_pct', 65.0)
            
    try:
        if args.only_predict:
            print(f"\n[*] Solicitando inferencia estocástica para línea {linea} (N={samples})...")
            pred = client.predict_oee(linea=linea, mc_samples=samples)
            format_prediction_report(pred)
        else:
            print(f"\n[*] Evaluando saturación y riesgo para {linea} ({pvos:,.0f} PVOs en {horas:,.1f} h)...")
            risk = client.evaluate_risk(
                linea=linea,
                pvos_a_fabricar=pvos,
                horas_disponibles=horas,
                oee_meta_pct=meta,
                mc_samples=samples
            )
            format_risk_report(risk)
    except (ValueError, RuntimeError, Timeout, ConnectionError) as err:
        print(f"\n[!] Error en la operación: {err}")
        sys.exit(1)
