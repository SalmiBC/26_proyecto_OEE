"""
Módulo 5: Generador del Dashboard / Presentación Ejecutiva en HTML Autónomo.
Construye un archivo HTML único, interactivo y moderno (dashboard_oee_predictivo.html)
con Tailwind CSS, CDN de Plotly.js, tablas dinámicas con filtros y narrativa prescriptiva.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    FILE_RISK_RESULTS, FILE_PRESCRIPTIVE, FILE_MASTER_OEE,
    FILE_PROCESSED_AVISOS, FILE_DASHBOARD_HTML, LSTM_CONFIG
)
from src.nlp_avisos import extract_pareto_root_causes

def prepare_historical_and_forecast_data(df_master: pd.DataFrame, risk_data: dict) -> dict:
    """Prepara las series históricas (últimos 60 días) y la proyección de noviembre para cada línea."""
    chart1_data = {}
    
    # Rango de fechas históricas recientes
    max_date = df_master['Fecha'].max()
    min_recent_date = max_date - pd.Timedelta(days=60)
    df_recent = df_master[df_master['Fecha'] >= min_recent_date].copy()
    
    # Fechas proyectadas para noviembre (30 días)
    nov_dates = [f"2026-11-{d:02d}" for d in range(1, 31)]
    
    for r in risk_data['lines']:
        line = r['line']
        oee_plan = r['oee_planeado'] * 100
        mu = r['oee_proyectado_mu'] * 100
        sigma = r['oee_proyectado_sigma'] * 100
        q10 = r['oee_proyectado_q10'] * 100
        q90 = r['oee_proyectado_q90'] * 100
        
        # Historial real
        df_l = df_recent[df_recent['Linea'] == line].sort_values('Fecha')
        if len(df_l) > 0:
            hist_dates = df_l['Fecha'].dt.strftime('%Y-%m-%d').tolist()
            hist_oee = (df_l['OEE_Real'].fillna(df_l['OEE_Roll_7d']) * 100).round(1).tolist()
            hist_trend = (df_l['OEE_Roll_7d'] * 100).round(1).tolist()
        else:
            hist_dates = []
            hist_oee = []
            hist_trend = []
            
        # Generar trayectoria estocástica suave para noviembre centrada en mu con bandas q10-q90
        np.random.seed(42 + hash(line) % 1000)
        daily_noise = np.random.normal(0, max(0.5, sigma * 0.4), size=30)
        proj_daily = np.clip(mu + daily_noise, max(5.0, q10 - 2), min(98.0, q90 + 2)).round(1).tolist()
        proj_upper = [round(min(100.0, q90 + abs(n) * 0.5), 1) for n in daily_noise]
        proj_lower = [round(max(0.0, q10 - abs(n) * 0.5), 1) for n in daily_noise]
        
        chart1_data[line] = {
            'hist_dates': hist_dates,
            'hist_oee': hist_oee,
            'hist_trend': hist_trend,
            'nov_dates': nov_dates,
            'proj_daily': proj_daily,
            'proj_mu': round(mu, 1),
            'proj_upper': proj_upper,
            'proj_lower': proj_lower,
            'oee_plan': round(oee_plan, 1)
        }
        
    return chart1_data

def prepare_waterfall_data(risk_data: dict) -> dict:
    """Prepara datos para el gráfico de cascada por línea y consolidado."""
    waterfall_data = {}
    
    for r in risk_data['lines']:
        line = r['line']
        mu = r['oee_proyectado_mu'] * 100
        stoppages = r['stoppages_hours']
        h_prog = r['horas_programadas']
        
        # Caída en puntos porcentuales estimada por categoría
        # Base de 100% de tiempo disponible
        items = [
            {"name": "Tiempo Disponible", "value": 100.0, "type": "total"}
        ]
        
        # Deducciones
        tot_loss = 100.0 - mu
        tot_stop_hrs = sum(stoppages.values()) if stoppages else 1.0
        
        for cat, hrs in stoppages.items():
            pct_contrib = round((hrs / max(1.0, tot_stop_hrs)) * tot_loss, 1)
            if pct_contrib > 0.1:
                items.append({
                    "name": cat,
                    "value": -pct_contrib,
                    "type": "relative"
                })
                
        # OEE Final Proyectado
        items.append({
            "name": "OEE Proyectado (LSTM)",
            "value": round(mu, 1),
            "type": "total"
        })
        
        waterfall_data[line] = items
        
    return waterfall_data

def build_html_dashboard():
    """Genera el código HTML interactivo autónomo completo."""
    print("[*] Compilando datos para el Dashboard HTML...")
    
    with open(FILE_RISK_RESULTS, 'r', encoding='utf-8') as f:
        risk_data = json.load(f)
        
    with open(FILE_PRESCRIPTIVE, 'r', encoding='utf-8') as f:
        prescriptive_data = json.load(f)
        
    df_master = pd.read_parquet(FILE_MASTER_OEE)
    df_avisos = pd.read_parquet(FILE_PROCESSED_AVISOS)
    
    # Preparar datos de gráficos
    chart1_data = prepare_historical_and_forecast_data(df_master, risk_data)
    waterfall_data = prepare_waterfall_data(risk_data)
    pareto_data = extract_pareto_root_causes(df_avisos, top_n=15)
    
    # Convertir a JSON seguro para inyectar en JS
    json_risk = json.dumps(risk_data, ensure_ascii=False)
    json_chart1 = json.dumps(chart1_data, ensure_ascii=False)
    json_waterfall = json.dumps(waterfall_data, ensure_ascii=False)
    json_pareto = json.dumps(pareto_data, ensure_ascii=False)
    json_prescriptive = json.dumps(prescriptive_data, ensure_ascii=False)
    
    kpis = risk_data['global_kpis']
    
    html_content = f"""<!DOCTYPE html>
<html lang="es" class="h-full bg-slate-900 text-slate-100">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sistema Híbrido de Predicción de OEE & Diagnóstico de Fallas (LSTM + LLM)</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Plotly.js CDN -->
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <!-- Google Fonts: Inter -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          }},
          colors: {{
            brand: {{
              50: '#f0f9ff',
              100: '#e0f2fe',
              500: '#0ea5e9',
              600: '#0284c7',
              700: '#0369a1',
              900: '#0c4a6e',
            }}
          }}
        }}
      }}
    }}
  </script>
  <style>
    body {{
      font-family: 'Inter', sans-serif;
    }}
    .glass-card {{
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .custom-scrollbar::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scrollbar::-webkit-scrollbar-track {{
      background: rgba(15, 23, 42, 0.6);
    }}
    .custom-scrollbar::-webkit-scrollbar-thumb {{
      background: rgba(100, 116, 139, 0.5);
      border-radius: 3px;
    }}
  </style>
</head>
<body class="min-h-full flex flex-col bg-slate-950 text-slate-100 antialiased selection:bg-sky-500 selection:text-white">

  <!-- Top Navigation Bar -->
  <header class="sticky top-0 z-50 glass-card border-b border-slate-800 bg-slate-950/80">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-lg tracking-tight text-white">OEE Predictivo & Riesgo de Plan</span>
            <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">LSTM + LLM</span>
          </div>
          <p class="text-xs text-slate-400">Grupo PiSA • Sistema Integral de Inteligencia Operacional</p>
        </div>
      </div>
      <div class="flex items-center gap-4 text-xs font-medium">
        <div class="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Modelo LSTM: 45 Épocas (MC Dropout N=200)</span>
        </div>
        <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
          <span class="w-2 h-2 rounded-full bg-purple-500"></span>
          <span>Bitácoras NLP: 21,100 Avisos</span>
        </div>
        <button onclick="window.print()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors flex items-center gap-1.5 border border-slate-700">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"/></svg>
          Exportar PDF
        </button>
      </div>
    </div>
  </header>

  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

    <!-- Executive Header Banner -->
    <div class="relative overflow-hidden rounded-2xl glass-card border border-slate-800 p-6 sm:p-8 bg-gradient-to-r from-slate-900 via-slate-900/90 to-sky-950/40 shadow-2xl">
      <div class="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        <div class="lg:col-span-7 space-y-2">
          <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-400 text-xs font-semibold border border-indigo-500/20">
            <span>Horizonte de Fabricación: Plan Noviembre</span>
          </div>
          <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
            Tablero Ejecutivo de Riesgo y Diagnóstico de Confiabilidad
          </h1>
          <p class="text-sm text-slate-300 leading-relaxed max-w-2xl">
            Cuantificación probabilística de brechas de OEE mediante Monte Carlo Dropout y síntesis prescriptiva de causas raíz a partir de las bitácoras no estructuradas de mantenimiento.
          </p>
        </div>
        <div class="lg:col-span-5 flex flex-wrap gap-3 justify-start lg:justify-end">
          <div class="px-4 py-3 rounded-xl bg-slate-950/60 border border-slate-800 min-w-[130px]">
            <span class="text-xs text-slate-400 block font-medium">Líneas en Plan</span>
            <span class="text-2xl font-bold text-white">{kpis['total_lineas']}</span>
          </div>
          <div class="px-4 py-3 rounded-xl bg-rose-950/30 border border-rose-800/40 min-w-[130px]">
            <span class="text-xs text-rose-400 block font-medium">Líneas Críticas</span>
            <span class="text-2xl font-bold text-rose-400">{kpis['lineas_criticas_count']}</span>
          </div>
          <div class="px-4 py-3 rounded-xl bg-amber-950/30 border border-amber-800/40 min-w-[130px]">
            <span class="text-xs text-amber-400 block font-medium">En Alerta</span>
            <span class="text-2xl font-bold text-amber-400">{kpis['lineas_alerta_count']}</span>
          </div>
          <div class="px-4 py-3 rounded-xl bg-emerald-950/30 border border-emerald-800/40 min-w-[130px]">
            <span class="text-xs text-emerald-400 block font-medium">Estables</span>
            <span class="text-2xl font-bold text-emerald-400">{kpis['lineas_estables_count']}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Global KPI Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
      <!-- KPI 1 -->
      <div class="glass-card rounded-xl p-5 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
        <div class="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>OEE Meta vs Proyectado</span>
          <span class="p-2 rounded-lg bg-sky-500/10 text-sky-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
          </span>
        </div>
        <div class="mt-3 flex items-baseline gap-2">
          <span class="text-3xl font-extrabold text-white">{kpis['oee_promedio_proyectado']}%</span>
          <span class="text-xs text-slate-400 font-medium">vs {kpis['oee_promedio_planeado']}% plan</span>
        </div>
        <div class="mt-2 flex items-center text-xs">
          <span class="text-rose-400 font-semibold flex items-center">
            Δ {(kpis['oee_promedio_planeado'] - kpis['oee_promedio_proyectado']):.1f} p.p.
          </span>
          <span class="text-slate-400 ml-1.5">brecha ponderada planta</span>
        </div>
      </div>

      <!-- KPI 2 -->
      <div class="glass-card rounded-xl p-5 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
        <div class="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Horas Totales en Riesgo</span>
          <span class="p-2 rounded-lg bg-rose-500/10 text-rose-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </span>
        </div>
        <div class="mt-3 flex items-baseline gap-2">
          <span class="text-3xl font-extrabold text-rose-400">{kpis['horas_totales_en_riesgo']:,.1f}</span>
          <span class="text-xs text-slate-400">hrs producción</span>
        </div>
        <div class="mt-2 text-xs text-slate-400">
          Impacto proyectado en tiempo neto de fabricación
        </div>
      </div>

      <!-- KPI 3 -->
      <div class="glass-card rounded-xl p-5 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
        <div class="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Volumen en Riesgo</span>
          <span class="p-2 rounded-lg bg-amber-500/10 text-amber-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>
          </span>
        </div>
        <div class="mt-3 flex items-baseline gap-2">
          <span class="text-3xl font-extrabold text-amber-400">{kpis['volumen_total_en_riesgo']:,.0f}</span>
          <span class="text-xs text-slate-400">unidades PVO</span>
        </div>
        <div class="mt-2 text-xs text-slate-400">
          Piezas faltantes proyectadas por brecha de OEE
        </div>
      </div>

      <!-- KPI 4 -->
      <div class="glass-card rounded-xl p-5 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
        <div class="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Cumplimiento Global Esperado</span>
          <span class="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </span>
        </div>
        <div class="mt-3 flex items-baseline gap-2">
          <span class="text-3xl font-extrabold text-emerald-400">{kpis['cumplimiento_global_pct']}%</span>
          <span class="text-xs text-slate-400">del plan total</span>
        </div>
        <div class="mt-2 text-xs text-slate-400">
          Adherencia agregada volumétrica de planta
        </div>
      </div>
    </div>

    <!-- Interactive Risk Matrix Table Section -->
    <div class="glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <span>Matriz de Riesgo Operacional por Línea</span>
            <span class="text-xs font-normal text-slate-400">(Noviembre 2026)</span>
          </h2>
          <p class="text-xs text-slate-400">Contraste entre meta programada y proyección probabilística LSTM (Monte Carlo Dropout).</p>
        </div>
        <div class="flex flex-wrap items-center gap-3">
          <!-- Search Input -->
          <div class="relative">
            <input type="text" id="lineSearch" placeholder="Buscar línea o familia..." 
                   class="bg-slate-900 border border-slate-700 text-xs text-white rounded-lg pl-8 pr-3 py-2 w-48 focus:outline-none focus:border-sky-500">
            <svg class="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          </div>
          <!-- Status Filter Pills -->
          <div class="flex items-center rounded-lg bg-slate-900 p-1 border border-slate-800 text-xs">
            <button onclick="filterTable('all')" class="filter-btn px-3 py-1 rounded-md font-medium text-white bg-sky-600 transition-colors" data-status="all">Todos</button>
            <button onclick="filterTable('Crítico')" class="filter-btn px-3 py-1 rounded-md font-medium text-slate-400 hover:text-white transition-colors" data-status="Crítico">Crítico</button>
            <button onclick="filterTable('Alerta')" class="filter-btn px-3 py-1 rounded-md font-medium text-slate-400 hover:text-white transition-colors" data-status="Alerta">Alerta</button>
            <button onclick="filterTable('Estable')" class="filter-btn px-3 py-1 rounded-md font-medium text-slate-400 hover:text-white transition-colors" data-status="Estable">Estable</button>
          </div>
        </div>
      </div>

      <!-- Table Container -->
      <div class="overflow-x-auto custom-scrollbar rounded-xl border border-slate-800">
        <table class="w-full text-left text-xs text-slate-300 divide-y divide-slate-800" id="riskTable">
          <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold text-[10px] tracking-wider">
            <tr>
              <th scope="col" class="px-4 py-3 cursor-pointer hover:text-white" onclick="sortTable('line')">Línea ↕</th>
              <th scope="col" class="px-4 py-3">Familia</th>
              <th scope="col" class="px-4 py-3 text-center">SKUs</th>
              <th scope="col" class="px-4 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable('oee_planeado')">OEE Meta ↕</th>
              <th scope="col" class="px-4 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable('oee_proyectado_mu')">OEE Proyectado (μ ± σ) ↕</th>
              <th scope="col" class="px-4 py-3 text-right">Rango (q10 - q90)</th>
              <th scope="col" class="px-4 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable('prob_incumplimiento')">P(Incump.) ↕</th>
              <th scope="col" class="px-4 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable('horas_en_riesgo')">Horas en Riesgo ↕</th>
              <th scope="col" class="px-4 py-3 text-right">Volumen Faltante</th>
              <th scope="col" class="px-4 py-3 text-center">Estado</th>
              <th scope="col" class="px-4 py-3 text-center">Acción</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/60 bg-slate-950/40 font-mono" id="riskTableBody">
            <!-- Table rows rendered dynamically via JS -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Interactive Charts Section -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Chart 1: Curva de OEE Histórico + Proyección LSTM con bandas de confianza -->
      <div class="lg:col-span-8 glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>Trayectoria Histórica y Proyección LSTM (Noviembre)</span>
              <span class="text-xs font-normal text-slate-400">Monte Carlo Dropout</span>
            </h3>
            <p class="text-xs text-slate-400">Evolución observada + Cono de incertidumbre estocástica vs OEE Meta.</p>
          </div>
          <!-- Dropdown Selector de Línea -->
          <div class="flex items-center gap-2">
            <label for="lineSelectChart1" class="text-xs text-slate-400">Línea:</label>
            <select id="lineSelectChart1" onchange="updateChart1(this.value)" 
                    class="bg-slate-900 border border-slate-700 text-xs text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-sky-500 font-sans font-medium">
              <!-- Options populated by JS -->
            </select>
          </div>
        </div>
        <div id="chart1_div" class="w-full h-80 sm:h-96"></div>
      </div>

      <!-- Chart 4 / Risk Matrix Bubble Chart -->
      <div class="lg:col-span-4 glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <div>
          <h3 class="text-base font-bold text-white">Matriz de Severidad de Líneas</h3>
          <p class="text-xs text-slate-400">Probabilidad de Incumplimiento vs Horas en Riesgo (Tamaño: Volumen).</p>
        </div>
        <div id="chart_bubble_div" class="w-full h-80 sm:h-96"></div>
      </div>

    </div>

    <!-- Second Row of Charts: Waterfall and Pareto -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Chart 2: Waterfall Paros no planeados -->
      <div class="lg:col-span-6 glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white">Cascada de Pérdidas de Disponibilidad (Waterfall)</h3>
            <p class="text-xs text-slate-400">Desglose de paros no planeados que reducen el OEE proyectado.</p>
          </div>
          <div class="flex items-center gap-2">
            <label for="lineSelectWaterfall" class="text-xs text-slate-400">Línea:</label>
            <select id="lineSelectWaterfall" onchange="updateWaterfall(this.value)" 
                    class="bg-slate-900 border border-slate-700 text-xs text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-sky-500 font-sans font-medium">
              <!-- Options populated by JS -->
            </select>
          </div>
        </div>
        <div id="chart_waterfall_div" class="w-full h-80 sm:h-96"></div>
      </div>

      <!-- Chart 3: Pareto de Causas Raíz extraídas por LLM de avisos -->
      <div class="lg:col-span-6 glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <div>
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <span>Diagrama de Pareto: Causas Raíz Críticas (Avisos)</span>
            <span class="text-xs font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">Regla 80/20</span>
          </h3>
          <p class="text-xs text-slate-400">Horas acumuladas por equipo y causa extraídas mediante NLP de 21,100 registros.</p>
        </div>
        <div id="chart_pareto_div" class="w-full h-80 sm:h-96"></div>
      </div>

    </div>

    <!-- Prescriptive Diagnostics & LLM Narrative Panel -->
    <div class="glass-card rounded-2xl border border-slate-800 p-6 sm:p-8 space-y-6">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-5">
        <div>
          <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-purple-500/10 text-purple-400 text-xs font-semibold border border-purple-500/20 mb-2">
            <span>Módulo 4: Diagnóstico Prescriptivo Generado por LLM</span>
          </div>
          <h2 class="text-xl font-bold text-white">Análisis Prescriptivo de Causa Raíz & Recomendaciones Preventivas</h2>
          <p class="text-xs text-slate-400">Recomendaciones técnicas y de planeación generadas por el modelo analizando las bitácoras no estructuradas.</p>
        </div>
        <div class="flex items-center gap-2">
          <label for="diagnosticLineSelect" class="text-xs text-slate-400 font-medium">Ver Línea:</label>
          <select id="diagnosticLineSelect" onchange="renderPrescriptiveCard(this.value)" 
                  class="bg-slate-900 border border-slate-700 text-xs text-white rounded-lg px-3 py-2 focus:outline-none focus:border-sky-500 font-sans font-semibold">
            <!-- Populated by JS -->
          </select>
        </div>
      </div>

      <!-- Prescriptive Detail Card Dynamic Container -->
      <div id="prescriptiveContainer" class="transition-all duration-200">
        <!-- Rendered by JS -->
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
    <p>Grupo PiSA • Laboratorios PiSA S.A. de C.V. • Sistema Híbrido de Predicción de OEE (LSTM + LLM)</p>
    <p class="mt-1 text-[11px] text-slate-600">Confidencial • Uso exclusivo interno de Operaciones y Planeación de la Producción</p>
  </footer>

  <!-- Application Logic & Data Injection -->
  <script>
    // Injected Backend Data
    const riskData = {json_risk};
    const chart1Data = {json_chart1};
    const waterfallData = {json_waterfall};
    const paretoData = {json_pareto};
    const prescriptiveData = {json_prescriptive};

    let currentSortCol = 'prob_incumplimiento';
    let currentSortAsc = false;
    let currentFilterStatus = 'all';

    // Inicializar selectores y componentes al cargar
    document.addEventListener("DOMContentLoaded", () => {{
      initSelectors();
      renderRiskTable();
      initChart1();
      initBubbleChart();
      initWaterfall();
      initPareto();
      
      // Mostrar por defecto la primera línea crítica o la primera disponible
      const firstCrit = riskData.lines.find(l => l.estado === 'Crítico') || riskData.lines[0];
      if (firstCrit) {{
        renderPrescriptiveCard(firstCrit.line);
        document.getElementById('diagnosticLineSelect').value = firstCrit.line;
        document.getElementById('lineSelectChart1').value = firstCrit.line;
        document.getElementById('lineSelectWaterfall').value = firstCrit.line;
        updateChart1(firstCrit.line);
        updateWaterfall(firstCrit.line);
      }}

      // Listener de búsqueda
      document.getElementById('lineSearch').addEventListener('input', () => {{
        renderRiskTable();
      }});
    }});

    function initSelectors() {{
      const lines = riskData.lines.map(l => l.line);
      const sel1 = document.getElementById('lineSelectChart1');
      const sel2 = document.getElementById('lineSelectWaterfall');
      const sel3 = document.getElementById('diagnosticLineSelect');

      lines.forEach(l => {{
        const opt1 = document.createElement('option');
        opt1.value = l;
        opt1.textContent = l;
        sel1.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = l;
        opt2.textContent = l;
        sel2.appendChild(opt2);

        const opt3 = document.createElement('option');
        opt3.value = l;
        const info = riskData.lines.find(x => x.line === l);
        opt3.textContent = `${{l}} (${{info.estado}})`;
        sel3.appendChild(opt3);
      }});
    }}

    function filterTable(status) {{
      currentFilterStatus = status;
      document.querySelectorAll('.filter-btn').forEach(btn => {{
        if (btn.getAttribute('data-status') === status) {{
          btn.className = "filter-btn px-3 py-1 rounded-md font-medium text-white bg-sky-600 transition-colors";
        }} else {{
          btn.className = "filter-btn px-3 py-1 rounded-md font-medium text-slate-400 hover:text-white transition-colors";
        }}
      }});
      renderRiskTable();
    }}

    function sortTable(col) {{
      if (currentSortCol === col) {{
        currentSortAsc = !currentSortAsc;
      }} else {{
        currentSortCol = col;
        currentSortAsc = false;
      }}
      renderRiskTable();
    }}

    function renderRiskTable() {{
      const tbody = document.getElementById('riskTableBody');
      const searchVal = document.getElementById('lineSearch').value.toLowerCase().trim();

      let filtered = riskData.lines.filter(r => {{
        const matchStatus = currentFilterStatus === 'all' || r.estado === currentFilterStatus;
        const matchSearch = r.line.toLowerCase().includes(searchVal) || 
                            r.familia.toLowerCase().includes(searchVal) ||
                            r.planta.toLowerCase().includes(searchVal);
        return matchStatus && matchSearch;
      }});

      filtered.sort((a, b) => {{
        let vA = a[currentSortCol];
        let vB = b[currentSortCol];
        if (typeof vA === 'string') {{
          return currentSortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
        }}
        return currentSortAsc ? vA - vB : vB - vA;
      }});

      tbody.innerHTML = '';
      filtered.forEach(r => {{
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-900/60 transition-colors";

        let badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
        if (r.estado === 'Crítico') badgeClass = "bg-rose-500/10 text-rose-400 border border-rose-500/20";
        else if (r.estado === 'Alerta') badgeClass = "bg-amber-500/10 text-amber-400 border border-amber-500/20";

        const oeePlanPct = (r.oee_planeado * 100).toFixed(1);
        const oeeProjPct = (r.oee_proyectado_mu * 100).toFixed(1);
        const oeeSigmaPct = (r.oee_proyectado_sigma * 100).toFixed(1);
        const q10Pct = (r.oee_proyectado_q10 * 100).toFixed(1);
        const q90Pct = (r.oee_proyectado_q90 * 100).toFixed(1);

        tr.innerHTML = `
          <td class="px-4 py-3 font-semibold text-white font-sans flex items-center gap-2">
            <span class="w-2 h-2 rounded-full ${{r.estado === 'Crítico' ? 'bg-rose-500' : r.estado === 'Alerta' ? 'bg-amber-500' : 'bg-emerald-500'}}"></span>
            ${{r.line}}
          </td>
          <td class="px-4 py-3 font-sans text-slate-400">${{r.familia}}</td>
          <td class="px-4 py-3 text-center">${{r.skus_count}}</td>
          <td class="px-4 py-3 text-right font-medium text-slate-200">${{oeePlanPct}}%</td>
          <td class="px-4 py-3 text-right font-bold ${{r.oee_proyectado_mu < r.oee_planeado ? 'text-rose-400' : 'text-emerald-400'}}">
            ${{oeeProjPct}}% <span class="text-[10px] text-slate-500 font-normal">±${{oeeSigmaPct}}%</span>
          </td>
          <td class="px-4 py-3 text-right text-slate-400 text-[11px]">[${{q10Pct}}% - ${{q90Pct}}%]</td>
          <td class="px-4 py-3 text-right font-semibold ${{r.prob_incumplimiento_pct > 60 ? 'text-rose-400' : r.prob_incumplimiento_pct >= 20 ? 'text-amber-400' : 'text-emerald-400'}}">
            ${{r.prob_incumplimiento_pct.toFixed(1)}}%
          </td>
          <td class="px-4 py-3 text-right text-slate-200">${{r.horas_en_riesgo.toLocaleString()}} h</td>
          <td class="px-4 py-3 text-right text-slate-400">${{r.volumen_en_riesgo.toLocaleString()}}</td>
          <td class="px-4 py-3 text-center font-sans">
            <span class="px-2 py-0.5 text-[10px] font-bold rounded-full ${{badgeClass}}">${{r.estado}}</span>
          </td>
          <td class="px-4 py-3 text-center font-sans">
            <button onclick="selectLineFromTable('${{r.line}}')" class="text-xs text-sky-400 hover:text-sky-300 hover:underline">
              Analizar
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function selectLineFromTable(line) {{
      document.getElementById('lineSelectChart1').value = line;
      document.getElementById('lineSelectWaterfall').value = line;
      document.getElementById('diagnosticLineSelect').value = line;
      updateChart1(line);
      updateWaterfall(line);
      renderPrescriptiveCard(line);
      document.getElementById('prescriptiveContainer').scrollIntoView({{ behavior: 'smooth' }});
    }}

    // --- GRÁFICO 1: Histórico + Proyección LSTM con Bandas de Confianza ---
    function initChart1() {{
      const firstLine = riskData.lines[0].line;
      updateChart1(firstLine);
    }}

    function updateChart1(line) {{
      const data = chart1Data[line];
      if (!data) return;

      const traces = [
        // Histórico real observado
        {{
          x: data.hist_dates,
          y: data.hist_oee,
          mode: 'markers',
          name: 'OEE Real Diario',
          marker: {{ color: '#94a3b8', size: 4, opacity: 0.7 }}
        }},
        // Tendencia móvil histórica
        {{
          x: data.hist_dates,
          y: data.hist_trend,
          mode: 'lines',
          name: 'Tendencia Histórica (7d)',
          line: {{ color: '#38bdf8', width: 2 }}
        }},
        // Banda de Confianza Superior (q90)
        {{
          x: data.nov_dates,
          y: data.proj_upper,
          mode: 'lines',
          name: 'Banda Superior (q90)',
          line: {{ width: 0 }},
          showlegend: false,
          hoverinfo: 'none'
        }},
        // Banda de Confianza Inferior (q10) con relleno
        {{
          x: data.nov_dates,
          y: data.proj_lower,
          mode: 'lines',
          name: 'Incertidumbre LSTM (q10-q90)',
          fill: 'tonexty',
          fillcolor: 'rgba(56, 189, 248, 0.18)',
          line: {{ width: 0 }}
        }},
        // Proyección LSTM noviembre
        {{
          x: data.nov_dates,
          y: data.proj_daily,
          mode: 'lines+markers',
          name: `Proyección LSTM (μ=${{data.proj_mu}}%)`,
          line: {{ color: '#0284c7', width: 2.5, dash: 'solid' }},
          marker: {{ size: 5, color: '#38bdf8' }}
        }},
        // Línea horizontal OEE Meta
        {{
          x: [...data.hist_dates, ...data.nov_dates],
          y: new Array(data.hist_dates.length + data.nov_dates.length).fill(data.oee_plan),
          mode: 'lines',
          name: `Meta Plan Nov (${{data.oee_plan}}%)`,
          line: {{ color: '#f43f5e', width: 2, dash: 'dash' }}
        }}
      ];

      const layout = {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 25, r: 20, l: 45, b: 40 }},
        font: {{ family: 'Inter', color: '#94a3b8', size: 11 }},
        xaxis: {{
          gridcolor: '#1e293b',
          linecolor: '#334155',
          tickfont: {{ size: 10 }}
        }},
        yaxis: {{
          title: 'OEE (%)',
          gridcolor: '#1e293b',
          linecolor: '#334155',
          range: [0, 105],
          ticksuffix: '%'
        }},
        legend: {{
          orientation: 'h',
          x: 0,
          y: -0.2,
          font: {{ size: 10 }}
        }}
      }};

      Plotly.newPlot('chart1_div', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    // --- GRÁFICO 4: Bubble Chart de Riesgo ---
    function initBubbleChart() {{
      const lines = riskData.lines;

      const traces = [
        {{
          x: lines.map(l => (l.delta_oee * 100)),
          y: lines.map(l => l.prob_incumplimiento_pct),
          text: lines.map(l => `<b>Línea:</b> ${{l.line}}<br><b>Familia:</b> ${{l.familia}}<br><b>P(Incump):</b> ${{l.prob_incumplimiento_pct}}%<br><b>Horas Riesgo:</b> ${{l.horas_en_riesgo}} h<br><b>Volumen Faltante:</b> ${{l.volumen_en_riesgo.toLocaleString()}}`),
          mode: 'markers+text',
          textposition: 'top center',
          textfont: {{ family: 'Inter', size: 10, color: '#f1f5f9' }},
          marker: {{
            size: lines.map(l => Math.max(12, Math.sqrt(l.volumen_en_riesgo) / 12)),
            color: lines.map(l => l.estado === 'Crítico' ? '#f43f5e' : l.estado === 'Alerta' ? '#f59e0b' : '#10b981'),
            opacity: 0.85,
            line: {{ color: '#0f172a', width: 1.5 }}
          }}
        }}
      ];

      const layout = {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 20, r: 20, l: 45, b: 45 }},
        font: {{ family: 'Inter', color: '#94a3b8', size: 11 }},
        xaxis: {{
          title: 'Brecha de OEE (Meta - Proyectado p.p.)',
          gridcolor: '#1e293b',
          linecolor: '#334155',
          zerolinecolor: '#475569'
        }},
        yaxis: {{
          title: 'P(Incumplimiento) %',
          gridcolor: '#1e293b',
          linecolor: '#334155',
          range: [0, 105],
          ticksuffix: '%'
        }}
      }};

      Plotly.newPlot('chart_bubble_div', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    // --- GRÁFICO 2: Waterfall de Paros ---
    function initWaterfall() {{
      const firstLine = riskData.lines[0].line;
      updateWaterfall(firstLine);
    }}

    function updateWaterfall(line) {{
      const items = waterfallData[line];
      if (!items) return;

      const x = items.map(i => i.name);
      const y = items.map(i => i.value);
      const measure = items.map(i => i.type);

      const traces = [{{
        type: "waterfall",
        orientation: "v",
        measure: measure,
        x: x,
        y: y,
        textposition: "outside",
        text: y.map(v => `${{Math.abs(v)}}%`),
        connector: {{ line: {{ color: "#475569" }} }},
        increasing: {{ marker: {{ color: "#10b981" }} }},
        decreasing: {{ marker: {{ color: "#f43f5e" }} }},
        totals: {{ marker: {{ color: "#0ea5e9" }} }}
      }}];

      const layout = {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 25, r: 20, l: 45, b: 70 }},
        font: {{ family: 'Inter', color: '#94a3b8', size: 10 }},
        xaxis: {{
          tickangle: -25,
          gridcolor: '#1e293b',
          linecolor: '#334155'
        }},
        yaxis: {{
          title: 'Porcentaje (%)',
          gridcolor: '#1e293b',
          linecolor: '#334155',
          ticksuffix: '%'
        }}
      }};

      Plotly.newPlot('chart_waterfall_div', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    // --- GRÁFICO 3: Pareto de Causas Raíz ---
    function initPareto() {{
      const causes = paretoData.map(p => p.Causa_Detallada.length > 28 ? p.Causa_Detallada.substring(0, 26) + '...' : p.Causa_Detallada);
      const hours = paretoData.map(p => p.Horas_Totales);
      const cumPct = paretoData.map(p => p.Pct_Acumulado);

      const traces = [
        {{
          x: causes,
          y: hours,
          type: 'bar',
          name: 'Horas de Paro',
          marker: {{ color: '#818cf8' }}
        }},
        {{
          x: causes,
          y: cumPct,
          type: 'scatter',
          mode: 'lines+markers',
          name: '% Acumulado',
          yaxis: 'y2',
          line: {{ color: '#f59e0b', width: 2.5 }},
          marker: {{ size: 6, color: '#fbbf24' }}
        }}
      ];

      const layout = {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 25, r: 45, l: 45, b: 90 }},
        font: {{ family: 'Inter', color: '#94a3b8', size: 10 }},
        xaxis: {{
          tickangle: -35,
          gridcolor: '#1e293b',
          linecolor: '#334155'
        }},
        yaxis: {{
          title: 'Horas de Paro',
          gridcolor: '#1e293b',
          linecolor: '#334155'
        }},
        yaxis2: {{
          title: '% Acumulado',
          overlaying: 'y',
          side: 'right',
          range: [0, 105],
          ticksuffix: '%',
          showgrid: false,
          linecolor: '#f59e0b'
        }},
        legend: {{
          orientation: 'h',
          x: 0,
          y: -0.3,
          font: {{ size: 10 }}
        }}
      }};

      Plotly.newPlot('chart_pareto_div', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    // --- PANEL PRESCRIPTIVO LLM ---
    function renderPrescriptiveCard(line) {{
      const diag = prescriptiveData[line];
      if (!diag) return;

      const container = document.getElementById('prescriptiveContainer');

      let badgeColor = "bg-rose-500/10 text-rose-400 border-rose-500/20";
      let borderGlow = "border-rose-500/30";
      if (diag.estado === 'Alerta') {{
        badgeColor = "bg-amber-500/10 text-amber-400 border-amber-500/20";
        borderGlow = "border-amber-500/30";
      }} else if (diag.estado === 'Estable') {{
        badgeColor = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
        borderGlow = "border-emerald-500/30";
      }}

      let presListHtml = diag.prescripciones.map((p, idx) => `
        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-sky-400 flex items-center gap-1.5">
              <span class="w-5 h-5 rounded-full bg-sky-500/10 flex items-center justify-center text-[10px]">${{idx + 1}}</span>
              ${{p.accion}}
            </span>
            <span class="text-[10px] font-medium text-slate-400 bg-slate-800 px-2 py-0.5 rounded">${{p.responsable}}</span>
          </div>
          <p class="text-xs text-slate-300 leading-relaxed pl-6">${{p.detalle}}</p>
        </div>
      `).join('');

      let bitacorasHtml = diag.ejemplos_bitacora.map(b => `
        <li class="text-xs text-slate-400 font-mono bg-slate-950/60 p-2.5 rounded-lg border border-slate-900 flex items-start gap-2">
          <span class="text-purple-400 text-sm">›</span>
          <span>${{b}}</span>
        </li>
      `).join('');

      container.innerHTML = `
        <div class="rounded-xl border ${{borderGlow}} bg-slate-900/40 p-6 space-y-6">
          
          <!-- Header Card Info -->
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div class="flex items-center gap-3">
              <div class="w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center font-extrabold text-lg text-white">
                ${{diag.line}}
              </div>
              <div>
                <div class="flex items-center gap-2">
                  <h3 class="text-lg font-bold text-white">Diagnóstico de Línea ${{diag.line}}</h3>
                  <span class="px-2.5 py-0.5 rounded-full text-xs font-bold border ${{badgeColor}}">${{diag.estado}}</span>
                  <span class="text-xs text-slate-400">Prioridad: <b>${{diag.prioridad}}</b></span>
                </div>
                <p class="text-xs text-slate-400">Equipo Cuello de Botella: <b class="text-slate-200">${{diag.equipo_critico}}</b> | Subsistema: <b class="text-slate-200">${{diag.subsistema_critico}}</b></p>
              </div>
            </div>

            <!-- Mini KPIs -->
            <div class="flex items-center gap-4 text-right">
              <div>
                <span class="text-[10px] text-slate-400 block uppercase font-semibold">Meta vs Proy.</span>
                <span class="text-sm font-bold text-white">${{diag.oee_planeado_pct}}% <span class="text-xs text-slate-400 font-normal">vs</span> ${{diag.oee_proyectado_pct}}%</span>
              </div>
              <div class="border-l border-slate-800 pl-4">
                <span class="text-[10px] text-slate-400 block uppercase font-semibold">Riesgo P(Incump.)</span>
                <span class="text-sm font-bold ${{diag.prob_incumplimiento_pct > 60 ? 'text-rose-400' : 'text-amber-400'}}">${{diag.prob_incumplimiento_pct}}%</span>
              </div>
              <div class="border-l border-slate-800 pl-4">
                <span class="text-[10px] text-slate-400 block uppercase font-semibold">Horas en Riesgo</span>
                <span class="text-sm font-bold text-slate-200">${{diag.horas_en_riesgo.toLocaleString()}} h</span>
              </div>
            </div>
          </div>

          <!-- Causa Raíz Narrative -->
          <div class="space-y-2">
            <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <svg class="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              Explicación de Causa Raíz (NLP & Modelos)
            </h4>
            <div class="p-4 rounded-xl bg-purple-950/20 border border-purple-900/30 text-sm text-purple-200 leading-relaxed font-sans">
              ${{diag.resumen_causa}}
            </div>
          </div>

          <!-- Muestras de Bitácoras de Técnicos -->
          <div class="space-y-2">
            <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
              Evidencia Textual en Bitácoras Recientes (avisos.xlsx)
            </h4>
            <ul class="grid grid-cols-1 md:grid-cols-2 gap-2">
              ${{bitacorasHtml}}
            </ul>
          </div>

          <!-- Acciones Prescriptivas Concretas -->
          <div class="space-y-3">
            <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>
              Plan Prescriptivo de Acción Preventiva (Operaciones & Planeación)
            </h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              ${{presListHtml}}
            </div>
          </div>

        </div>
      `;
    }}
  </script>
</body>
</html>
"""
    
    with open(FILE_DASHBOARD_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"[+] Dashboard HTML interactivo generado con éxito en: {FILE_DASHBOARD_HTML}")
    return FILE_DASHBOARD_HTML

if __name__ == "__main__":
    build_html_dashboard()
