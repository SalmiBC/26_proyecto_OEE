# Sistema Híbrido de Predicción de OEE, Estimación de Riesgo de Plan y Diagnóstico de Fallas

**Proyecto Individual de Aplicación Industrial**  
**Empresa:** Laboratorios PiSA S.A. de C.V. — Dirección de Planeación de la Producción  
*Alineado a los contenidos y rúbrica formal de evaluación de la materia (Bases de Datos, Pipelines/ETL, Jobs Programados, Machine Learning y Arquitectura Medallion).*

---

## 🎯 Matriz de Cumplimiento de la Rúbrica de Evaluación

| Criterio de Rúbrica | Requerimiento Oficial | Implementación en este Proyecto | Sección en Documento / Código |
| :--- | :--- | :--- | :--- |
| **Criterio 1: Limpieza y transformación de datos** | Tratamiento justificado de nulos, duplicados, outliers y tipos de dato. Normalización y codificación. Metodología por capas (Arquitectura Medallion: bronce, plata, oro). | • **Nulos:** Imputación justificada a `0.0` en horas de paro (ausencia de evento).<br>• **Duplicados:** Deduplicación por llave compuesta (Fecha + Línea).<br>• **Outliers:** Filtrado de registros físicamente inconsistentes ($OEE > 100\%$ o horas negativas).<br>• **Tipos de dato:** Cast estricto (`pd.to_datetime`, `pd.to_numeric`).<br>• **Normalización:** `LINE_MAPPING` canónico y `MinMaxScaler` ajustado solo en Train.<br>• **Medallion:** Separación física en Bronce (Raw), Plata (Cleaned/NLP Parquet) y Oro (Sliding Windows 30d). | [Sección 2.3](#23-arquitectura-medallion-paso-a-paso-criterio-1-mlops) y [Sección 11.1](#111-criterio-1-limpieza-transformación-y-arquitectura-medallion)<br>Código: [`src/data_pipeline.py`](src/data_pipeline.py) |
| **Criterio 2: Definición del problema y métricas** | Tipo de tarea (clasificación, regresión, clustering, NLP, etc.). Métrica principal justificada (ej. F1 si las clases están desbalanceadas). | • **Tarea Dual + NLP:** Regresión continua de OEE/paros temporales + Clasificación de incumplimiento por saturación ($Sat > 100\%$) + NLP TF-IDF para bitácoras.<br>• **Métricas de Regresión:** MAE ($14.55\%$), RMSE ($20.17\%$) y MAPE para error de magnitud física.<br>• **Métrica de Clasificación Justificada:** **F1-Score y Recall** debido al desbalance natural de clases (los paros críticos y sobre-saturaciones son eventos minoritarios donde un falso negativo detiene la planta). | [Sección 11.2](#112-criterio-2-definición-del-problema-y-métricas)<br>Código: [`src/lstm_model.py`](src/lstm_model.py) y [`src/risk_engine.py`](src/risk_engine.py) |
| **Criterio 3: Publicación del modelo** | Modelo expuesto como servicio consultable: API REST (FastAPI, Flask), contenedor Docker, servicio en la nube o endpoint gestionado. API documentada (Swagger u OpenAPI) y con validación de entradas. | • **API REST:** Microservicio en **FastAPI** ([`src/api.py`](src/api.py)).<br>• **Contenedor:** Dockerfile de producción con soporte dinámico de arquitectura y CPU PyTorch ([`Dockerfile`](Dockerfile)).<br>• **Nube en Producción:** Desplegado en **AWS EC2 (Ohio us-east-2, t4g.micro ARM64)**: [http://13.59.96.134](http://13.59.96.134).<br>• **Documentación:** Swagger UI (`/docs`) y OpenAPI JSON (`/openapi.json`).<br>• **Validación de Entradas:** Esquemas **Pydantic v2** con validación de tipos, rangos (1 a 744 h/mes) y control de errores HTTP 422. | [Sección 11.3](#113-criterio-3-publicación-del-modelo-fastapi-docker-y-aws)<br>Código: [`src/api.py`](src/api.py) y [`Dockerfile`](Dockerfile) |
| **Criterio 4: Integración del modelo con aplicación de consulta** | La aplicación consume el endpoint publicado, no un modelo copiado localmente. Maneja errores, tiempos de espera y respuestas inválidas. | • **Consumo Desacoplado:** Tanto el dashboard web ([`app_consulta.html`](app_consulta.html)) como el cliente CLI ([`client.py`](client.py)) consumen exclusivamente el endpoint REST HTTP/HTTPS publicado.<br>• **Tiempos de Espera:** Control de timeout de 12 segundos vía `AbortController` (JS) y `requests.Timeout` (Python).<br>• **Manejo de Errores:** Captura controlada de HTTP 422 (validación), 500 (servidor), 503 (inicialización) y excepciones de red/conexión. | [Sección 11.4](#114-criterio-4-integración-del-modelo-con-la-aplicación-de-consulta)<br>Código: [`app_consulta.html`](app_consulta.html) y [`client.py`](client.py) |
| **Temas del Curso** | Bases de datos, pipelines, jobs o tareas programadas, Machine Learning, ETL, arquitectura Medallion. | • **Bases de Datos / Formatos:** Almacenamiento columnar en Parquet particionado.<br>• **ETL / Pipelines:** Extracción y transformación modular en `src/data_pipeline.py`.<br>• **Jobs Programados:** Automatización batch vía `main.py` y `run_pipeline.bat` para ejecuciones periódicas o cron.<br>• **Machine Learning:** Red neuronal PyTorch LSTM con Monte Carlo Dropout para cuantificación de incertidumbre epistémica.<br>• **Medallion:** Separación Bronce, Plata y Oro. | [Sección 2](#2-fuentes-de-datos-y-reglas-de-negocio-exactas), [Sección 2.3](#23-arquitectura-medallion-paso-a-paso-criterio-1-mlops) y [Sección 6](#6-guía-de-uso-y-ejecución) |

---

## 1. Identificación y Objetivos del Proyecto

El **Sistema Híbrido de Predicción de OEE** combina modelos estocásticos de aprendizaje profundo (**Deep Learning con redes recurrentes LSTM y Monte Carlo Dropout**) con procesamiento de lenguaje natural y modelos de lenguaje (**LLM**) para abordar tres necesidades críticas en la manufactura farmacéutica:

1. **Predicción Probabilística del OEE Real**: Proyectar la distribución de probabilidad del OEE ($\mu$, $\sigma$, $q_{10}$, $q_{50}$, $q_{90}$) por línea de producción para el horizonte de planeación mensual (Noviembre), incorporando tanto la inercia temporal cuantitativa como la dinámica de fallas en equipos.
2. **Cuantificación Precisa del Riesgo de Plan**: Contrastar la proyección estocástica del OEE contra las metas comprometidas en el Plan Maestro de Fabricación (`Plan_nov.xlsx`), calculando la probabilidad matemática de incumplimiento $P(OEE_{\text{real}} < OEE_{\text{planeado}})$, las horas máquina en riesgo y el volumen de producto terminado (piezas) potencialmente faltante.
3. **Diagnóstico Causal y Prescripción Accionable (LLM)**: Estructurar más de 21,100 registros textuales de bitácoras de mantenimiento (`avisos.xlsx`) mediante taxonomías de fallas por subsistema y equipo, identificando causas raíz recurrentes (Pareto 80/20) y prescribiendo planes de acción preventivos por línea antes del arranque de producción.

---

## 2. Fuentes de Datos y Reglas de Negocio Exactas

### 2.1. Fuentes de Entrada

| Archivo | Contenido Clave | Rol en el Sistema |
| :--- | :--- | :--- |
| `OEE 2024-2026.xlsx` | 18,959 registros diarios por línea. Horas de fabricación (Col N), paros planeados (Cols V a AK), paros no planeados (Cols O a U) y tiempo para fabricación (Col AL). | **Entrada Cuantitativa** para series temporales y feature engineering. |
| `avisos.xlsx` | 21,100 órdenes y bitácoras de mantenimiento con línea, equipo, síntomas, causas y textos libres de técnicos. | **Entrada Cualitativa** para extracción de taxonomías y variables de falla. |
| `Plan_nov.xlsx` | 167 órdenes programadas para noviembre en 19 líneas: SKUs, volumen programado (PVO), fechas y OEE meta. | **Línea Base (Target)** para cuantificación de brecha y riesgo de entrega. |

### 2.2. Fórmulas de Negocio Implementadas

1. **Horas para Fabricar (Tiempo para Fabricación)**:
   $$\text{Horas para Fabricar} = \text{Tiempo para Fabricación [Col AL]}$$
   *(Incluye la base completa de tiempo operativo y las 7 categorías de paros no planeados: Ineficiencia, Cambios y limpiezas, Exceso Cambios y limpiezas, Falta de material, Falta de personal, Falta de servicios y Comedor).*

2. **OEE Real Calculado**:
   $$OEE_{\text{real}} = \frac{\text{Horas de Fabricación [Col N]}}{\text{Horas para Fabricar}} \times 100$$

3. **Brecha de OEE ($\Delta OEE$)**:
   $$\Delta OEE = OEE_{\text{planeado}} - \mu_{OEE\text{ (proyectado)}}$$

4. **Probabilidad de Incumplimiento del Plan**:
   Bajo inferencia estocástica con Monte Carlo Dropout ($N = 200$ simulaciones hacia adelante):
   $$P(\text{Incumplimiento}) = P(OEE_{\text{real}} < OEE_{\text{planeado}}) = \Phi\left(\frac{OEE_{\text{planeado}} - \mu_{OEE}}{\sigma_{OEE}}\right)$$

5. **Horas de Producción en Riesgo**:
   $$\text{Horas Faltantes} = \max\left(0, (OEE_{\text{planeado}} - \mu_{OEE})\right) \times \text{Horas Programadas del Mes}$$

6. **Volumen Faltante en Riesgo (Unidades)**:
   $$\text{Volumen Faltante} = \frac{\text{Horas Faltantes}}{\text{Horas Programadas}} \times \text{Volumen Total Programado}$$

7. **Criterios de Semaforización Oficial**:
   - 🟢 **Estable**: $P(\text{Incumplimiento}) < 20\%$
   - 🟡 **Alerta**: $20\% \le P(\text{Incumplimiento}) \le 60\%$
   - 🔴 **Crítico**: $P(\text{Incumplimiento}) > 60\%$

---

## 2.3. Arquitectura Medallion: Paso a Paso (Criterio 1 MLOps)

El procesamiento y refinamiento de los datos sigue formalmente el patrón de **Arquitectura Medallion** (Bronce $\rightarrow$ Plata $\rightarrow$ Oro), garantizando trazabilidad, calidad de datos y prevención de *Data Leakage*:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITECTURA MEDALLION DEL PROYECTO                          │
├────────────────────────────┬────────────────────────────┬──────────────────────────────┤
│   🥉 CAPA BRONCE (Raw)     │   🥈 CAPA PLATA (Cleaned)  │    🥇 CAPA ORO (Features)    │
├────────────────────────────┼────────────────────────────┼──────────────────────────────┤
│ • OEE 2024-2026.xlsx       │ • Tratamiento de Nulos     │ • Ventanas Deslizantes (30d) │
│ • avisos.xlsx (21.1k filas)│ • Homologación 19 Líneas   │ • Escalamiento sin Leakage   │
│ • Velocidad y OEE meta.xlsx│ • Cálculo OEE verificado   │ • Partición Train/Test       │
│ • Plan_nov.xlsx            │ • NLP & TF-IDF Mantenimiento│ • Tensores PyTorch [B,30,84] │
│   (Ingesta cruda inmutable)│   (data/master_oee...parquet) (Alimentación LSTM + Dropout) │
└────────────────────────────┴────────────────────────────┴──────────────────────────────┘
```

### Paso 1: Capa Bronce (Raw Data Lake)
- **Objetivo:** Ingesta y preservación íntegra de las fuentes operativas en su formato nativo sin transformaciones destructivas.
- **Fuentes Ingeridas:**
  1. `OEE 2024-2026.xlsx` (Hoja `BD`): 18,959 registros diarios transaccionales por línea (Col N para fabricación, Cols O a U para 7 paros no planeados, Cols V a AK para paros planeados, Col AL para tiempo operativo).
  2. `avisos.xlsx`: 21,100 registros con textos libres de bitácoras de órdenes de mantenimiento preventivo y correctivo.
  3. `Velocidad y OEE meta.xlsx`: Especificaciones técnicas de ingeniería industrial con velocidades nominales (PVOs/hora) y OEE estándar para las 19 líneas.
  4. `Plan_nov.xlsx`: 167 órdenes programadas para el mes target de evaluación.

### Paso 2: Capa Plata (Cleaned, Standardized & Conformed Data)
- **Objetivo:** Depuración de calidad, tipificación de datos, resolución de entidades y cruce multivariado.
- **Transformaciones ejecutadas (`src/data_pipeline.py` y `src/nlp_avisos.py`):**
  1. **Tratamiento de Inconsistencias y Caracteres Residuales:** Limpieza de guiones de Excel (`' - '`), strings vacíos y valores nulos, imputándolos a `0.0` para las columnas de horas y paros.
  2. **Normalización Canónica de Líneas:** Mapeo estricto de las variantes de nombres de líneas hacia el catálogo oficial de 19 subfamilias mediante diccionario estandarizado (`LINE_MAPPING`).
  3. **Cálculo Riguroso de Métricas Operativas:** Aplicación de la fórmula institucional de OEE real verificado:
     $$OEE_{\text{real}} = \frac{\text{Horas Fabricación [Col N]}}{\text{Tiempo para Fabricación [Col AL]}} \times 100$$
     filtrando divisiones por cero y registros donde las horas de fabricación excedan el tiempo programado.
  4. **Enriquecimiento Textual con NLP:** Vectorización TF-IDF de las bitácoras de mantenimiento (`avisos.xlsx`), agrupando conteos y severidades de averías por línea y fecha (`data/daily_maintenance_features.parquet`).
  5. **Unión Multivariada Conforme:** Merge temporal diario entre series de OEE y variables de mantenimiento, generando el dataset consolidado intermedio `data/master_oee_dataset.parquet` (18,959 filas x 84 columnas).

### Paso 3: Capa Oro (Feature Store & Deep Learning Ready)
- **Objetivo:** Preparación de estructuras tensoriales optimizadas para el entrenamiento y la inferencia del modelo LSTM estocástico.
- **Transformaciones ejecutadas (`src/lstm_model.py`):**
  1. **Generación de Ventanas Temporales Deslizantes (Sliding Windows):** Segmentación de series temporales continuas por línea en ventanas de 30 días de historial ($T = 30$), estructurando tensores de entrada con forma `[Batch, 30, Features]`.
  2. **Escalamiento Robusto Sin Fuga de Información (*Data Leakage*):** El escalador multivariado (`MinMaxScaler`) se ajusta exclusivamente con el subconjunto de entrenamiento histórico ($t \le 2026-01-31$) y se congela en `models/scaler.joblib`. El conjunto de prueba se transforma estrictamente usando los parámetros aprendidos en entrenamiento.
  3. **Partición Temporal Estricta (*Out-of-Sample Split*):**
     - **Train Set:** Enero 2024 al 31 de Enero de 2026 (13,325 secuencias temporales).
     - **Test Set Fuera de Muestra:** 1 de Febrero al 30 de Septiembre de 2026 (5,326 secuencias de validación real).
  4. **Estructuración Multi-Target:** Configuración de cabezales duales de predicción: OEE porcentual proyectado (salida Sigmoid acotada $[0, 1]$) y vector de horas proyectadas para las 7 categorías de paros no planeados (salida ReLU $\ge 0$).

---

## 3. Arquitectura Técnica del Sistema

```
┌─────────────────────────────────┐       ┌───────────────────────────────────┐
│       Archivo: avisos.xlsx      │       │     Archivo: OEE 2024-2026.xlsx   │
│  (21,100 bitácoras de mantto)   │       │   (18,959 registros diarios OEE)  │
└────────────────┬────────────────┘       └─────────────────┬─────────────────┘
                 │                                          │
                 ▼                                          ▼
   ┌────────────────────────────┐              ┌───────────────────────────┐
   │    Módulo 1: NLP LLM       │              │ Feature Engineering       │
   │ - Taxonomía (5 Fallas)     │              │ - OEE real verificado     │
   │ - Severidad (1 a 5)        │              │ - Rolling means/std, lags │
   │ - Vectores temporales 7d   │              │ - Componentes cíclicos    │
   └─────────────┬──────────────┘              └─────────────┬─────────────┘
                 │                                           │
                 └────────────────┬──────────────────────────┘
                                  │ (84 variables consolidadas)
                                  ▼
                ┌───────────────────────────────────┐
                │      Módulo 2: Red LSTM Dual      │
                │ - 2 capas recurrentes (h=64)      │
                │ - Dual Head: Sigmoid (OEE)        │
                │   + ReLU (7 Paros No Planeados)   │
                │ - Monte Carlo Dropout (N=200)     │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │     Módulo 3: Motor de Riesgo de Plan         │
          │ - Cruce con Plan_nov (167 SKUs / 19 Líneas)   │
          │ - P(OEE_real < OEE_plan), Horas & Vol Riesgo  │
          │ - Descomposición en Cascada (Waterfall)       │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │   Módulo 4: Motor Prescriptivo LLM & HTML     │
          │ - Diagnóstico causal por equipo crítico       │
          │ - Acciones preventivas con roles asignados    │
          │ - Tablero interactivo (dashboard_oee...html)  │
          └───────────────────────────────────────────────┘
```

---

## 4. Estructura del Repositorio

```
26_proyecto_OEE/
│
├── dashboard_oee_predictivo.html          # Tablero interactivo único y autónomo (HTML + Tailwind + Plotly.js)
├── presentacion_maestria_oee.html         # Presentación ejecutiva/académica de maestría (12 diapositivas con KaTeX)
├── presentacion_didactica_lstm_llm.html   # Presentación didáctica escolar: LSTM + LLM para principiantes (12 slides)
├── main.py                                # Orquestador central por línea de comando
├── run_pipeline.bat                       # Lanzador interactivo rápido para Windows
│
├── src/
│   ├── config.py                    # Rutas, hiperparámetros, catálogo de líneas y umbrales de riesgo
│   ├── nlp_avisos.py                # Pipeline NLP de bitácoras de mantenimiento y taxonomía
│   ├── data_pipeline.py             # Limpieza de OEE, cálculo de métricas y cruce de variables
│   ├── lstm_model.py                # Arquitectura PyTorch LSTM Multi-Output con Monte Carlo Dropout
│   ├── risk_engine.py               # Motor estocástico de riesgo, semáforos y cálculo de brechas
│   ├── prescriptive_engine.py       # Generador de diagnósticos causales y planes de acción LLM
│   ├── multimonth_forecast.py       # Motor de proyección multi-mes estocástico (Nov 2026 - Abr 2027)
│   ├── generate_dashboard.py              # Compilador del dashboard HTML operativo
│   ├── generate_presentation.py           # Compilador de la presentación de maestría HTML
│   ├── generate_didactic_presentation.py  # Compilador de la presentación didáctica escolar
│   └── templates/
│       ├── presentation_template.html          # Plantilla oficial de defensa de maestría
│       └── didactic_presentation_template.html # Plantilla interactiva para estudiantes
│
├── data/
│   ├── processed_avisos.parquet            # Bitácoras de mantenimiento procesadas y etiquetadas
│   ├── daily_maintenance_features.parquet  # Vectores temporales agregados por línea y fecha
│   ├── master_oee_dataset.parquet          # Dataset consolidado multivariado (18,959 filas x 84 cols)
│   ├── risk_assessment_results.json        # Evaluación completa de riesgo por línea para Noviembre
│   ├── prescriptive_diagnostics.json       # Diagnósticos y recomendaciones prescriptivas
│   └── multimonth_forecast_results.json    # Proyecciones a 6 meses con intervalos de confianza
│
└── models/
    ├── best_lstm_model.pth          # Pesos guardados del modelo neuronal PyTorch entrenado
    └── scaler.joblib                # Escalador multivariado RobustScaler / StandardScaler
```

---

## 5. Resumen de Resultados Globales (Noviembre 2026)

Evaluación efectuada sobre las **19 líneas de manufactura** programadas en `Plan_nov.xlsx`:

| Indicador Global | Meta / Plan | Proyección LSTM | Impacto / Riesgo |
| :--- | :---: | :---: | :---: |
| **OEE Promedio de la Planta** | **55.3%** | **53.8%** | **-1.5% Brecha Global** |
| **Cumplimiento Global Esperado** | 100.0% | **98.1%** | Desviación contenida |
| **Horas Totales de Fabricación en Riesgo** | — | — | **512.1 Horas** |
| **Volumen de Producción en Riesgo** | 43,628,095 pzas | 42,954,956 pzas | **673,139 Unidades** |
| **Líneas Críticas ($P > 60\%$)** | — | — | **13 Líneas** |
| **Líneas en Alerta ($20\% \le P \le 60\%$)** | — | — | **1 Línea** (`VA1`) |
| **Líneas Estables ($P < 20\%$)** | — | — | **5 Líneas** (`ANES`, `BP`, `HM`, `JERINGAS`, `SV1`) |

### Principales Líneas Críticas con Mayor Volumen / Horas en Riesgo:
1. **Línea A4**: OEE Plan 54.0% vs Proyectado 43.5% | $P = 99.4\%$ | **98.3 hrs en riesgo** (126,500 piezas). Falla crítica: *Etiquetadora C9 & Encartonadora*.
2. **Línea AV6**: OEE Plan 58.0% vs Proyectado 52.8% | $P = 86.8\%$ | **47.6 hrs en riesgo** (141,600 piezas). Falla crítica: *Inyectora y Sistema de Enfriamiento*.
3. **Línea VA3**: OEE Plan 60.0% vs Proyectado 54.4% | $P = 89.2\%$ | **53.9 hrs en riesgo** (86,200 piezas). Falla crítica: *Selladora y Formadora de Envases*.
4. **Línea OFT**: OEE Plan 58.0% vs Proyectado 52.6% | $P = 87.4\%$ | **42.1 hrs en riesgo** (44,700 piezas). Falla crítica: *Lavadora y Dosificadora Rotativa*.

---

## 6. Guía de Uso y Ejecución

### 6.1. Requisitos del Entorno
- Python 3.10 o superior (validado en Python 3.14).
- Librerías principales: `torch`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `openpyxl`.

### 6.2. Ejecución Rápida en Windows
Doble clic o ejecución directa en terminal:
```cmd
run_pipeline.bat
```
El script mostrará un menú interactivo para abrir el dashboard o re-entrenar el modelo.

### 6.3. Ejecución por Línea de Comandos (`main.py`)

1. **Re-generar únicamente el Dashboard HTML**:
   ```powershell
   python main.py --dashboard-only
   ```
2. **Re-evaluar el Motor de Riesgo con el modelo entrenado**:
   ```powershell
   python main.py --skip-train --mc-samples 200
   ```
3. **Re-entrenar el Pipeline Completo de Punta a Punta**:
   ```powershell
   python main.py --all --epochs 45 --mc-samples 200
   ```

---

## 7. Funcionalidades del Dashboard HTML (`dashboard_oee_predictivo.html`)

El tablero es un archivo HTML autónomo que no requiere backend ni servidores web para funcionar:

- **Tarjetas de KPI Globales**: OEE Planeado vs Proyectado, Horas en Riesgo, Cumplimiento y conteo de semáforos en tiempo real.
- **Gráfico 1 (Curva Histórica + LSTM con Bandas de Confianza)**: Visualización temporal interactiva con los últimos 60 días reales, la trayectoria esperada para noviembre y el sombreado estocástico del intervalo de confianza $q_{10} - q_{90}$, contrastado contra la línea horizontal de meta.
- **Gráfico 2 (Cascada de Paros / Waterfall)**: Desglose dinámico de cómo el OEE Ideal (100%) se degrada paso a paso por Paros Menores, Fallas Mecánicas, Fallas Eléctricas, Ajustes y Cambios de Formato.
- **Gráfico 3 (Diagrama de Pareto 80/20 de Causas Raíz)**: Ranking de equipos y causas extraídas por el LLM a partir de los 21,100 avisos de mantenimiento, con línea de porcentaje acumulado.
- **Gráfico 4 (Matriz de Riesgo Operativo - Bubble Chart)**: Mapeo de dispersión donde el eje X representa el OEE Planeado, el eje Y la Probabilidad de Incumplimiento, el tamaño de burbuja las Horas en Riesgo y el color el semáforo.
- **Matriz de Riesgo con Filtros y Búsqueda**: Tabla interactiva con filtros por estado (Todos, Crítico, Alerta, Estable), buscador por nombre de línea y ordenamiento dinámico.
- **Panel Prescriptivo por Línea (Narrativa LLM)**: Selector de línea que despliega el equipo crítico diagnosticado, extractos reales de bitácoras de técnicos y un plan de acción preventivo con responsables asignados.

---

## 8. Presentación Ejecutiva y Académica de Maestría (`presentacion_maestria_oee.html`)

Para la defensa de tesis o presentación académica/directiva, se compiló un archivo HTML de diapositivas interactivas en pantalla completa con soporte de renderizado matemático (KaTeX), navegación por teclado y gráficos interactivos de Plotly.js:

### 8.1. Controles de Navegación:
- `Flecha Derecha` / `Espacio`: Siguiente diapositiva.
- `Flecha Izquierda`: Diapositiva anterior.
- `F`: Alternar modo pantalla completa.
- `Inicio (Home)` / `Fin (End)`: Ir a la primera o última diapositiva.
- Barra de progreso interactiva en la parte superior.

### 8.2. Estructura de las 12 Diapositivas:
1. **Portada**: Título oficial, maestría, empresa colaboradora y autores.
2. **Contexto Operativo**: Problema del OEE en la industria farmacéutica y motivación.
3. **Fuentes de Datos y Reglas**: Diagrama de flujo de entrada, fórmulas Col N/AL y reglas de negocio.
4. **Módulo 1 - NLP & LLM**: Normalización de bitácoras no estructuradas, taxonomía de 5 clases y severidad 1-5.
5. **Módulo 2 - Arquitectura LSTM**: Formulación matemática de compuertas ($i_t, f_t, o_t, c_t$) y salida dual.
6. **Inferencia Bayesiana**: Monte Carlo Dropout, aproximación variacional e incertidumbre epistémica vs aleatoria.
7. **Módulo 3 - Motor de Riesgo**: Ecuaciones de $P(\text{Incumplimiento})$, horas y piezas faltantes, semáforos.
8. **Proyección Multi-Mes (Fan Chart)**: Discusión teórica de proyección a 6 meses con gráfico estocástico interactivo.
9. **Resultados Globales**: Resumen de las 19 líneas, KPIs agregados y distribución de criticidad.
10. **Pareto 80/20 y Prescripción**: Causas raíz extraídas por NLP y planes de acción preventivos.
11. **Estudios de Caso Reales**: Comparativa profunda entre Línea A4 (Crítica) vs Línea SV1 (Estable).
12. **Conclusiones y Trabajo Futuro**: Contribución académica, impacto industrial y roadmap a tiempo real.

---

## 9. Proyección Multi-Mes de OEE ($H = 6$ Meses: Nov 2026 - Abr 2027)

El sistema soporta la proyección a múltiples meses hacia el futuro mediante **estrategia autorregresiva iterativa con Monte Carlo Dropout**:

1. **Metodología**:
   - Cada mes $t+1$ predice $(\mu, \sigma)$ de OEE y las 7 componentes de paros no planeados.
   - Las salidas se retroalimentan como lags y medias móviles para proyectar $t+2, \dots, t+6$.
   - **Propagación Natural de Incertidumbre**: El desvío estándar $\sigma$ crece progresivamente ($\sigma_{\text{M1}} \approx 1.7\%$ a $\sigma_{\text{M6}} \approx 2.4\%$), reflejando el aumento de entropía temporal.
2. **Resultados Guardados**: `data/multimonth_forecast_results.json`.
3. **Visualización**: Disponible interactivamente en la Diapositiva 8 de la Presentación de Maestría.

---

## 10. Presentación Escolar Didáctica (`presentacion_didactica_lstm_llm.html`)

Para exposiciones dirigidas a estudiantes, compañeros o audiencias sin experiencia previa en Machine Learning o ciencia de datos, se construyó una suite de diapositivas interactiva con enfoque pedagógico explicando el super equipo **LSTM + LLM**:

### 10.1. Características Pedagógicas:
- **Metáforas Cotidianas**:
  - **LLM**: *"El Detective de Bitácoras"*, que lee más de 21,100 notas escritas por mecánicos y traduce síntomas humanos a números de severidad (1 a 5).
  - **LSTM**: *"La libreta inteligente del operador experto (Don José)"*, con sus 3 sellos (Goma de borrar = Olvido, Lápiz = Entrada, Megáfono = Salida).
  - **Fusión Híbrida**: Por qué un cerebro solo no basta (LSTM no sabe leer texto y LLM no puede calcular series de tiempo numéricas con bandas de confianza).
- **Ejemplo Central de la Vida Real**: Casos concretos de **Laboratorios PiSA** (fabricación de ampolletas, fallas en etiquetadora C9, sueros en hospitales).
- **Widgets Interactivos**:
  - **Simulador de Lectura LLM**: Muestra notas reales de mecánicos en SAP y permite ver cómo el LLM las clasifica en vivo (Mecánica, Eléctrica, Operación).
  - **Simulador de Compuertas LSTM**: Botones interactivos para ver cómo reacciona cada compuerta ante eventos reales de mantenimiento de PiSA.
  - **El Gran Duelo (Plotly)**: Visualización interactiva que enfrenta un modelo ciego contra el sistema híbrido (LSTM + LLM) sobre la Línea A4.
  - **El LLM Prescriptivo**: Visualización del plan preventivo generado automáticamente para los ingenieros de planta.
  - **Trivia / Pop Quiz de Clase**: 3 tarjetas de preguntas interactivas con botón para revelar la respuesta y dinamizar la sesión escolar.

---

## 11. Cumplimiento Detallado de la Rúbrica de Evaluación

Este proyecto fue diseñado e implementado satisfaciendo puntualmente cada uno de los cuatro criterios de evaluación de la materia:

### 11.1. Criterio 1: Limpieza y Transformación de Datos
* **Tratamiento Justificado de Nulos:** En las columnas transaccionales de paros planeados (Cols V a AK) y no planeados (Cols O a U), los valores en blanco o guiones de Excel (`' - '`) representan la **ausencia del evento de paro** en ese turno de producción; por tanto, su imputación a `0.0` es matemáticamente correcta y preserva la integridad del balance de horas. Para la variable `Fecha`, los registros nulos se eliminan (`dropna`) por carecer de anclaje temporal.
* **Tratamiento de Duplicados:** Se eliminaron registros redundantes ocasionados por cierres de turno superpuestos en SAP, aplicando agregación temporal diaria (`groupby(['Fecha', 'Linea'])`) para garantizar una serie unificada por día y línea.
* **Tratamiento de Outliers e Inconsistencias Físicas:** Se identificaron registros atípicos donde las horas de fabricación capturadas superaban el tiempo programado del día (errores humanos de captura en planta que producían $OEE > 100\%$). Se aplicaron filtros de consistencia física acotando el $OEE \in [0, 100\%]$ y descartando paros con horas negativas.
* **Tipos de Dato:** Casteo y tipificación estricta mediante `pd.to_datetime` para índices temporales y `pd.to_numeric(..., errors='coerce')` para todas las variables continuas.
* **Normalización y Codificación:** Estandarización canónica de las 19 líneas de manufactura mediante `LINE_MAPPING` y escalamiento de características con `MinMaxScaler` ajustado **únicamente en el conjunto de entrenamiento histórico** ($t \le 2026-01-31$) y congelado en `models/scaler.joblib`.
* **Metodología por Capas (Arquitectura Medallion):**
  - **Capa Bronce (Raw):** Ingesta cruda e inmutable de los archivos Excel (`OEE 2024-2026.xlsx`, `avisos.xlsx`, `Velocidad y OEE meta.xlsx`, `Plan_nov.xlsx`).
  - **Capa Plata (Cleaned & Conformed):** Depuración, tipificación y cruce con NLP de bitácoras, guardado en formato columnar de alto rendimiento [`data/master_oee_dataset.parquet`](data/master_oee_dataset.parquet).
  - **Capa Oro (Feature Store):** Construcción de secuencias temporales en ventanas deslizantes de 30 días ($T=30$) estructuradas en tensores PyTorch `[Batch, 30, 84]`, con partición temporal estricta fuera de muestra (Train: 13,325 secuencias; Test: 5,326 registros).

### 11.2. Criterio 2: Definición del Problema y Métricas
* **Tipo de Tarea:** Enfoque Integral Multitarea:
  1. **Regresión Continua de Series de Tiempo:** Proyección estocástica del OEE porcentual futuro ($\mu$, $\sigma$) y desglose de horas de 7 categorías de paros no planeados mediante red recurrente PyTorch LSTM con Monte Carlo Dropout.
  2. **Clasificación Binaria de Riesgo de Saturación:** Determinación de factibilidad de entrega del plan industrial de fabricación:
     $$\text{Clase 0 (Factible / Cumple): } \text{Saturación Proyectada} \le 100\%$$
     $$\text{Clase 1 (Riesgo / Incumple): } \text{Saturación Proyectada} > 100\%$$
  3. **NLP (Procesamiento de Lenguaje Natural):** Extracción de taxonomías de averías en 21,100 bitácoras libres de mantenimiento con TF-IDF.
* **Métricas de Regresión (Evaluadas Out-of-Sample sobre 5,326 registros):**
  - **MAE (Mean Absolute Error) = 14.55%:** Métrica primaria que cuantifica el error promedio en unidades físicas directas (% de OEE), esencial para los ingenieros de planeación.
  - **RMSE (Root Mean Squared Error) = 20.17%:** Métrica complementaria que penaliza con mayor severidad las desviaciones grandes en líneas críticas.
  - **MAPE (Mean Absolute Percentage Error) = 73.28%:** Medida relativa de error porcentual.
* **Métrica Principal de Clasificación Justificada (F1-Score y Recall para Clases Desbalanceadas):**
  - En manufactura farmacéutica, las situaciones de incumplimiento de plan o saturación crítica son **clases fuertemente desbalanceadas** (la mayoría de las órdenes programadas se mantienen en rangos factibles).
  - La métrica de **Exactitud (*Accuracy*) no es representativa** porque un modelo ingenuo que siempre prediga "Cumple" tendría un Accuracy alto pero provocaría desabasto severo de medicamentos en hospitales.
  - Por ello, la evaluación de clasificación se rige por el **F1-Score y Recall**:
    $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
    Se prioriza maximizar el **Recall** en la clase de Incumplimiento para evitar a toda costa los *Falsos Negativos* (aprobar planes imposibles de fabricar en planta), alertando a la dirección con suficiente tiempo para programar turnos extras o maquilas.

### 11.3. Criterio 3: Publicación del Modelo (FastAPI, Docker y AWS)
* **Servicio Consultable REST:** Microservicio publicado en Python usando **FastAPI** y servidor asíncrono **Uvicorn** ([`src/api.py`](src/api.py)), con endpoints estructurados:
  - `GET /health`: Monitoreo de estado operativo y carga de modelos en memoria.
  - `GET /lines` y `GET /line-specs`: Catálogo de líneas y especificaciones nominales de ingeniería.
  - `POST /predict`: Inferencia estocástica pura de OEE con Monte Carlo Dropout.
  - `POST /evaluate-risk`: Evaluación probabilística de saturación, piezas faltantes y dictamen de cumplimiento.
  - `GET /` y `GET /presentacion`: Entrega unificada de las interfaces visuales interactivas.
* **Contenedorización en Docker:** Empaquetado de producción en [`Dockerfile`](Dockerfile) con imagen ligera `python:3.11-slim`, soporte dinámico para la variable `$PORT` y trabajadores configurables.
* **Despliegue en la Nube (AWS EC2 en Producción):** Desplegado de forma nativa en **Amazon Web Services (AWS EC2 / CloudShell)** en la región **Ohio (`us-east-2`)**, ejecutándose en una instancia **`t4g.micro` (AWS Graviton2 ARM64)** bajo la capa gratuita (*AWS Free Tier*, $0 USD):
  - **URL Producción AWS:** [`http://13.59.96.134`](http://13.59.96.134)
  - **Swagger OpenAPI en vivo:** [`http://13.59.96.134/docs`](http://13.59.96.134/docs)
  - **Rúbrica Interactiva en vivo:** [`http://13.59.96.134/presentacion`](http://13.59.96.134/presentacion)
  - **Healthcheck en vivo:** [`http://13.59.96.134/health`](http://13.59.96.134/health)
* **Documentación Interactiva:** Especificación OpenAPI v3 y documentación interactiva Swagger UI disponible de fábrica en el endpoint `/docs` y Redoc en `/redoc`.
* **Validación Robusta de Entradas:** Modelos declarativos con **Pydantic v2**:
  - Control de tipos, restricción de líneas al catálogo oficial, y validación de rangos ($1.0 \le \text{horas} \le 744.0$ h/mes, tope de 31 días naturales).
  - Manejo automático de respuestas de validación HTTP 422 (*Unprocessable Entity*) con mensajes de error explícitos.

### 11.4. Criterio 4: Integración del Modelo con la Aplicación de Consulta
* **Consumo Exclusivo del Endpoint Publicado:** Tanto el cliente web como el cliente CLI consumen **únicamente el servicio REST remoto/publicado**, sin importar ni cargar librerías del modelo (`.pth`) en el cliente.
* **Cliente Web Interactivo ([`app_consulta.html`](app_consulta.html)):**
  - Permite ingresar Línea, Horas Programadas (hasta 744 h), PVOs a fabricar, OEE Meta y Muestras Monte Carlo.
  - Dictamen en tiempo real (Banner Verde/Rojo, Saturación real vs disponible, OEE proyectado y piezas faltantes).
* **Cliente CLI de Terminal ([`client.py`](client.py)):**
  - Cliente Python con modo interactivo (`python client.py -i`) y modo de argumentos (`python client.py --linea AV3 --pvos 50000 --horas 510`).
  - Configurable vía parámetro `--url` o variable de entorno `OEE_API_URL` para conectarse a endpoints locales o en AWS.
* **Manejo Integral de Errores y Tiempos de Espera:**
  - **Tiempos de Espera (Timeout):** Cancelación automática de peticiones lentas tras 12 segundos mediante `AbortController` (web) y `timeout=12.0` (CLI).
  - **Respuestas Inválidas (HTTP 422):** Interceptación y visualización amigable de errores de validación de datos.
  - **Manejo de Errores de Red:** Diagnóstico contextual si el servicio no está disponible o rechaza la conexión, indicando los pasos de solución tanto en local como en AWS CloudWatch.
