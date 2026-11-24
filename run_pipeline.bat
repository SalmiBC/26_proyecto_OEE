@echo off
chcp 65001 > nul
echo ===============================================================================
echo  SISTEMA HIBRIDO DE PREDICCION DE OEE, RIESGO Y DIAGNOSTICO (LSTM + LLM)
echo  LABORATORIOS PISA S.A. DE C.V. - PLANEACION DE LA PRODUCCION
echo ===============================================================================
echo.

set PYTHON_EXE=C:\Users\Sbecerra\AppData\Local\Python\bin\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se encontro el interprete de Python en:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

echo Selecciona una opcion:
echo 1) Abrir Dashboard Operativo Interactivo (dashboard_oee_predictivo.html)
echo 2) Abrir Presentacion de Maestria Oficial (presentacion_maestria_oee.html)
echo 3) Abrir Presentacion Didactica para Alumnos (presentacion_didactica_lstm_llm.html)
echo 4) Generar Proyeccion Multimes (6 meses: Nov 2026 - Abr 2027)
echo 5) Re-evaluar Riesgo y Prescripciones (Usa modelo LSTM existente)
echo 6) Re-entrenar modelo LSTM y Pipeline completo (De punta a punta)
echo 7) Iniciar Servidor API REST de Inferencia (FastAPI / Uvicorn en puerto 8000)
echo 8) Abrir Aplicacion Web de Consulta Desacoplada (app_consulta.html)
echo 9) Salir
echo.
set /p OPTION="Ingresa tu opcion (1-9): "

if "%OPTION%"=="1" (
    echo.
    echo [*] Abriendo dashboard_oee_predictivo.html...
    start "" "dashboard_oee_predictivo.html"
    exit /b 0
)

if "%OPTION%"=="2" (
    echo.
    echo [*] Abriendo presentacion_maestria_oee.html...
    start "" "presentacion_maestria_oee.html"
    exit /b 0
)

if "%OPTION%"=="3" (
    echo.
    echo [*] Abriendo presentacion_didactica_lstm_llm.html...
    start "" "presentacion_didactica_lstm_llm.html"
    exit /b 0
)

if "%OPTION%"=="4" (
    echo.
    echo [*] Ejecutando proyeccion multimes (Iterative Monte Carlo Dropout)...
    "%PYTHON_EXE%" -m src.multimonth_forecast
    "%PYTHON_EXE%" -m src.generate_presentation
    "%PYTHON_EXE%" -m src.generate_didactic_presentation
    if %ERRORLEVEL% equ 0 (
        echo [OK] Proyecciones multimes y presentaciones actualizadas.
        start "" "presentacion_maestria_oee.html"
    )
    pause
    exit /b 0
)

if "%OPTION%"=="5" (
    echo.
    echo [*] Ejecutando evaluacion de riesgo y actualizando dashboard...
    "%PYTHON_EXE%" main.py --skip-train
    "%PYTHON_EXE%" -m src.generate_presentation
    "%PYTHON_EXE%" -m src.generate_didactic_presentation
    if %ERRORLEVEL% equ 0 (
        echo [OK] Proceso terminado con exito.
        start "" "dashboard_oee_predictivo.html"
    )
    pause
    exit /b 0
)

if "%OPTION%"=="6" (
    echo.
    echo [*] Ejecutando pipeline completo (NLP + Datos + LSTM + Riesgo + Dashboard + Presentaciones)...
    "%PYTHON_EXE%" main.py --all --epochs 45 --mc-samples 200
    "%PYTHON_EXE%" -m src.multimonth_forecast
    "%PYTHON_EXE%" -m src.generate_presentation
    "%PYTHON_EXE%" -m src.generate_didactic_presentation
    if %ERRORLEVEL% equ 0 (
        echo [OK] Pipeline completado exitosamente.
        start "" "dashboard_oee_predictivo.html"
    )
    pause
    exit /b 0
)

if "%OPTION%"=="7" (
    echo.
    echo [*] Levantando API REST en http://127.0.0.1:8000 ...
    echo [*] Documentacion interactiva Swagger en: http://127.0.0.1:8000/docs
    start "" "http://127.0.0.1:8000/docs"
    "%PYTHON_EXE%" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
    exit /b 0
)

if "%OPTION%"=="8" (
    echo.
    echo [*] Abriendo app_consulta.html...
    start "" "app_consulta.html"
    exit /b 0
)

echo Saliendo...
exit /b 0
