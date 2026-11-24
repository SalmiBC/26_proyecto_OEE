@echo off
chcp 65001 > nul
echo ===============================================================================
echo  INICIANDO SERVICIO REST DE PREDICCION DE OEE Y RIESGO (FASTAPI + UVICORN)
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

echo [*] Levantando API REST en http://127.0.0.1:8000
echo [*] Documentacion Swagger interactiva en http://127.0.0.1:8000/docs
echo.
"%PYTHON_EXE%" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
pause
