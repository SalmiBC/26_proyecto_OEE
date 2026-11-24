# Imagen base ultra ligera de Python
FROM python:3.11-slim

# Metadatos del contenedor
LABEL maintainer="Planeación de la Producción - Laboratorios PiSA S.A. de C.V."
LABEL description="Contenedor Docker para el Servicio REST de Predicción de OEE y Riesgo"

# Evitar escritura de bytecode y habilitar buffer directo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKERS=2

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar PyTorch CPU ligero según arquitectura (x86_64 o arm64/aarch64) y dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, datos procesados, modelos e interfaces web
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY app_consulta.html ./app_consulta.html
COPY presentacion_rubrica_oee.html ./presentacion_rubrica_oee.html
COPY dashboard_oee_predictivo.html ./dashboard_oee_predictivo.html
COPY presentacion_didactica_lstm_llm.html ./presentacion_didactica_lstm_llm.html

# Exponer el puerto por defecto de la API
EXPOSE 8000

# Comando de ejecución con soporte para puerto dinámico (AWS App Runner / ECS / Fargate)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}"]
