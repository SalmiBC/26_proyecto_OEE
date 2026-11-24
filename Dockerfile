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

# Instalar PyTorch versión CPU (180MB en vez de 4GB de drivers CUDA innecesarios) y dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, interfaces web y artefactos del modelo
COPY src/ ./src/
COPY data/master_oee_dataset.parquet ./data/master_oee_dataset.parquet
COPY models/ ./models/
COPY app_consulta.html ./app_consulta.html
COPY presentacion_rubrica_oee.html ./presentacion_rubrica_oee.html

# Exponer el puerto por defecto de la API
EXPOSE 8000

# Comando de ejecución con soporte para puerto dinámico (AWS App Runner / ECS / Fargate)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}"]
