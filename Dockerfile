# Usa una imagen base de Python ligera
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los requisitos y los instala primero (para usar cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

# El puerto por defecto para Render es 8000
EXPOSE 8000

# Comando para iniciar la aplicación FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]