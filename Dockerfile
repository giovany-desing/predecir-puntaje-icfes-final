# Usa una imagen base de Python ligera
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los requisitos y los instala primero (para usar cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El comando dvc[s3] está en requirements.txt, pero asegúrate de que la API no falle si no se usa dvc.
# Para el entorno de FastAPI, la API solo necesita el modelo, no el dvc pull.

# Copia todo el código del proyecto
# Esto incluye api/, train_model/, y cualquier modelo previamente entrenado que se haya subido al contexto
COPY . . 

# El puerto por defecto para Render es 8000
EXPOSE 8000

# Comando para iniciar la aplicación FastAPI
# Nota: Asegúrate de que 'api.app:app' es la ruta correcta a tu instancia de FastAPI (app) dentro del módulo api/app.py
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]