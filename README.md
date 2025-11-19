⏺ Aquí está el contenido completo del README.md:

  # 🎓 Predicción de Puntajes ICFES - Sistema MLOps en Producción

  Sistema completo de Machine Learning para predecir puntajes del examen ICFES con precisión del 98.4%. Implementa pipeline
  MLOps end-to-end con versionado de datos, experimentación sistemática, CI/CD automatizado y deployment en producción.

  [![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
  [![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
  [![DVC](https://img.shields.io/badge/DVC-S3-purple.svg)](https://dvc.org/)

  ---

  ## 📖 Descripción del Proyecto

  ### Objetivo
  Predecir el puntaje global del examen ICFES (prueba estandarizada colombiana) basándose en los puntajes de las áreas
  individuales evaluadas.

  ### Variables
  **Features de entrada (5):**
  - Razonamiento Cuantitativo (0-100)
  - Lectura Crítica (0-100)
  - Competencias Ciudadanas (0-100)
  - Inglés (0-100)
  - Comunicación Escrita (0-100)

  **Target:**
  - Puntaje Global (0-500)

  ### Dataset
  - **Registros:** ~9,000 estudiantes
  - **Tamaño:** 8.1 MB
  - **Almacenamiento:** AWS S3 (versionado con DVC)

  ### Rendimiento del Modelo
  - **Algoritmo:** RandomForest (seleccionado automáticamente)
  - **R² Score:** 98.4%
  - **MAE:** ~5 puntos
  - **Optimización:** 50 trials con Optuna (búsqueda bayesiana)

  ---

  ## 🏗️ Stack MLOps

  ### Versionado y Almacenamiento
  - **Git:** Control de versiones del código
  - **DVC (Data Version Control):** Versionado de datos y modelos
  - **AWS S3:** Almacenamiento remoto (bucket: `proyecto-icfes-data`)

  ### Experimentación y Entrenamiento
  - **Optuna:** Optimización bayesiana de hiperparámetros
  - **MLflow:** Tracking de experimentos, métricas y parámetros
  - **scikit-learn:** Pipeline de ML con StandardScaler
  - **XGBoost:** Algoritmo de gradient boosting

  ### API y Deployment
  - **FastAPI:** Framework de API REST de alto rendimiento
  - **Uvicorn:** Servidor ASGI
  - **Pydantic:** Validación de datos

  ### DevOps
  - **Docker:** Containerización de la aplicación
  - **GitHub Actions:** Pipeline CI/CD automatizado
  - **Render:** Plataforma de deployment cloud
  - **Docker Hub:** Registro de imágenes

  ---

  ## 🔄 Pipeline MLOps Implementado

  El proyecto implementa un ciclo MLOps completo:

  **1. Versionado de Datos**
  - DVC trackea datasets y modelos en S3
  - Git trackea código y archivos .dvc
  - Reproducibilidad garantizada con hashes

  **2. Experimentación Sistemática**
  - MLflow registra todos los experimentos
  - Optuna optimiza hiperparámetros automáticamente
  - Comparación de 3 algoritmos: RandomForest, GradientBoosting, XGBoost

  **3. Entrenamiento Automatizado**
  - Pipeline de limpieza de datos
  - Cross-validation de 5 folds
  - Métricas completas: R², MAE, RMSE, MAPE
  - Selección automática del mejor modelo

  **4. CI/CD Automatizado**
  - GitHub Actions trigger en push a main
  - DVC pull para descargar datos/modelos
  - Entrenamiento automático del modelo
  - Build y push de imagen Docker
  - Deploy automático a Render

  **5. Serving en Producción**
  - API REST con FastAPI
  - Health checks para load balancers
  - Validación automática de inputs/outputs
  - Logging estructurado

  ---

  ## 🚀 Quick Start para Desarrolladores

  ### Prerequisitos
  ```bash
  # Instalar dependencias del sistema
  brew install python@3.9  # macOS
  sudo apt install python3.9  # Ubuntu

  # Instalar DVC
  pip install dvc dvc-s3

  Setup del Proyecto

  # 1. Clonar repositorio
  git clone <repository-url>
  cd predecir_puntaje_icfes

  # 2. Crear entorno virtual
  python3 -m venv venv
  source venv/bin/activate  # Linux/Mac
  # venv\Scripts\activate   # Windows

  # 3. Instalar dependencias
  pip install -r requirements.txt

  # 4. Configurar AWS (solo primera vez)
  dvc remote modify s3_remote access_key_id YOUR_AWS_KEY
  dvc remote modify s3_remote secret_access_key YOUR_AWS_SECRET

  # 5. Descargar datos y modelos
  dvc pull

  ---
  📘 Manual de Uso

  1. Entrenar Modelo Localmente

  # Descargar datos de entrenamiento
  dvc pull data/raw/data_train.csv.dvc

  # Instalar dependencias
  pip install -r requirements.txt

  # Ejecutar entrenamiento
  python train_model/train_model.py

  ¿Qué hace el script?
  1. Carga y limpia datos con DataPipeline
  2. Split 80/20 (train/test)
  3. Optimiza hiperparámetros (50 trials × 3 modelos)
  4. Entrena con mejores parámetros
  5. Evalúa con 5-fold CV
  6. Registra en MLflow
  7. Guarda mejor modelo en models/best_model.pkl

  Tiempo estimado: 5-10 minutos

  Archivos generados:
  - models/best_model.pkl - Modelo serializado
  - models/model_metadata.pkl - Metadata (métricas, params, git hash)
  - plots_temp/*.png - Gráficas de optimización
  - mlruns/ - Experimentos MLflow

  ---
  2. Versionar Cambios (DVC + Git)

  # Después de entrenar un nuevo modelo
  dvc add models/best_model.pkl
  dvc add models/model_metadata.pkl

  # Commit archivos .dvc
  git add models/*.dvc
  git commit -m "Update model - improved R2 to 98.5%"

  # Push modelo a S3
  dvc push

  # Push código a GitHub
  git push origin main

  Beneficios:
  - Modelos rastreables por hash
  - Recuperación de cualquier versión anterior
  - Colaboración sin conflictos en archivos grandes

  ---
  3. Correr API en Local

  Opción A: Python directo (Desarrollo)
  # Asegurarse de tener modelos descargados
  dvc pull

  # Iniciar servidor de desarrollo
  uvicorn api.app:app --reload --port 8000

  Opción B: Docker (Producción)
  # Descargar modelos
  dvc pull

  # Build imagen
  docker build -t icfes-api .

  # Run contenedor
  docker run -p 8000:8000 icfes-api

  Verificar API:
  # Health check
  curl http://localhost:8000/health

  # Documentación interactiva
  open http://localhost:8000/docs

  ---
  4. Hacer Predicciones

  Usando cURL:
  curl -X POST http://localhost:8000/predict/ \
    -H "Content-Type: application/json" \
    -d '{
      "MOD_RAZONA_CUANTITATIVO_PNAL": 75,
      "MOD_LECTURA_CRITICA_PNAL": 80,
      "MOD_COMPETEN_CIUDADA_PNAL": 70,
      "MOD_INGLES_PNAL": 65,
      "MOD_COMUNI_ESCRITA_PNAL": 72
    }'

  Respuesta:
  {
    "prediction": 285.4
  }

  Usando Python:
  import requests

  response = requests.post(
      "http://localhost:8000/predict/",
      json={
          "MOD_RAZONA_CUANTITATIVO_PNAL": 75,
          "MOD_LECTURA_CRITICA_PNAL": 80,
          "MOD_COMPETEN_CIUDADA_PNAL": 70,
          "MOD_INGLES_PNAL": 65,
          "MOD_COMUNI_ESCRITA_PNAL": 72
      }
  )
  print(response.json())

  ---
  5. Explorar Experimentos con MLflow

  # Iniciar MLflow UI
  mlflow ui --backend-store-uri file:./mlruns

  # Abrir en navegador
  open http://localhost:5000

  Features de MLflow UI:
  - Comparar múltiples runs
  - Visualizar métricas (R², MAE, RMSE)
  - Ver hiperparámetros utilizados
  - Descargar artifacts (gráficas, modelos)
  - Filtrar por tags (git commit, data hash)

  ---
  📁 Estructura del Proyecto

  predecir_puntaje_icfes/
  ├── api/
  │   └── app.py                      # API FastAPI con 4 endpoints
  ├── train_model/
  │   └── train_model.py              # Script de entrenamiento (600 líneas)
  ├── utils/
  │   ├── data_clean.py               # DataPipeline para limpieza (263 líneas)
  │   └── config.py                   # Config singleton (83 líneas)
  ├── data/raw/
  │   ├── data_train.csv              # Dataset (~9000 filas, 8.1 MB)
  │   └── data_train.csv.dvc          # Puntero DVC a S3
  ├── models/
  │   ├── best_model.pkl              # Modelo entrenado (714 KB)
  │   ├── model_metadata.pkl          # Metadata del modelo (556 B)
  │   ├── best_model.pkl.dvc          # Puntero DVC
  │   └── model_metadata.pkl.dvc      # Puntero DVC
  ├── .github/workflows/
  │   └── pipeline.yml                # CI/CD automatizado
  ├── .dvc/
  │   └── config                      # Configuración S3 remoto
  ├── config.yaml                     # Configuración centralizada
  ├── Dockerfile                      # Containerización
  ├── buildspec.yml                   # AWS CodeBuild (alternativo)
  ├── requirements.txt                # 170 dependencias Python
  └── README.md                       # Este archivo

  ---
  🔬 Detalles Técnicos

  Algoritmos Evaluados

  | Modelo           | Hiperparámetros Optimizados                                                           | Trials | Best
   R² |
  |------------------|---------------------------------------------------------------------------------------|--------|-----
  ----|
  | RandomForest     | n_estimators, max_depth, min_samples_split/leaf, max_features                         | 50     |
  ~98.4%  |
  | GradientBoosting | n_estimators, max_depth, learning_rate, subsample, min_samples                        | 50     |
  ~98.2%  |
  | XGBoost          | n_estimators, max_depth, learning_rate, subsample, colsample, gamma, reg_alpha/lambda | 50     |
  ~98.1%  |

  Estrategia de selección: Mejor R² score en cross-validation de 5 folds

  Pipeline de Preprocesamiento

  DataPipeline ejecuta:
  1. Carga del CSV crudo
  2. Selección de 6 columnas (5 features + 1 target)
  3. Eliminación de duplicados
  4. Eliminación de valores nulos
  5. Validación de rangos (features: 0-100, target: 0-500)
  6. Reset de índices
  7. Estadísticas descriptivas

  Pipeline del modelo:
  Pipeline([
      ('scaler', StandardScaler()),  # Normalización Z-score
      ('model', RandomForestRegressor(**best_params))
  ])

  Métricas Calculadas

  Durante entrenamiento:
  - R² Score: Coeficiente de determinación
  - MAE: Mean Absolute Error
  - RMSE: Root Mean Squared Error
  - MAPE: Mean Absolute Percentage Error

  Cross-validation:
  - 5-fold CV para evaluar generalización
  - Promedio y desviación estándar de métricas

  ---
  🌐 Endpoints de la API

  | Método | Endpoint  | Descripción                          | Ejemplo                           |
  |--------|-----------|--------------------------------------|-----------------------------------|
  | GET    | /         | Información general de la API        | curl http://localhost:8000/       |
  | GET    | /health   | Health check (valida modelo cargado) | curl http://localhost:8000/health |
  | POST   | /predict/ | Predicción de puntaje ICFES          | Ver sección "Hacer Predicciones"  |
  | GET    | /config   | Configuración actual (debugging)     | curl http://localhost:8000/config |
  | GET    | /docs     | Documentación Swagger interactiva    | open http://localhost:8000/docs   |

  Características de la API:
  - Validación automática con Pydantic
  - Schema dinámico desde config.yaml
  - Logging estructurado
  - Manejo robusto de errores
  - Validación de outputs (0-500)

  ---
  🔄 CI/CD Pipeline (GitHub Actions)

  Trigger: Push a branch main

  Fases automatizadas:

  1. Setup
     - Checkout código
     - Setup Python 3.9
     - Instalar dependencias
     - Configurar credenciales AWS

  2. Data
     - dvc pull (descargar datos desde S3)

  3. Train (CI)
     - Ejecutar train_model.py
     - Generar best_model.pkl

  4. Build (CD)
     - docker build
     - docker push a Docker Hub

  5. Deploy (CD)
     - Trigger webhook de Render
     - Deploy automático

  Tiempo total: ~8 minutos

  Secretos requeridos:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_REGION
  - DOCKER_USERNAME
  - DOCKER_PASSWORD
  - RENDER_DEPLOY_HOOK

  ---
  🔐 Configuración de Secretos

  Para DVC (local)

  # Opción 1: DVC remote config
  dvc remote modify s3_remote access_key_id YOUR_KEY
  dvc remote modify s3_remote secret_access_key YOUR_SECRET

  # Opción 2: Variables de entorno
  export AWS_ACCESS_KEY_ID=your_key
  export AWS_SECRET_ACCESS_KEY=your_secret
  export AWS_DEFAULT_REGION=us-east-1

  Para GitHub Actions

  Settings → Secrets and variables → Actions → New repository secret

  Agregar:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_REGION
  - DOCKER_USERNAME
  - DOCKER_PASSWORD
  - RENDER_DEPLOY_HOOK

  ---
  🧪 Testing y Validación

  Test de API en local

  # Health check
  curl http://localhost:8000/health

  # Predicción válida
  curl -X POST http://localhost:8000/predict/ \
    -H "Content-Type: application/json" \
    -d '{"MOD_RAZONA_CUANTITATIVO_PNAL": 80, "MOD_LECTURA_CRITICA_PNAL": 85, "MOD_COMPETEN_CIUDADA_PNAL": 75, 
  "MOD_INGLES_PNAL": 70, "MOD_COMUNI_ESCRITA_PNAL": 78}'

  # Test de validación (debe fallar)
  curl -X POST http://localhost:8000/predict/ \
    -H "Content-Type: application/json" \
    -d '{"MOD_RAZONA_CUANTITATIVO_PNAL": 150}'  # Fuera de rango

  Verificar modelo entrenado

  import joblib

  # Cargar modelo y metadata
  model = joblib.load('models/best_model.pkl')
  metadata = joblib.load('models/model_metadata.pkl')

  print(f"Modelo: {metadata['model_name']}")
  print(f"R² Score: {metadata['test_r2']:.4f}")
  print(f"MAE: {metadata['test_mae']:.2f}")

  ---
  🐛 Troubleshooting

  Problema: dvc pull falla con error de credenciales

  # Solución: Verificar configuración
  dvc remote list
  dvc config cache.dir

  # Reconfigurar remoto
  dvc remote modify s3_remote access_key_id YOUR_KEY
  dvc remote modify s3_remote secret_access_key YOUR_SECRET

  # Test de conexión
  aws s3 ls s3://proyecto-icfes-data/

  Problema: Docker build falla por falta de modelos

  # Solución: Descargar modelos antes de build
  dvc pull models/

  # Verificar que existen
  ls -lh models/*.pkl

  # Rebuild
  docker build -t icfes-api .

  Problema: API retorna error 500 al predecir

  # Solución: Verificar logs
  docker logs <container_id>

  # Revisar que config.yaml tiene todas las features
  cat config.yaml

  # Probar predicción con todos los campos
  curl -X POST http://localhost:8000/predict/ \
    -H "Content-Type: application/json" \
    -d '{
      "MOD_RAZONA_CUANTITATIVO_PNAL": 75,
      "MOD_LECTURA_CRITICA_PNAL": 80,
      "MOD_COMPETEN_CIUDADA_PNAL": 70,
      "MOD_INGLES_PNAL": 65,
      "MOD_COMUNI_ESCRITA_PNAL": 72
    }'

  Problema: Entrenamiento falla por memoria

  # Solución: Reducir trials de Optuna
  # Editar config.yaml:
  training:
    optuna_trials: 20  # Reducir de 50 a 20

  # O usar menos datos para testing rápido

  ---
  📊 Metadata del Modelo

  El archivo model_metadata.pkl contiene:

  {
      "model_name": "RandomForest",
      "cv_r2_mean": 0.984,
      "cv_r2_std": 0.002,
      "test_r2": 0.982,
      "test_mae": 5.23,
      "test_rmse": 7.15,
      "test_mape": 1.85,
      "mlflow_run_id": "abc123...",
      "feature_names": [...],
      "best_params": {...},
      "git_commit": "cadd00e",
      "data_hash": "0def2cc71...",
      "trained_at": "2024-11-18T18:07:00"
  }

  Uso de metadata:
  - Auditoría de modelos en producción
  - Reproducibilidad (git commit + data hash)
  - Comparación de versiones
  - Debugging de performance

  ---
  🤝 Contribución

  Workflow de contribución

  # 1. Fork y clonar
  git clone <your-fork-url>
  cd predecir_puntaje_icfes

  # 2. Crear branch
  git checkout -b feature/nueva-funcionalidad

  # 3. Hacer cambios y commit
  git add .
  git commit -m "Add: nueva funcionalidad"

  # 4. Push y crear PR
  git push origin feature/nueva-funcionalidad

  Guidelines

  - Seguir PEP 8 para código Python
  - Agregar docstrings a funciones nuevas
  - Actualizar requirements.txt si se agregan dependencias
  - Probar localmente antes de PR
  - Incluir descripción clara en el PR

  ---
  📚 Recursos Adicionales

  Documentación de herramientas:
  - https://dvc.org/doc
  - https://mlflow.org/docs/latest/index.html
  - https://fastapi.tiangolo.com/
  - https://optuna.readthedocs.io/

  Best practices MLOps:
  - https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
  - https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model

  ---
  📝 Licencia

  Este proyecto es de código abierto y está disponible bajo la licencia MIT.

  ---
  👤 Autor

  Edgar Yovany Samaca Acuña

  Proyecto desarrollado como demostración de habilidades en Machine Learning y MLOps, implementando pipeline completo desde
  experimentación hasta deployment en producción.

  ---
  🎯 Próximas Mejoras

  - Feature Store con Feast
  - A/B testing framework
  - Data drift monitoring con Evidently AI
  - Kubernetes deployment con Helm
  - Model registry formal con MLflow
  - Tests automatizados (pytest)
  - Monitoring con Prometheus + Grafana