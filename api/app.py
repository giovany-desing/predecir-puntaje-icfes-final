from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import logging
from pathlib import Path
import sys


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
# Importar configuración centralizada
from utils.config import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
logger = logging.getLogger("api")

# Crear app FastAPI
app = FastAPI(
  title="ICFES Prediction API",
  description="API para predecir puntajes globales del examen ICFES",
  version="1.0.0"
  )

# Variable global para el modelo
model = None


def create_prediction_input_schema():
    """
    Crea dinámicamente el schema de Pydantic usando las features del config.yaml
    """
    features = config.features
    logger.info(f"Creando schema con features: {features}")

    # Crear campos dinámicamente: {nombre_feature: (tipo, requerido)}
    fields = {feature: (float, ...) for feature in features}

    # Crear modelo Pydantic dinámicamente
    return create_model('PredictionInput', **fields)


# Crear el schema de entrada
PredictionInput = create_prediction_input_schema()


@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la aplicación"""
    global model

    model_path = config.model_path
    logger.info(f"Intentando cargar modelo desde: {model_path}")

    # Verificar que el modelo existe
    if not model_path.exists():
        logger.error(f"❌ MODELO NO ENCONTRADO: {model_path}")
        logger.error("=" * 60)
        logger.error("SOLUCIÓN:")
        logger.error("1. Entrenar el modelo con:")
        logger.error("   docker-compose run --rm api python train_optuna.py")
        logger.error("2. O si ya entrenaste localmente, reinicia:")
        logger.error("   docker-compose restart api")
        logger.error("=" * 60)
        # NO fallar el startup, solo advertir
        return

    try:
        # Cargar el modelo
        model = joblib.load(model_path)
        logger.info("✅ Modelo cargado exitosamente")
        logger.info(f"Tipo de modelo: {type(model)}")

        # Cargar metadata si existe
        metadata_path = config.metadata_path
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            logger.info(f"📊 Metadata del modelo: {metadata.get('model_name', 'N/A')}")
            logger.info(f"📊 R² Test: {metadata.get('test_r2', 'N/A'):.4f}")

    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # NO fallar el startup
        return


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Puntajes ICFES",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "features": config.features,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict/"
          }
      }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: Modelo no cargado. Ejecuta: docker-compose run --rm api python train_optuna.py"
          )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(config.model_path)
    }


@app.post("/predict/")
async def predict(data: PredictionInput):
    """
    Endpoint de predicción.
    Recibe features y devuelve predicción del puntaje global.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Servicio no disponible",
                "message": "El modelo no está cargado",
                "solution": "Ejecuta: docker-compose run --rm api python train_optuna.py"
            }
        )

    try:
        # Obtener features en el orden correcto
        features = config.features

        # Convertir input a lista en el orden correcto
        input_data = [[getattr(data, feature) for feature in features]]

        logger.info(f"Input recibido: {input_data}")

        # Hacer predicción
        prediction = model.predict(input_data)[0]

        logger.info(f"Predicción realizada: {prediction}")

        # Validar que la predicción está en rango válido
        if not (0 <= prediction <= 500):
            logger.warning(f"⚠️ Predicción fuera de rango esperado: {prediction}")

        return {
            "prediction": float(prediction),
            "features_used": features,
            "input_values": dict(data)
        }

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al hacer predicción: {str(e)}"
        )


@app.get("/config")
async def get_config():
    """Endpoint para ver la configuración actual (útil para debugging)"""
    return {
        "features": config.features,
        "target": config.target_col,
        "model_path": str(config.model_path),
        "data_path": str(config.raw_data_path)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)