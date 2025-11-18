import pandas as pd
import numpy as np
import os
import sys
import hashlib
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

# --- AÑADIR SHUTIL PARA LIMPIEZA TEMPORAL ---
import shutil 
import tempfile
# --- FIN CAMBIO ---

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import xgboost as xgb

# Optuna
import optuna

# MLflow
import mlflow
import mlflow.sklearn

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# --- IMPORTAR JOBLIB AQUÍ ---
import joblib 
# --- FIN CAMBIO ---

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# IMPORTAR CONFIGURACIÓN CENTRALIZADA
from utils.config import config

# Importar DataPipeline desde el módulo de preprocesamiento
# Asumiendo que el script de preprocesamiento está en una carpeta llamada 'preprocessing'
preprocessing_path = project_root / "utils"
sys.path.append(str(preprocessing_path))
from utils.data_clean import DataPipeline  # Ajusta el nombre del archivo según corresponda

# Configuración
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# USAR CONFIGURACIÓN CENTRALIZADA
DATA_PATH = config.raw_data_path
FEATURES = config.features
TARGET = config.target_col
MODEL_PATH = config.model_path
METADATA_PATH = config.metadata_path
EXPERIMENT_NAME = config.get('mlflow.experiment_name', 'ICFES Prediction')
TEST_SIZE = config.get('training.test_size', 0.2)
RANDOM_STATE = config.get('training.random_state', 42)
CV_FOLDS = config.get('training.cv_folds', 5)
OPTUNA_TRIALS = config.get('training.optuna_trials', 50)

print(f"✅ Configuración cargada:")
print(f"   - Features: {FEATURES}")
print(f"   - Target: {TARGET}")
print(f"   - Model Path: {MODEL_PATH}")

# ============================================================================
# UTILIDADES
# ============================================================================

def get_git_commit_hash():
    """Obtiene el hash del commit actual de Git."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit[:8]
    except:
        return "unknown"

def get_git_branch():
    """Obtiene la rama actual de Git."""
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return branch
    except:
        return "unknown"

def calculate_data_hash(df):
    """Calcula MD5 hash del DataFrame."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:16]

def load_and_preprocess_data(config_path="config.yaml"):
    """Carga y preprocesa los datos usando DataPipeline."""
    print(f"📊 Cargando y preprocesando datos...")
    
    try:
        # Inicializar el pipeline de datos
        data_pipeline = DataPipeline(config_path=config_path)
        
        # Ejecutar el pipeline completo
        df_clean = data_pipeline.run()
        
        print(f"✅ Preprocesamiento completado:")
        print(f"   - Filas procesadas: {len(df_clean)}")
        print(f"   - Columnas: {list(df_clean.columns)}")
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Error en el preprocesamiento: {e}")
        raise

# ============================================================================
# MÉTRICAS Y EVALUACIÓN
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calcula múltiples métricas de regresión."""
    return {
        'r2_score': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def cross_validate_model(model, X, y, cv=5):
    """Realiza cross-validation y calcula métricas para cada fold."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    all_metrics = {
        'r2': [],
        'mae': [],
        'rmse': [],
        'mape': []
    }

    fold_predictions = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        metrics = calculate_metrics(y_val_fold, y_pred)

        all_metrics['r2'].append(metrics['r2_score'])
        all_metrics['mae'].append(metrics['mae'])
        all_metrics['rmse'].append(metrics['rmse'])
        all_metrics['mape'].append(metrics['mape'])

        fold_predictions.append((y_val_fold, y_pred))

        print(f"   Fold {fold_idx}/{cv}: "
              f"R2={metrics['r2_score']:.4f}, "
              f"MAE={metrics['mae']:.2f}, "
              f"RMSE={metrics['rmse']:.2f}")

    results = {
        'mean_r2': np.mean(all_metrics['r2']),
        'std_r2': np.std(all_metrics['r2']),
        'mean_mae': np.mean(all_metrics['mae']),
        'std_mae': np.std(all_metrics['mae']),
        'mean_rmse': np.mean(all_metrics['rmse']),
        'std_rmse': np.std(all_metrics['rmse']),
        'mean_mape': np.mean(all_metrics['mape']),
        'std_mape': np.std(all_metrics['mape']),
        'fold_predictions': fold_predictions
    }

    return results

# ============================================================================
# VISUALIZACIONES (Código no modificado)
# ============================================================================

def create_feature_importance_plot(model, feature_names, output_path):
    """Crea gráfico de importancia de features."""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_

        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    return False

def create_actual_vs_predicted_plot(fold_predictions, output_path):
    """Crea scatter plot de valores reales vs predichos."""
    plt.figure(figsize=(10, 8))

    y_true_all = np.concatenate([y_true.values for y_true, _ in fold_predictions])
    y_pred_all = np.concatenate([y_pred for _, y_pred in fold_predictions])

    plt.scatter(y_true_all, y_pred_all, alpha=0.5, s=20)

    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted (All CV Folds)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_residuals_plot(fold_predictions, output_path):
    """Crea gráfico de residuales."""
    y_true_all = np.concatenate([y_true.values for y_true, _ in fold_predictions])
    y_pred_all = np.concatenate([y_pred for _, y_pred in fold_predictions])
    residuals = y_true_all - y_pred_all

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(y_pred_all, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_optimization_history_plot(study, output_path):
    """Crea gráfico del historial de optimización de Optuna."""
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    values = [t.value if t.value is not None else 0 for t in trials]

    best_values = []
    best_so_far = -float('inf')

    for val in values:
        best_so_far = max(best_so_far, val)
        best_values.append(best_so_far)

    plt.figure(figsize=(12, 6))
    plt.plot(trial_numbers, values, 'o-', alpha=0.6, label='Trial Value')
    plt.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('R2 Score', fontsize=12)
    plt.title('Optimization History', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# OPTUNA OPTIMIZATION (Código no modificado)
# ============================================================================

def create_objective_function(X, y, model_name):
    """Crea función objetivo para Optuna según el tipo de modelo."""
    def objective(trial):
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': RANDOM_STATE
            }
            model = RandomForestRegressor(**params)

        elif model_name == "GradientBoosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': RANDOM_STATE
            }
            model = GradientBoostingRegressor(**params)

        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': RANDOM_STATE
            }
            model = xgb.XGBRegressor(**params, verbosity=0)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )

        return cv_scores.mean()

    return objective

def optimize_model(model_name, X, y, n_trials=50):
    """Optimiza hiperparámetros de un modelo usando Optuna."""
    print(f"\n{'='*70}")
    print(f"🔧 OPTIMIZANDO: {model_name}")
    print(f"{'='*70}")

    study = optuna.create_study(
        direction='maximize',
        study_name=f"{model_name}_optimization"
    )

    objective = create_objective_function(X, y, model_name)

    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    optimization_time = time.time() - start_time

    print(f"\n✅ Optimización completada: {n_trials} trials en {optimization_time/60:.1f} min")
    print(f"\n📋 Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")
    print(f"\n🎯 Mejor R2 Score (CV): {study.best_value:.4f}")

    return study.best_params, study

# ============================================================================
# TRAINING Y MLFLOW LOGGING
# ============================================================================

def train_optimized_model(model_name, best_params, X_train, y_train, X_test, y_test,
                          feature_names, metadata):
    """Entrena modelo con mejores parámetros y logguea todo a MLflow."""
    print(f"\n🔄 Entrenando modelo final con Cross-Validation ({CV_FOLDS} folds)...")

    if model_name == "RandomForest":
        model = RandomForestRegressor(**best_params)
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(**best_params)
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(**best_params, verbosity=0)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    start_time = time.time()
    cv_results = cross_validate_model(pipeline, X_train, y_train, cv=CV_FOLDS)
    training_time = time.time() - start_time

    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    print(f"\n📊 Resultados Finales (Mean ± Std):")
    print(f"   - R2 Score:  {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
    print(f"   - MAE:       {cv_results['mean_mae']:.2f} ± {cv_results['std_mae']:.2f}")
    print(f"   - RMSE:      {cv_results['mean_rmse']:.2f} ± {cv_results['std_rmse']:.2f}")
    print(f"\n📊 Test Set:")
    print(f"   - R2 Score:  {test_metrics['r2_score']:.4f}")
    print(f"   - MAE:       {test_metrics['mae']:.2f}")
    print(f"   - RMSE:      {test_metrics['rmse']:.2f}")
    
    # --- IMPLEMENTACIÓN DE LA OPCIÓN C: USAR LOG_ARTIFACT ---
    # 1. Crear directorio temporal para el modelo
    temp_dir = Path("./temp_mlflow_model")
    temp_dir.mkdir(exist_ok=True)
    
    # 2. Guardar el pipeline localmente usando joblib
    joblib.dump(pipeline, temp_dir / "pipeline.pkl")
    
    # 3. Crear un archivo de metadatos simple (para que MLflow lo pueda inferir)
    model_metadata = {
        "model_type": model_name,
        "artifact_path": "pipeline.pkl"
    }
    with open(temp_dir / "MLmodel", "w") as f:
        # Esto es un placeholder, pero ayuda a MLflow a inferir el 'flavor'
        f.write('artifact_path: pipeline.pkl\n')
        f.write(f'flavors:\n  python_function: \n    loader_module: mlflow.sklearn\n')
        f.write('  sklearn:\n    serialization_format: cloudpickle\n')


    # MLflow Logging
    with mlflow.start_run(run_name=f"{model_name}_Optimized") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", CV_FOLDS)

        mlflow.set_tags({
            "git_commit": metadata['git_commit'],
            "git_branch": metadata['git_branch'],
            "data_hash": metadata['data_hash'],
            "model_name": model_name
        })

        mlflow.log_metrics({
            "cv_r2_mean": cv_results['mean_r2'],
            "cv_r2_std": cv_results['std_r2'],
            "cv_mae_mean": cv_results['mean_mae'],
            "cv_mae_std": cv_results['std_mae'],
            "cv_rmse_mean": cv_results['mean_rmse'],
            "cv_rmse_std": cv_results['std_rmse'],
            "test_r2": test_metrics['r2_score'],
            "test_mae": test_metrics['mae'],
            "test_rmse": test_metrics['rmse'],
            "training_time_seconds": training_time
        })

        print("\n✅ Métricas registradas en MLflow (visualizaciones omitidas)")
        
        # 4. Usar la función segura: log_artifacts para subir la carpeta temporal
        mlflow.log_artifacts(local_dir=temp_dir, artifact_path="model_artifact_pipeline")

        run_id = run.info.run_id

        print(f"\n✅ Modelo (como artefacto) registrado en MLflow")
        print(f"   Run ID: {run_id}")
        
    # 5. Limpieza del directorio temporal
    shutil.rmtree(temp_dir)
    print(f"🧹 Directorio temporal {temp_dir} eliminado.")

    # --- FIN IMPLEMENTACIÓN OPCIÓN C ---

    return {
        'model_name': model_name,
        'pipeline': pipeline,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'run_id': run_id,
        'training_time': training_time
    }

# ============================================================================
# MAIN WORKFLOW (Código modificado solo con la corrección de la ruta)
# ============================================================================

def main():
    """Función principal."""
    print("\n" + "="*70)
    print("  FASE 2: Experimentación con Optuna + MLflow")
    print("="*70 + "\n")

    # Cargar y preprocesar datos usando DataPipeline
    data = load_and_preprocess_data(config_path="config.yaml")

    print(f"✅ Datos preprocesados:")
    print(f"   - Filas: {len(data)}")
    print(f"   - Columnas: {list(data.columns)}")

    # Metadata para trazabilidad
    metadata = {
        'git_commit': get_git_commit_hash(),
        'git_branch': get_git_branch(),
        'data_hash': calculate_data_hash(data),
        'data_rows': len(data),
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n🔍 Metadata:")
    print(f"   - Git Commit: {metadata['git_commit']}")
    print(f"   - Data Hash: {metadata['data_hash']}")

    # Split datos
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\n📊 Split de datos:")
    print(f"   - Train: {len(X_train)} samples")
    print(f"   - Test:  {len(X_test)} samples")
    
    # --- CORRECCIÓN DE RUTA DE TRACKING URI (para doble seguridad) ---
    current_working_directory = os.getcwd()

    # Configurar MLflow
    #mlflow.set_tracking_uri(f"file:{current_working_directory}/mlruns")
    #mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n✅ MLflow configurado: {EXPERIMENT_NAME}")
    print(f"\n✅ MLflow configurado: {EXPERIMENT_NAME}")
    print(f"   (Ruta de Tracking: {mlflow.get_tracking_uri()})")
    # --- FIN CORRECCIÓN ---

    # Modelos a optimizar
    models_to_optimize = ["RandomForest", "GradientBoosting", "XGBoost"]

    all_results = []

    # Optimizar y entrenar cada modelo
    for model_name in models_to_optimize:
        print(f"\n{'█'*70}")
        print(f"  MODELO: {model_name}")
        print(f"{'█'*70}")

        best_params, study = optimize_model(model_name, X_train, y_train, n_trials=OPTUNA_TRIALS)

        results = train_optimized_model(
            model_name=model_name,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=FEATURES,
            metadata=metadata
        )

        plots_dir = Path("plots_temp")
        plots_dir.mkdir(exist_ok=True)
        opt_history_path = plots_dir / f"{model_name}_optimization_history.png"
        create_optimization_history_plot(study, opt_history_path)

        with mlflow.start_run(run_id=results['run_id']):
            mlflow.log_artifact(opt_history_path)

        

        all_results.append(results)

    # Comparar modelos
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE TODOS LOS MODELOS")
    print(f"{'='*70}\n")

    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'CV R2': f"{result['cv_results']['mean_r2']:.4f}",
            'Test R2': f"{result['test_metrics']['r2_score']:.4f}",
            'Test MAE': f"{result['test_metrics']['mae']:.2f}",
            'Test RMSE': f"{result['test_metrics']['rmse']:.2f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Seleccionar mejor modelo
    best_result = max(all_results, key=lambda x: x['cv_results']['mean_r2'])

    print(f"\n🏆 MEJOR MODELO: {best_result['model_name']}")
    print(f"   - CV R2: {best_result['cv_results']['mean_r2']:.4f}")
    print(f"   - Test R2: {best_result['test_metrics']['r2_score']:.4f}")

    # Guardar mejor modelo usando rutas del config
    

    # Crear directorio si no existe
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Guardando modelo en: {MODEL_PATH}")
    joblib.dump(best_result['pipeline'], MODEL_PATH)

    metadata_dict = {
        "model_name": best_result['model_name'],
        "cv_r2_mean": best_result['cv_results']['mean_r2'],
        "cv_r2_std": best_result['cv_results']['std_r2'],
        "test_r2": best_result['test_metrics']['r2_score'],
        "test_mae": best_result['test_metrics']['mae'],
        "test_rmse": best_result['test_metrics']['rmse'],
        "mlflow_run_id": best_result['run_id'],
        "feature_names": FEATURES,
        "git_commit": metadata['git_commit'],
        "data_hash": metadata['data_hash'],
        "trained_at": metadata['timestamp']
    }

    print(f"💾 Guardando metadata en: {METADATA_PATH}")
    joblib.dump(metadata_dict, METADATA_PATH)

    print(f"\n✅ Modelo guardado exitosamente:")
    print(f"   - Modelo: {MODEL_PATH}")
    print(f"   - Metadata: {METADATA_PATH}")

    print(f"\n{'='*70}")
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()