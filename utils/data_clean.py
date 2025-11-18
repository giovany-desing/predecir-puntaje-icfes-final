"""Script para limpiar y preprocesar la data de entrenamiento, esta data se guardara en el path data/raw/data_train.csv"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import json


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline de preparación de datos para ICFES"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el pipeline

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.data_path = self.config['data']['raw_path']
        self.features = self.config['data']['features']
        self.target = self.config['data']['target_col']
        self.required_columns = self.features + [self.target]

        # Métricas del pipeline
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(self.data_path),
            'pipeline_version': '1.0.0'
        }

    def _load_config(self, config_path: str) -> Dict:
        """Carga configuración desde YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuración cargada desde {config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ Error al cargar configuración: {e}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        """Carga datos crudos"""
        logger.info(f"\n{'='*60}")
        logger.info("CARGANDO DATOS CRUDOS")
        logger.info(f"{'='*60}")

        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"✓ Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

            self.metrics['raw_rows'] = len(df)
            self.metrics['raw_columns'] = len(df.columns)

            return df
        except Exception as e:
            logger.error(f"✗ Error al cargar datos: {e}")
            raise

    def select_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona solo las columnas necesarias"""
        logger.info(f"\n{'='*60}")
        logger.info("SELECCIONANDO COLUMNAS REQUERIDAS")
        logger.info(f"{'='*60}")

        # Verificar que existan las columnas
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")

        df_selected = df[self.required_columns].copy()
        logger.info(f"✓ Seleccionadas {len(self.required_columns)} columnas")
        logger.info(f"  Columnas: {', '.join(self.required_columns)}")

        return df_selected

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve filas duplicadas"""
        logger.info(f"\n{'='*60}")
        logger.info("REMOVIENDO DUPLICADOS")
        logger.info(f"{'='*60}")

        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        final_rows = len(df_clean)
        removed = initial_rows - final_rows

        logger.info(f"  Filas iniciales: {initial_rows:,}")
        logger.info(f"  Filas finales: {final_rows:,}")
        logger.info(f"  Duplicados removidos: {removed:,} ({removed/initial_rows*100:.2f}%)")

        self.metrics['duplicates_removed'] = removed
        self.metrics['duplicates_percentage'] = round(removed/initial_rows*100, 2)

        return df_clean

    def remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve filas con valores nulos"""
        logger.info(f"\n{'='*60}")
        logger.info("REMOVIENDO VALORES NULOS")
        logger.info(f"{'='*60}")

        initial_rows = len(df)

        # Mostrar nulos por columna
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info("  Valores nulos por columna:")
            for col in self.required_columns:
                if null_counts[col] > 0:
                    logger.info(f"    - {col}: {null_counts[col]:,}")

        # Remover nulos
        df_clean = df.dropna()
        final_rows = len(df_clean)
        removed = initial_rows - final_rows

        logger.info(f"\n  Filas iniciales: {initial_rows:,}")
        logger.info(f"  Filas finales: {final_rows:,}")
        logger.info(f"  Nulos removidos: {removed:,} ({removed/initial_rows*100:.2f}%)")

        self.metrics['nulls_removed'] = removed
        self.metrics['nulls_percentage'] = round(removed/initial_rows*100, 2)

        return df_clean

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida y filtra valores fuera de rango"""
        logger.info(f"\n{'='*60}")
        logger.info("VALIDANDO RANGOS")
        logger.info(f"{'='*60}")

        initial_rows = len(df)

        # Rangos válidos
        feature_min, feature_max = 0, 100
        target_min, target_max = 0, 500

        # Filtrar features (0-100)
        for col in self.features:
            invalid_mask = (df[col] < feature_min) | (df[col] > feature_max)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logger.info(f"  '{col}': {invalid_count} valores fuera de rango [{feature_min}, {feature_max}]")
            df = df[~invalid_mask]

        # Filtrar target (0-500)
        invalid_mask = (df[self.target] < target_min) | (df[self.target] > target_max)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.info(f"  '{self.target}': {invalid_count} valores fuera de rango [{target_min}, {target_max}]")
        df = df[~invalid_mask]

        final_rows = len(df)
        removed = initial_rows - final_rows

        logger.info(f"\n  Filas iniciales: {initial_rows:,}")
        logger.info(f"  Filas finales: {final_rows:,}")
        logger.info(f"  Fuera de rango removidos: {removed:,} ({removed/initial_rows*100:.2f}%)")

        self.metrics['out_of_range_removed'] = removed
        self.metrics['out_of_range_percentage'] = round(removed/initial_rows*100, 2) if initial_rows > 0 else 0

        return df

    def reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resetea el índice después de la limpieza"""
        return df.reset_index(drop=True)

    def show_data_summary(self, df: pd.DataFrame):
        """Muestra resumen estadístico de datos limpios"""
        logger.info(f"\n{'='*60}")
        logger.info("RESUMEN DE DATOS LIMPIOS")
        logger.info(f"{'='*60}")

        logger.info(f"\nTotal de registros: {len(df):,}")
        logger.info(f"Total de columnas: {len(df.columns)}")

        logger.info(f"\nEstadísticas descriptivas:")
        stats = df.describe()

        for col in self.required_columns:
            logger.info(f"\n{col}:")
            logger.info(f"  Media: {stats.loc['mean', col]:.2f}")
            logger.info(f"  Mediana: {stats.loc['50%', col]:.2f}")
            logger.info(f"  Std: {stats.loc['std', col]:.2f}")
            logger.info(f"  Min: {stats.loc['min', col]:.2f}")
            logger.info(f"  Max: {stats.loc['max', col]:.2f}")

        # Guardar estadísticas en métricas
        self.metrics['clean_rows'] = len(df)
        self.metrics['clean_columns'] = len(df.columns)
        self.metrics['statistics'] = stats.to_dict()

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo

        Returns:
            DataFrame con los datos procesados
        """
        logger.info("="*60)
        logger.info("EJECUTANDO PIPELINE DE DATOS")
        logger.info("="*60)

        # 1. Cargar datos crudos
        df = self.load_raw_data()

        # 2. Seleccionar columnas requeridas
        df = self.select_required_columns(df)

        # 3. Remover duplicados
        df = self.remove_duplicates(df)

        # 4. Remover nulos
        df = self.remove_nulls(df)

        # 5. Validar rangos
        df = self.validate_ranges(df)

        # 6. Resetear índice
        df = self.reset_index(df)

        # 7. Mostrar resumen
        self.show_data_summary(df)

        # Resumen final
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(f"{'='*60}")
        logger.info(f"Filas procesadas: {self.metrics['raw_rows']:,} → {self.metrics['clean_rows']:,}")
        logger.info(f"Reducción: {self.metrics['raw_rows'] - self.metrics['clean_rows']:,} filas "
                   f"({(self.metrics['raw_rows'] - self.metrics['clean_rows'])/self.metrics['raw_rows']*100:.2f}%)")
        logger.info(f"{'='*60}\n")

        return df


def main():
    """Función principal"""
    pipeline = DataPipeline(config_path="config.yaml")
    df_clean = pipeline.run()

    logger.info(f"✓ Pipeline finalizado. Datos limpios listos para usar")
    return df_clean


if __name__ == "__main__":
    df_processed = main()
    
    print(df_processed.head(5))