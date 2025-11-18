"""Este script se encargara de validar la data de entrenamiento del path data/raw/data_train.csv que todo este ok antes de hacer el entrenamiento del modelo"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Clase para validar datasets de ICFES"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el validador cargando la configuración

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.data_path = self.config['data']['raw_path']
        self.features = self.config['data']['features']
        self.target = self.config['data']['target_col']
        self.required_columns = self.features + [self.target]

        # Rangos válidos para puntajes ICFES (0-100 por materia, 0-500 global)
        self.feature_min = 0
        self.feature_max = 100
        self.target_min = 0
        self.target_max = 500

    def _load_config(self, config_path: str) -> Dict:
        """Carga el archivo de configuración YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuración cargada desde {config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ Error al cargar configuración: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Carga el dataset desde el archivo CSV"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            logger.error(f"✗ Error al cargar datos: {e}")
            raise

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Valida que existan todas las columnas requeridas"""
        logger.info("\n=== Validación de Columnas ===")
        missing_cols = set(self.required_columns) - set(df.columns)

        if missing_cols:
            logger.error(f"✗ Columnas faltantes: {missing_cols}")
            return False
        else:
            logger.info(f"✓ Todas las columnas requeridas están presentes ({len(self.required_columns)} columnas)")
            for col in self.required_columns:
                logger.info(f"  - {col}")
            return True

    def validate_dtypes(self, df: pd.DataFrame) -> bool:
        """Valida que las columnas numéricas tengan el tipo correcto"""
        logger.info("\n=== Validación de Tipos de Datos ===")
        all_valid = True

        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"✗ Columna '{col}' no es numérica (tipo: {df[col].dtype})")
                all_valid = False
            else:
                logger.info(f"✓ Columna '{col}' es numérica (tipo: {df[col].dtype})")

        return all_valid

    def validate_nulls(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Valida la presencia de valores nulos"""
        logger.info("\n=== Validación de Valores Nulos ===")
        null_counts = df[self.required_columns].isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)

        has_nulls = null_counts.sum() > 0

        if has_nulls:
            logger.warning(f"⚠ Encontrados valores nulos:")
            for col in self.required_columns:
                if null_counts[col] > 0:
                    logger.warning(f"  - {col}: {null_counts[col]} ({null_percentages[col]}%)")
        else:
            logger.info(f"✓ No hay valores nulos en las columnas requeridas")

        return not has_nulls, null_counts.to_dict()

    def validate_ranges(self, df: pd.DataFrame) -> bool:
        """Valida que los puntajes estén en rangos válidos"""
        logger.info("\n=== Validación de Rangos ===")
        all_valid = True

        # Validar features (0-100)
        for col in self.features:
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val < self.feature_min or max_val > self.feature_max:
                logger.error(
                    f"✗ '{col}' fuera de rango [{self.feature_min}, {self.feature_max}]: "
                    f"min={min_val}, max={max_val}"
                )
                all_valid = False
            else:
                logger.info(f"✓ '{col}' en rango válido: [{min_val}, {max_val}]")

        # Validar target (0-500)
        target_min = df[self.target].min()
        target_max = df[self.target].max()

        if target_min < self.target_min or target_max > self.target_max:
            logger.error(
                f"✗ '{self.target}' fuera de rango [{self.target_min}, {self.target_max}]: "
                f"min={target_min}, max={target_max}"
            )
            all_valid = False
        else:
            logger.info(f"✓ '{self.target}' en rango válido: [{target_min}, {target_max}]")

        return all_valid

    def validate_duplicates(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """Detecta filas duplicadas"""
        logger.info("\n=== Validación de Duplicados ===")
        n_duplicates = df[self.required_columns].duplicated().sum()
        duplicate_percentage = (n_duplicates / len(df) * 100).round(2)

        if n_duplicates > 0:
            logger.warning(
                f"⚠ Encontrados {n_duplicates} duplicados ({duplicate_percentage}%)"
            )
            return False, n_duplicates
        else:
            logger.info(f"✓ No hay duplicados")
            return True, 0

    def show_statistics(self, df: pd.DataFrame):
        """Muestra estadísticas descriptivas del dataset"""
        logger.info("\n=== Estadísticas Descriptivas ===")
        stats = df[self.required_columns].describe()

        logger.info(f"\nTotal de registros: {len(df):,}")
        logger.info(f"\nEstadísticas por columna:")
        for col in self.required_columns:
            logger.info(f"\n{col}:")
            logger.info(f"  Media: {stats.loc['mean', col]:.2f}")
            logger.info(f"  Mediana: {stats.loc['50%', col]:.2f}")
            logger.info(f"  Std Dev: {stats.loc['std', col]:.2f}")
            logger.info(f"  Min: {stats.loc['min', col]:.2f}")
            logger.info(f"  Max: {stats.loc['max', col]:.2f}")

    def run_all_validations(self) -> Dict:
        """
        Ejecuta todas las validaciones y retorna un reporte

        Returns:
            Dict con resultados de todas las validaciones
        """
        logger.info("="*60)
        logger.info("INICIANDO VALIDACIÓN DE DATOS ICFES")
        logger.info("="*60)

        # Cargar datos
        df = self.load_data()

        # Ejecutar validaciones
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_valid': self.validate_columns(df),
            'dtypes_valid': self.validate_dtypes(df),
            'nulls_valid': self.validate_nulls(df)[0],
            'null_counts': self.validate_nulls(df)[1],
            'ranges_valid': self.validate_ranges(df),
            'duplicates_valid': self.validate_duplicates(df)[0],
            'n_duplicates': self.validate_duplicates(df)[1]
        }

        # Mostrar estadísticas
        self.show_statistics(df)

        # Resumen final
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE VALIDACIÓN")
        logger.info("="*60)

        all_passed = all([
            results['columns_valid'],
            results['dtypes_valid'],
            results['ranges_valid']
        ])

        if all_passed:
            logger.info("✓ TODAS LAS VALIDACIONES CRÍTICAS PASARON")
        else:
            logger.error("✗ ALGUNAS VALIDACIONES FALLARON - REVISAR LOGS")

        if not results['nulls_valid']:
            logger.warning("⚠ Hay valores nulos que deben ser limpiados")

        if not results['duplicates_valid']:
            logger.warning("⚠ Hay duplicados que deben ser removidos")

        logger.info("="*60)

        return results


def main():
    """Función principal"""
    validator = DataValidator(config_path="config.yaml")
    results = validator.run_all_validations()

    # Retornar código de salida basado en validaciones críticas
    if all([results['columns_valid'], results['dtypes_valid'], results['ranges_valid']]):
        exit(0)  # Éxito
    else:
        exit(1)  # Fallo


if __name__ == "__main__":
    main()