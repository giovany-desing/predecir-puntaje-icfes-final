"""
  Módulo para cargar configuración centralizada.
  TODA la configuración se lee desde config.yaml.
  """
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Ruta al config.yaml (desde la raíz del proyecto)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class Config:
    """Clase singleton para manejar la configuración del proyecto."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """Cargar configuración desde config.yaml"""
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"No se encontró config.yaml en {CONFIG_PATH}")

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
              cls._config = yaml.safe_load(f)

    # Propiedades para acceso fácil
    @property
    def features(self) -> List[str]:
        """Lista de features para el modelo"""
        return self._config['data']['features']

    @property
    def target_col(self) -> str:
        """Nombre de la columna target"""
        return self._config['data']['target_col']

    @property
    def model_path(self) -> Path:
        """Ruta al archivo del modelo"""
        return Path(self._config['model']['path'])

    @property
    def metadata_path(self) -> Path:
        """Ruta al archivo de metadata del modelo"""
        return Path(self._config['model']['metadata_path'])

    @property
    def raw_data_path(self) -> Path:
        """Ruta al dataset crudo"""
        return Path(self._config['data']['raw_path'])

    @property
    def processed_data_path(self) -> Path:
        """Ruta al dataset procesado"""
        return Path(self._config['data']['processed_path'])

    def get(self, key: str, default=None) -> Any:
        """
        Acceso genérico a cualquier valor en config usando notación de puntos.
        Ejemplo: config.get('training.test_size') → 0.2
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
            if value is None:
                return default
        return value


# Instancia global para importar en otros módulos
config = Config()