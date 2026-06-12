"""
Pipeline Causal Estandarizado de Segmentacion.

Identifica que palancas ACCIONABLES mueven clientes entre segmentos de valor,
las valida causalmente (DML + placebo + backtesting temporal), segmenta la
base en una matriz de inversion de 4 cuadrantes y genera estrategias
priorizadas por ROI listas para el motor NBA.

Todo el comportamiento se controla via un archivo YAML (ver config/).
"""

__version__ = "2.0.0"

from .config import cargar_config  # noqa: F401
