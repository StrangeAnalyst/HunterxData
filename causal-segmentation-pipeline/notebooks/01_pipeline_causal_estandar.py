# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Causal Estandarizado — De Datos a Estrategia
# MAGIC
# MAGIC **Que hace:** identifica que acciones del banco hacen que un cliente **evolucione**
# MAGIC de segmento (y cuales lo protegen de **caer**), las valida con backtesting historico,
# MAGIC segmenta la base en una **matriz de inversion de 4 cuadrantes** y produce estrategias
# MAGIC priorizadas por ROI listas para el motor NBA.
# MAGIC
# MAGIC **Como usarlo en otro caso de uso:** crear un YAML en `config/` (copiar `_template.yaml`),
# MAGIC apuntar el widget `config_path` y ejecutar todo. **No se toca codigo.**
# MAGIC
# MAGIC | Fase | Que hace | Pregunta de negocio |
# MAGIC |------|----------|---------------------|
# MAGIC | 1. Datos | Carga snapshot actual + historia | ¿Con que trabajamos? |
# MAGIC | 2. Integridad | Detecta leakage | ¿Podemos confiar en los datos? |
# MAGIC | 3. Descubrimiento | LightGBM + SHAP triangulado | ¿Que variables SE ASOCIAN a evolucionar? |
# MAGIC | 4. Causalidad | Double ML + placebo | ¿Cuales realmente CAUSAN la evolucion? |
# MAGIC | 5. Backtesting | Estabilidad temporal + cuadrantes vs realidad + sensibilidad | ¿Es verdad lo que dice el analisis? |
# MAGIC | 6. Cuadrantes | Matriz de inversion (X=P(evolucion), Y=tau) | ¿En quien invertir? |
# MAGIC | 7. Estrategia | Brechas, prioridad ROI, playbook, A/B | ¿Que hacemos exactamente? |
# MAGIC | 8. Salidas | Tablas Delta + resumen ejecutivo | ¿Como lo consume NBA? |

# COMMAND ----------

# MAGIC %pip install "econml>=0.15" lightgbm shap "statsmodels>=0.14.4" pyyaml --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# WIDGETS: la unica entrada del pipeline es el archivo de config
# ============================================================
dbutils.widgets.text('config_path', '../config/pyme_personas.yaml', 'Ruta al config YAML')

# COMMAND ----------

# ============================================================
# FASE 0: CARGA DE CONFIGURACION
# ============================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# El paquete vive en ../src relativo a este notebook (repo de Databricks)
_repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .notebookPath().get().lstrip('/')))
for cand in [f'/Workspace/{_repo_root}/src', os.path.abspath('../src')]:
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from causal_pipeline.config import cargar_config, resumen_config
from causal_pipeline import data as cp_data
from causal_pipeline import integrity as cp_integrity
from causal_pipeline import discovery as cp_discovery
from causal_pipeline import causal as cp_causal
from causal_pipeline import backtesting as cp_backtesting
from causal_pipeline import quadrants as cp_quadrants
from causal_pipeline import strategy as cp_strategy
from causal_pipeline import outputs as cp_outputs

CONFIG_PATH = dbutils.widgets.get('config_path')
cfg = cargar_config(CONFIG_PATH)
resumen_config(cfg)

if cfg.get('mlflow', {}).get('activo'):
    import mlflow
    mlflow.set_experiment(cfg['mlflow']['experimento'])

# COMMAND ----------

# ============================================================
# FASE 1: DATOS — snapshot actual + historia para backtesting
# ============================================================
df, vars_info = cp_data.preparar_dataset(spark, cfg)

historia = {}
if cfg['backtesting'].get('activo', True):
    print('\n[DATOS] Cargando historia para backtesting...')
    historia = cp_data.cargar_historia(spark, cfg)
    historia = cp_backtesting.preparar_historia(historia, cfg)

PARES_EVOL = cfg['_derivados']['pares_evolucion']
PARES_INVOL = cfg['_derivados']['pares_involucion']

# COMMAND ----------

# ============================================================
# FASE 2: INTEGRIDAD — leakage antes de modelar
# ============================================================
df_integridad = cp_integrity.validar_integridad(df, cfg, vars_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EVOLUCION — ¿Que hace que un cliente SUBA de segmento?

# COMMAND ----------

# ============================================================
# FASE 3: DESCUBRIMIENTO — candidatas trianguladas por dimension
# (esto todavia es CORRELACION, no causalidad)
# ============================================================
descubrimiento = cp_discovery.descubrir_candidatas(df, cfg, vars_info, PARES_EVOL, 'evolucion')

# COMMAND ----------

# ============================================================
# FASE 4: CAUSALIDAD — Double ML + placebo sobre las candidatas
# ============================================================
df_causal = cp_causal.validar_causalmente(descubrimiento, cfg, vars_info, 'evolucion')
df_causal = cp_causal.impacto_practico(df_causal, descubrimiento)

if len(df_causal) > 0:
    print('\nImpacto practico de las validadas (ATE x IQR, en puntos porcentuales):')
    for _, r in df_causal[df_causal['validada']].iterrows():
        print(f"  {r['par']:28s} | {r['palanca']:35s} | {r['impacto_pp']:+.2f} pp")

# COMMAND ----------

# ============================================================
# FASE 5: BACKTESTING — ¿es verdad lo que dice el analisis causal?
# Pilar 1: ¿el efecto existe en TODOS los periodos o solo en este snapshot?
# Pilar 3: ¿el modelo responde en la misma direccion que el ATE?
# (El Pilar 2 — cuadrantes vs realidad — corre despues de segmentar)
# ============================================================
if cfg['backtesting'].get('activo', True) and len(historia) >= cfg['backtesting']['min_periodos']:
    df_estabilidad = cp_backtesting.backtest_estabilidad(historia, cfg, vars_info, df_causal, PARES_EVOL)
    df_sensibilidad = cp_backtesting.sensibilidad_direccional(descubrimiento, df_estabilidad, cfg)
else:
    print('[BACKTESTING] Desactivado o historia insuficiente: las palancas quedaran EN_OBSERVACION')
    df_estabilidad, df_sensibilidad = pd.DataFrame(), pd.DataFrame()

df_veredicto = cp_backtesting.veredicto_final(df_causal, df_estabilidad, df_sensibilidad)

# COMMAND ----------

# ============================================================
# FASE 6: CUADRANTES — matriz de inversion
#   X = P(evolucion): ¿tiene probabilidad de moverse?
#   Y = tau:          ¿la intervencion lo acelera?
#
#   Persuadable  (P alta, tau alto) -> INVERTIR PRIMERO
#   Sure-thing   (P alta, tau bajo) -> evoluciona solo, no gastar
#   Sleeping-dog (P baja, tau alto) -> mucho impacto, baja probabilidad hoy
#   Lost-cause   (P baja, tau bajo) -> no invertir
# ============================================================
df_scores, info_pares = cp_quadrants.segmentar_clientes(descubrimiento, df_causal, cfg, vars_info)

fig = cp_quadrants.graficar_cuadrantes(df_scores, info_pares,
                                       f"Matriz de Inversion — {cfg['caso_uso']}")
if fig is not None:
    plt.show()

# COMMAND ----------

# ============================================================
# FASE 5b (Pilar 2): CUADRANTES vs REALIDAD T -> T+1
# ¿Los clientes con P alta realmente subieron mas que los de P baja?
# ============================================================
validacion_t1 = None
if cfg['backtesting'].get('validar_cuadrantes_t1', True) and len(historia) >= 2:
    validacion_t1 = cp_backtesting.validar_cuadrantes_t1(historia, cfg, df_scores)

# COMMAND ----------

# ============================================================
# FASE 7: ESTRATEGIA — de palancas aprobadas a acciones concretas
# ============================================================
df_gap = cp_strategy.brecha_vs_objetivo(df, cfg, df_veredicto, descubrimiento)
df_prioridad = cp_strategy.priorizar(df_scores, df_gap, cfg, descubrimiento)
df_playbook = cp_strategy.playbook(df_scores, df_veredicto, info_pares, cfg)
df_ab = cp_strategy.disenar_ab_test(df_scores, cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## INVOLUCION — ¿Que hace que un cliente CAIGA de segmento?
# MAGIC
# MAGIC Mismo motor, transiciones invertidas. La lectura cambia:
# MAGIC - **FACTOR DE RIESGO** (ATE > 0): cuanto mas de esto, mas probable la caida.
# MAGIC - **FACTOR PROTECTOR** (ATE < 0): mantenerlo alto previene la caida -> alertas tempranas.

# COMMAND ----------

df_invol_causal, df_invol_scores = pd.DataFrame(), pd.DataFrame()

if cfg.get('involucion', {}).get('activo', True):
    descubrimiento_inv = cp_discovery.descubrir_candidatas(df, cfg, vars_info, PARES_INVOL, 'involucion')
    df_invol_causal = cp_causal.validar_causalmente(descubrimiento_inv, cfg, vars_info, 'involucion')
    df_invol_causal = cp_causal.impacto_practico(df_invol_causal, descubrimiento_inv)

    # Matriz de riesgo: misma mecanica, etiquetas de retencion
    df_invol_scores, info_inv = cp_quadrants.segmentar_clientes(
        descubrimiento_inv, df_invol_causal, cfg, vars_info)
    if len(df_invol_scores) > 0:
        df_invol_scores = df_invol_scores.rename(columns={
            'p_evolucion': 'p_involucion', 'cuadrante': 'cuadrante_riesgo'})
        df_invol_scores['cuadrante_riesgo'] = df_invol_scores['cuadrante_riesgo'].map({
            'Persuadable': 'Alto-Riesgo',          # P caida alta + palanca con impacto -> retener YA
            'Sure-thing': 'Riesgo-Estructural',    # va a caer y la palanca no lo frena
            'Sleeping-dog': 'Riesgo-Latente',      # hoy estable pero sensible a la palanca
            'Lost-cause': 'Seguro',
        })
        print('\n[INVOLUCION] Cuadrantes de riesgo:')
        print(df_invol_scores['cuadrante_riesgo'].value_counts().to_string())

# COMMAND ----------

# ============================================================
# FASE 8: SALIDAS — tablas Delta + resumen ejecutivo
# ============================================================
resultados = {
    'df_veredicto': df_veredicto,
    'df_prioridad': df_prioridad,
    'df_scores': df_scores,
    'df_gap': df_gap,
    'df_playbook': df_playbook,
    'df_ab': df_ab,
    'df_estabilidad': df_estabilidad.drop(columns=['historial'], errors='ignore'),
    'df_invol_causal': df_invol_causal,
    'df_invol_scores': df_invol_scores,
    'validacion_t1': validacion_t1,
}
cp_outputs.guardar_resultados(spark, cfg, resultados)

# COMMAND ----------

displayHTML(cp_outputs.resumen_ejecutivo_html(cfg, resultados))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Como leer los resultados (guia rapida)
# MAGIC
# MAGIC 1. **`causal_palancas_*`** — la columna `veredicto` es la unica que importa para negocio:
# MAGIC    - `APROBADA`: causal en el snapshot + estable en la historia + direccion coherente. **Usable en campanas.**
# MAGIC    - `EN_OBSERVACION`: prometedora pero sin suficiente evidencia. No construir campanas todavia.
# MAGIC    - `RECHAZADA`: parecia palanca pero el backtesting la refuto. **Ignorar aunque SHAP la rankee alto.**
# MAGIC 2. **`causal_scores_clientes_*`** — por cliente: `cuadrante` dice que hacer,
# MAGIC    `score_prioridad` dice en que orden (mayor = mas valor por peso invertido).
# MAGIC 3. **`causal_prescripciones_*`** — la brecha concreta: "a este cliente le faltan X
# MAGIC    unidades de la palanca Y para parecerse al segmento objetivo".
# MAGIC 4. **`causal_ab_assignment_*`** — el experimento que CONFIRMA el tau en campo.
# MAGIC    Hasta que el A/B cierre, el eje vertical de la matriz es una estimacion.
