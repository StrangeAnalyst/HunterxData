# Pipeline Causal Estandarizado de Segmentación

**De datos → a palancas validadas → a estrategias accionables.**

Este proyecto responde tres preguntas de negocio para cualquier portafolio de
clientes segmentado por valor (Pyme Personas, Banca Premium, PyME RNC, etc.):

1. **¿Qué acciones del banco hacen que un cliente SUBA de segmento?** (evolución)
2. **¿Qué señales nos dicen que un cliente va a CAER?** (involución)
3. **¿En qué clientes invertir primero para maximizar el retorno?** (priorización)

Y lo hace separando **correlación de causalidad**: un modelo predictivo puede decir
que "los clientes top usan más crédito", pero eso no prueba que dar crédito convierta
a un cliente en top. Este pipeline solo recomienda palancas que pasaron **tres filtros
de verdad**: efecto causal (Double ML), prueba placebo y **backtesting histórico**.

---

## La idea central: la Matriz de Inversión (4 cuadrantes)

Cada cliente se ubica en un plano con dos ejes **de negocio**:

- **Eje X — P(evolución):** ¿qué tan probable es que este cliente suba de segmento?
- **Eje Y — tau (impacto causal):** si el banco interviene (con la palanca validada),
  ¿cuánto **acelera** eso su evolución?

```
                        tau ALTO (la intervención mueve la aguja)
                                 │
        SLEEPING-DOG             │            PERSUADABLE
        Mucho impacto, pero hoy  │            Puede evolucionar Y la
        NO tiene probabilidad    │            intervención lo acelera
        de moverse               │
        → apuesta selectiva,     │            → INVERTIR PRIMERO
          construir condiciones  │              (máximo ROI de campaña)
   ──────────────────────────────┼──────────────────────────────────→  P(evolución)
        LOST-CAUSE               │            SURE-THING
        Ni probabilidad ni       │            Va a evolucionar SOLO,
        respuesta al estímulo    │            el empujón no agrega casi nada
        → no invertir            │            → no gastar; monitorear y
                                 │              proteger (retención)
```

**Lectura en una frase:** el lado **derecho** (Persuadable + Sure-thing) son los que
*pueden* evolucionar; de ellos, el **Persuadable** es donde la intervención tiene más
impacto y lo hace evolucionar **más rápido** — ahí va el presupuesto. El
**Sleeping-dog** es el caso contrario: la palanca le impactaría mucho, pero hoy no
tiene probabilidad de moverse — es una apuesta de mediano plazo, no campaña masiva.

Para **involución** la misma matriz se relee como riesgo:
`Alto-Riesgo` (caída probable y la palanca puede frenarla → retener YA),
`Riesgo-Estructural` (va a caer y la palanca no lo frena), `Riesgo-Latente`
(estable hoy pero sensible), `Seguro`.

---

## Cómo funciona (8 fases)

| Fase | Qué hace | Por qué importa |
|------|----------|-----------------|
| **1. Datos** | Carga el snapshot actual + N períodos de historia desde Delta. Codifica categóricas (canales, MCC) para que también compitan como palancas. | Sin historia no hay backtesting; sin encoding se pierden dimensiones completas. |
| **2. Integridad** | Tests de leakage (AUC sospechoso, ablación top-20, variables circulares). | Modelar sobre leakage produce "palancas" falsas. |
| **3. Descubrimiento** | LightGBM por par de transición + triangulación de importancia (SHAP + Gain + Permutation → ranking Borda), **estratificada por dimensión de negocio** (mínimo 2 variables por área: productos, canales, mora…). | Genera candidatas sin puntos ciegos. **Esto todavía es correlación.** |
| **4. Causalidad** | Double ML (LinearDML) estima el efecto de cada candidata controlando por todas las demás + contexto. Prueba placebo: con el outcome permutado, el efecto debe desaparecer. | Separa "se asocia con subir" de "hace que suban". Detecta efectos espurios y paradojas de Simpson (palancas que invierten su signo según la transición). |
| **5. Backtesting** ★ | **Pilar 1:** re-estima el ATE en cada período histórico → `CONFIRMADA` (≥60% de períodos significativa con el mismo signo), `INESTABLE` (30-60%), `REFUTADA` (<30%). **Pilar 2:** clasifica con datos de T y verifica contra quién subió de verdad en T+1 (los cuadrantes de P alta deben transitar más; curva Qini sobre tau). **Pilar 3:** mueve cada palanca ±IQR y verifica que la respuesta del modelo vaya en la misma dirección que el ATE. | Es la **auditoría integrada**: confirma que lo que dice el análisis causal es verdad y no un artefacto del snapshot. |
| **6. Cuadrantes** | Calcula P(evolución) (modelo predictivo) y tau individual (X-learner) y asigna la matriz de inversión. | Convierte estadística en una decisión simple por cliente. |
| **7. Estrategia** | (a) **Brecha vs segmento objetivo:** cuánto le falta a cada Persuadable en cada palanca aprobada para parecerse a la mediana del segmento destino. (b) **Score de prioridad** = tau × ΔLTV / costo. (c) **Playbook** por transición y cuadrante. (d) **Diseño A/B** (poder estadístico + asignación Treatment/Control/Holdout). | El propósito final: estrategias concretas, priorizadas y verificables en campo. |
| **8. Salidas** | Tablas Delta + resumen ejecutivo HTML. | Consumo directo por NBA, dashboards y campañas. |

★ **Veredicto final por palanca** (combina los 3 pilares):

- `APROBADA` = causal + estable en el tiempo + dirección coherente → **usable en campañas**
- `EN_OBSERVACION` = prometedora pero evidencia insuficiente → esperar más historia o A/B
- `RECHAZADA` = el backtesting la refutó → **ignorar aunque el modelo predictivo la rankee alto**

---

## Cómo correrlo en un nuevo caso de uso (3 pasos)

Todo está parametrizado en un **YAML**. No se toca código.

```bash
# 1. Copiar la plantilla
cp config/_template.yaml config/banca_premium.yaml

# 2. Completar: tabla, filtros, orden de segmentos, LTV por segmento,
#    variables no accionables, controles, dimensiones, períodos históricos
```

```text
# 3. En Databricks: abrir notebooks/01_pipeline_causal_estandar.py,
#    poner el widget config_path = ../config/banca_premium.yaml y Run All
```

Requisitos mínimos del caso de uso:

- Una tabla Delta con **1 fila por cliente-período**, una columna de segmento
  (ordenable de menor a mayor valor) y una columna de período.
- **≥ 3 períodos históricos** (idealmente 6) para que el backtesting pueda
  clasificar palancas. Con menos historia el pipeline corre, pero todo queda
  `EN_OBSERVACION`.

---

## Estructura del proyecto

```
causal-segmentation-pipeline/
├── README.md                        ← este archivo
├── config/
│   ├── _template.yaml               ← plantilla comentada para nuevos casos
│   └── pyme_personas.yaml           ← ejemplo real (Pyme Personas)
├── src/causal_pipeline/             ← lógica reutilizable (no se toca por caso de uso)
│   ├── config.py                    ← carga y validación del YAML (falla rápido)
│   ├── data.py                      ← carga Delta, encoding, accionables vs controles
│   ├── integrity.py                 ← detección de leakage
│   ├── discovery.py                 ← LightGBM + triangulación SHAP/Gain/Permutation
│   ├── causal.py                    ← Double ML + placebo + dosis-respuesta
│   ├── backtesting.py               ← los 3 pilares de validación + veredicto final
│   ├── quadrants.py                 ← matriz de inversión (definición ÚNICA de cuadrantes)
│   ├── strategy.py                  ← brechas, prioridad ROI, playbook, diseño A/B
│   └── outputs.py                   ← tablas Delta + resumen ejecutivo HTML
├── notebooks/
│   └── 01_pipeline_causal_estandar.py   ← driver Databricks (único notebook a ejecutar)
└── tools/
    └── build_bundle.py              ← genera el bundle .txt para compartir por correo
```

## Tablas de salida (sufijo = `salidas.sufijo` del config)

| Tabla | Grano | Para qué |
|-------|-------|----------|
| `causal_palancas_*` | palanca × transición | Veredicto completo (`APROBADA`/`EN_OBSERVACION`/`RECHAZADA`), ATE, impacto en pp, estabilidad temporal |
| `causal_scores_clientes_*` | cliente × transición | P(evolución), tau, **cuadrante**, score y ranking de prioridad |
| `causal_prescripciones_*` | cliente × palanca | Brecha concreta vs segmento objetivo + costo estimado |
| `causal_playbook_*` | transición × cuadrante | Acción recomendada, tamaño del grupo, palancas aprobadas |
| `causal_ab_assignment_*` | cliente | Brazo del experimento (Treatment/Control/Holdout) |
| `causal_backtesting_*` | palanca × transición | Detalle de estabilidad por período (auditoría) |
| `causal_involucion_palancas_*` | palanca × transición | Factores de RIESGO y PROTECTORES |
| `causal_involucion_scores_*` | cliente × transición | P(involución) + cuadrante de riesgo (alertas tempranas) |

---

## Preguntas frecuentes

**¿Por qué no basta con SHAP / feature importance?**
Porque mide asociación. Una variable puede rankear #1 en SHAP y tener efecto causal
*negativo* (espuria, confundida por otra). El pipeline las detecta y las marca.

**¿Qué significa que una palanca esté `RECHAZADA` si "se veía clarísima" en los datos?**
Que su efecto solo existía en el snapshot donde se descubrió (ruido o confusión).
El Pilar 1 lo demuestra re-estimando el efecto en 6 períodos.

**¿El tau de los cuadrantes está 100% probado?**
El eje X (P de evolución) se valida directamente con transiciones reales T→T+1
(Pilar 2). El eje Y (tau) se valida indirectamente (Qini, sensibilidad, estabilidad),
pero su confirmación definitiva es el **A/B test** que el propio pipeline diseña.
Por eso la salida incluye la asignación experimental: ejecutar la campaña sobre
`Treatment` y comparar contra `Control` cierra el ciclo.

**¿Qué pasó con DiCE (contrafactuales) del pipeline anterior?**
Se reemplazó por el análisis de **brecha vs segmento objetivo**: misma utilidad
("¿qué tendría que cambiar este cliente?"), pero determinista, explicable a negocio
y sin fragilidad numérica. Solo usa palancas `APROBADAS`, nunca variables espurias.

**¿Cada cuánto se corre?**
Mensual, al llegar el nuevo snapshot: agregar el período a `periodos_historicos`,
actualizar `periodo_actual` y re-ejecutar. El backtesting se vuelve más sólido con
cada mes adicional de historia.
