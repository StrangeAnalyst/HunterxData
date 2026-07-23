# Análisis del notebook "EDA Completo — Incidentes y Cambios" y mejoras aplicadas en la v2

**Contexto:** modelo de riesgo para el CAB — dado un cambio en un aplicativo, estimar la probabilidad de que cause un incidente. Prevalencia ~1% (166 positivos / 14,764 cambios en la corrida original).

**Veredicto general:** el notebook original es un trabajo sólido y por encima del promedio para un MVP: la discusión de leakage con timeline de disponibilidad, la evaluación empírica de 4 definiciones del target y el análisis operacional por deciles son exactamente lo que un primer modelo para TI necesita. Sin embargo, tiene **3 bugs de código que afectan resultados**, **2 fallas metodológicas que inflan las métricas reportadas**, y varias oportunidades claras de mejora en NLP, features y modelado. Todo esto se corrige/agrega en `EDA_Incidentes_Cambios_MVP1_v2.ipynb` sin alterar la estructura ni la narrativa original.

---

## 1. Bugs encontrados (afectan resultados)

### 1.1 Pares cambio↔incidente desalineados (Sección 8) — CRÍTICO
```python
d1 = set(zip(inc['cxc_n'].dropna(), inc['ticket_n']))
```
`dropna()` compacta la serie: el i-ésimo `causado_por_cambio` no nulo queda emparejado con el i-ésimo ticket **global**, no con el de su propia fila. Además se inyectaban pares invertidos `(incidente, cambio)` al conjunto. Como estos pares definen la variable objetivo, **el target podía quedar mal etiquetado**. *Fix v2:* `dropna(subset=...)` sobre el DataFrame antes del zip y eliminación de los pares invertidos.

### 1.2 Fechas desalineadas tras reordenar (Sección 11) — ALTO
`cambios_solapados`, `cambios_ci_7d/30d` y `_fecha_ref` se construían con la serie `f_inicio` calculada **antes** del `sort_values(...).reset_index(drop=True)`: cada fila recibía la fecha de otra fila. *Fix v2:* las fechas se recalculan desde el DataFrame ya ordenado.

### 1.3 `NameError` en el resumen (Sección 14)
`N_POS` y `PREV` no existen (las variables reales son `N_D1`/`PREV_D1`); la celda revienta al ejecutarse en orden. Señal de que el notebook no se había corrido limpio de arriba a abajo. *Fix v2:* corregido, y la comparación de targets de la Sección 9 ya no depende de features que se crean en la Sección 11 (se construyen versiones mínimas inline).

---

## 2. Fallas metodológicas (inflan las métricas reportadas)

### 2.1 Selección de hiperparámetros y umbral sobre el test — CRÍTICO
En la Sección 17 el "GANADOR" del grid se elegía maximizando **ROC-AUC de test**, y en las Secciones 19/21 el umbral EQopt se optimizaba **sobre el test**. El test dejó de ser una estimación honesta: el ROC 0.8781 / gap 0.0153 reportados son optimistas por construcción. *Fix v2:* split temporal **train 64% / validación 16% / test 20%**; hiperparámetros, umbral y calibración se deciden en validación; el test se evalúa una sola vez con todo congelado.

### 2.2 Leakage sutil en la feature más importante — ALTO
`hist_fallos_previos_ci` (la feature dominante del modelo) contaba como "fallo previo" cualquier cambio anterior etiquetado como positivo, **aunque su incidente aún no hubiera ocurrido** a la fecha del cambio que se puntúa. Ejemplo: cambio A del lunes causa incidente el viernes; un cambio B del miércoles sobre el mismo CI "sabía" del fallo de A dos días antes de que existiera. Además, en producción el enlace cambio→incidente se documenta con rezago, cosa que el cálculo ignoraba. *Fix v2:* el fallo cuenta solo si la **fecha del incidente** es anterior al inicio del cambio actual. (Queda para MVP 2: descontar además el rezago de documentación.) Menciones análogas: la learning curve usaba KFold aleatorio (ahora `TimeSeriesSplit`).

Lo que el notebook original **sí hace bien** en leakage y merece reconocerse: exclusión justificada de las 9 variables post-evento con triple evidencia (timing de negocio, lift >3x, modelo con/sin fuga), TF-IDF ajustado solo en train, y descarte del par correctivo por dirección temporal.

---

## 3. Variable objetivo — de acuerdo con D1, con matices

La elección de **D1 (link directo + filtro temporal)** sobre D2-D4 (CI + ventana) es correcta y está bien defendida: la ventana temporal etiqueta coincidencias, no causas, e infla 3-5x. Dos matices:

1. La comparación "calidad de etiqueta vía PR-AUC de un modelo naive" es débil como evidencia: PR-AUC no es comparable entre targets con prevalencias distintas. El argumento fuerte es el de precisión de etiqueta (<30% de los positivos D2 están en D1) — con eso basta.
2. El riesgo real de D1 es el **sub-registro** (solo 17.6% de incidentes tienen `causado_por_cambio`): los "negativos" contienen positivos no documentados, lo que deprime la precisión aparente del modelo. La v1 lo intentó resolver con weak supervision (efecto cero, conclusión correcta: calidad > cantidad). La ruta correcta es de **proceso**, no de algoritmo: mejorar la captura del campo en Operaciones (queda en el roadmap v2).

---

## 4. Métrica de optimización — faltaba una decisión explícita

La v1 reporta ROC-AUC/Gini/KS como métricas de cabecera y menciona de pasada que "la métrica importante es el lift del top decil". Con 99:1, ROC-AUC es engañoso (premia ordenar bien los negativos, que es lo fácil). La v2 lo hace explícito (nueva Sección 15B):

- **PR-AUC (average precision)** → métrica de **selección** de modelo e hiperparámetros (en validación).
- **Recall@10%** (top decil) → métrica de **negocio**: "si el CAB solo puede revisar a fondo el 10% más riesgoso, ¿qué % de incidentes atrapa?". Se reporta **con intervalo bootstrap** (nueva Sección 22B), porque con ~30 positivos en test el punto estimado tiene varianza enorme.
- ROC-AUC/Gini/KS → gobernanza y filtro anti-overfit (gap train-val < 3pp), no selección.

---

## 5. Feature engineering — bueno, ampliado en v2

Las 5 familias de la v1 son razonables y el historial del CI es la apuesta correcta. La v2 agrega (Sección 11B, todas estrictamente previas al cambio):

| Nueva feature | Racional |
|---|---|
| `dias_desde_ultimo_cambio_ci` | Un CI tocado ayer ≠ un CI estable hace 6 meses |
| `dias_desde_ultimo_incidente_ci` | CI "caliente" (incidente reciente de cualquier causa); usa la fecha de apertura del incidente, conocida en tiempo real |
| `hist_*_grupo` (3) | El riesgo también depende de quién ejecuta, no solo del activo |
| `es_boilerplate`, `freq_texto` | Documentar con plantilla = menor esfuerzo de planificación |
| `kw_*` (9 flags) | Riesgo técnico interpretable ante el CAB (producción, BD, firewall, migración…) |

---

## 6. NLP — mejoras de bajo costo (Sección 7B)

1. **Stopwords en español**: la v1 usaba `stop_words=None`; parte de las 300 features TF-IDF se gastaban en "de/para/realizar".
2. **Boilerplate como señal**: en vez de solo constatar el problema, se convierte en feature (`es_boilerplate`, `freq_texto`).
3. **Keywords de riesgo**: flags explicables, complemento interpretable del TF-IDF.
4. **TF-IDF→SVD(50)**: con `max_depth=1`, los stumps apenas explotan 300 columnas dispersas; la variante comprimida (LSA) se compara en el benchmark 17B. Embeddings quedan bien ubicados en MVP 2 — solo si el benchmark muestra que el texto aporta.

## 7. Modelado — faltaba amplitud de comparación

La v1 solo compara LR vs XGBoost (y la demo de overfitting está muy bien contada). La v2 agrega el benchmark 17B con protocolo idéntico (fit en train, comparación en validación): **Dummy (piso), LR L1, Random Forest, HistGradientBoosting, XGBoost final, LightGBM (opcional), XGB+SVD**. Además:

- **Calibración adelantada a MVP 1** (Sección 21B): Platt vs isotónica ajustadas en validación, Brier + curva de calibración. Con `scale_pos_weight≈84` los scores crudos están sistemáticamente inflados; sin calibrar, "0.70" no significa nada para el CAB.
- **Persistencia y serving** (Sección 23B): bundle `joblib` (modelo + tfidf + features + umbral + calibrador + metadatos) y `score_nuevos_cambios()` — un MVP que vive en variables de kernel no es un MVP.
- **Monitoreo** (Sección 23C): PSI de score y features como línea base de detección de drift.

## 8. Detalles menores corregidos

`use_label_encoder` deprecado; celda pip con IP interna hardcodeada; ruta Windows sin raw-string y sin override por variable de entorno; placeholder `'sin informacion'` duplicado (faltaba la versión con tilde); hardcodes 14764 y factor anual x4; código muerto `if False else` en la construcción de la matriz.

---

## Cómo usar la v2

```bash
pip install -r requirements.txt
export DATA_INCIDENTES_XLSX="/ruta/al/Data Incidentes y Cambios.xlsx"   # opcional
jupyter notebook EDA_Incidentes_Cambios_MVP1_v2.ipynb
```

El notebook fue verificado ejecutándolo completo (85 celdas) contra un dataset sintético con la misma estructura del origen (hojas Incidentes/Cambios, duplicados por estado, links bidireccionales). **Al re-ejecutar con los datos reales, esperar cifras distintas a las de la v1** — especialmente por los fixes 1.1, 2.1 y 2.2. Números más bajos no son una regresión: son los números honestos.
