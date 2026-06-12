"""Descubrimiento de palancas candidatas (analisis asociacional).

Para cada par de transicion entrena un LightGBM que separa el segmento bajo
del alto y triangula la importancia de variables con 3 metodos independientes
(SHAP, Gain, Permutation -> ranking de Borda). La seleccion final se
ESTRATIFICA por dimension de negocio para que el filtro causal evalue
palancas de TODAS las areas (productos, canales, mora...), no solo de la
dimension dominante.

IMPORTANTE: lo que sale de aqui es CORRELACION. La validacion causal
(causal.py + backtesting.py) decide que es palanca real.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import permutation_importance

from .data import preparar_par


def seleccion_estratificada(borda, dimension_map, k_total, min_por_dim):
    """Top-k por Borda garantizando representacion de cada dimension de negocio."""
    dim_vars = {}
    for var, dim in dimension_map.items():
        if var in borda.index:
            dim_vars.setdefault(dim, []).append(var)

    seleccion, cobertura = [], {}
    presupuesto = k_total
    for dim in sorted(dim_vars):
        mejores = borda[dim_vars[dim]].nsmallest(min(min_por_dim, len(dim_vars[dim])))
        cobertura[dim] = mejores.index.tolist()
        seleccion.extend(cobertura[dim])
        presupuesto -= len(cobertura[dim])

    if presupuesto > 0:
        restantes = borda.drop(index=seleccion, errors='ignore').nsmallest(presupuesto)
        for var in restantes.index:
            dim = dimension_map.get(var, 'OTROS')
            cobertura.setdefault(dim, []).append(var)
        seleccion.extend(restantes.index.tolist())

    return seleccion[:k_total], cobertura


def descubrir_candidatas(df, cfg, vars_info, pares, etiqueta='evolucion'):
    """Entrena el modelo por par y selecciona candidatas trianguladas.

    Devuelve dict {nombre_par: {...}} con modelo, splits, metricas y candidatas.
    """
    m = cfg['modelado']
    rs = m['random_state']
    accionables = vars_info['accionables']
    cortos = cfg['segmentos']['nombres_cortos']
    resultados = {}

    print(f'[DESCUBRIMIENTO:{etiqueta}] LightGBM + triangulacion SHAP/Gain/Permutation')

    for seg_o, seg_d in pares:
        nombre = f'{cortos[seg_o]}_to_{cortos[seg_d]}'
        df_par, y = preparar_par(df, cfg, seg_o, seg_d, accionables,
                                 m.get('max_sample_per_par'), rs)
        if df_par is None:
            print(f'  {nombre}: muestra insuficiente, se omite')
            continue

        X = df_par[accionables].fillna(df_par[accionables].median())
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=m['test_size'], random_state=rs, stratify=y)
        spw = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)

        mdl = lgb.LGBMClassifier(**{**m['lgbm'], 'random_state': rs, 'scale_pos_weight': spw})
        mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.log_evaluation(0)])

        proba = mdl.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        f1 = f1_score(y_te, mdl.predict(X_te))

        # Triangulacion de importancias
        n_shap = min(m.get('shap_max_samples', 500), len(X_te))
        shap_vals = shap.TreeExplainer(mdl).shap_values(X_te.iloc[:n_shap])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        imp_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=accionables)
        imp_gain = pd.Series(mdl.booster_.feature_importance(importance_type='gain'),
                             index=accionables)
        perm = permutation_importance(mdl, X_te, y_te, n_repeats=10,
                                      random_state=rs, scoring='roc_auc')
        imp_perm = pd.Series(perm.importances_mean, index=accionables)

        borda = (imp_shap.rank(ascending=False) + imp_gain.rank(ascending=False)
                 + imp_perm.rank(ascending=False))
        candidatas, cobertura = seleccion_estratificada(
            borda, vars_info['dimension_map'], m['top_k_palancas'], m['min_por_dimension'])

        resultados[nombre] = {
            'segmento_origen': seg_o, 'segmento_destino': seg_d,
            'modelo': mdl, 'df_par': df_par, 'y': y,
            'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te,
            'auc': auc, 'f1': f1, 'candidatas': candidatas,
            'borda': borda, 'imp_shap': imp_shap, 'cobertura_dim': cobertura,
        }
        print(f'  {nombre:30s} | n={len(y):,} | AUC={auc:.4f} | '
              f'candidatas={len(candidatas)} en {len(cobertura)} dimensiones')

    return resultados


def dosis_respuesta(modelo, X, variable, n_bins=10):
    """Curva de respuesta del modelo al mover una variable (ALE 1D simplificado).

    Sirve para responder: "¿a partir de que nivel esta palanca deja de sumar?"
    Devuelve (centros_bins, efecto_acumulado_pp) o (None, None).
    """
    vals = X[variable].values
    pct = np.unique(np.percentile(vals[~np.isnan(vals)], np.linspace(0, 100, n_bins + 1)))
    if len(pct) < 3:
        return None, None
    efectos, centros = [], []
    for i in range(len(pct) - 1):
        mask = (vals >= pct[i]) & (vals <= pct[i + 1])
        if mask.sum() < 10:
            continue
        X_lo, X_hi = X[mask].copy(), X[mask].copy()
        X_lo[variable], X_hi[variable] = pct[i], pct[i + 1]
        delta = (modelo.predict_proba(X_hi)[:, 1] - modelo.predict_proba(X_lo)[:, 1]).mean()
        efectos.append(delta * 100)
        centros.append((pct[i] + pct[i + 1]) / 2)
    if not efectos:
        return None, None
    acumulado = np.cumsum(efectos)
    return np.array(centros), acumulado - acumulado.mean()
