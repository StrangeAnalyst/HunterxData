"""Validacion de integridad: detectar leakage ANTES de modelar.

Si una variable "predice perfectamente" el segmento, casi siempre es porque
contiene informacion del propio segmento (circularidad), no porque sea una
palanca milagrosa. Modelar sobre leakage produce estrategias falsas.

Tests:
  1. AUC sospechoso: AUC > 0.98 en el par => alerta.
  2. Ablacion: quitar las top-20 variables; si el AUC casi no baja, la senal
     esta repartida (sano); si el modelo colapsa, dependia de pocas variables.
  3. Exclusion de variables circulares (cluster / etiqueta de segmento).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .data import preparar_par


def validar_integridad(df, cfg, vars_info):
    """Corre los tests de leakage por par de transicion. Devuelve DataFrame resumen."""
    m = cfg['modelado']
    rs = m['random_state']
    accionables = vars_info['accionables']
    resultados = []

    print('[INTEGRIDAD] Tests de leakage por par de transicion')

    # Test 3 primero: variables circulares en accionables
    col_seg = cfg['data']['col_segmento']
    circulares = [c for c in accionables if 'cluster' in c.lower() or c == col_seg]
    if circulares:
        print(f'  [FIX] Variables circulares removidas de accionables: {circulares}')
        vars_info['accionables'] = [c for c in accionables if c not in circulares]
        accionables = vars_info['accionables']

    for seg_o, seg_d in cfg['_derivados']['pares_evolucion']:
        df_par, y = preparar_par(df, cfg, seg_o, seg_d, accionables,
                                 m.get('max_sample_per_par'), rs)
        if df_par is None:
            continue
        nombre = f"{cfg['segmentos']['nombres_cortos'][seg_o]}_to_{cfg['segmentos']['nombres_cortos'][seg_d]}"

        X = df_par[accionables].fillna(df_par[accionables].median())
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=rs, stratify=y)
        spw = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)

        mdl = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1,
                                 scale_pos_weight=spw, random_state=rs, n_jobs=-1)
        mdl.fit(X_tr, y_tr)
        auc_full = roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])

        n_ablar = min(20, max(1, len(accionables) // 2))
        top20 = pd.Series(mdl.feature_importances_, index=accionables).nlargest(n_ablar).index.tolist()
        mdl_abl = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1,
                                     scale_pos_weight=spw, random_state=rs, n_jobs=-1)
        mdl_abl.fit(X_tr.drop(columns=top20), y_tr)
        auc_abl = roc_auc_score(y_te, mdl_abl.predict_proba(X_te.drop(columns=top20))[:, 1])

        alerta = auc_full > 0.98
        resultados.append({'par': nombre, 'auc_full': auc_full, 'auc_ablated': auc_abl,
                           'auc_drop': auc_full - auc_abl, 'alerta_leakage': alerta})
        tag = '[ALERTA leakage]' if alerta else '[OK]'
        print(f'  {nombre:30s} | AUC={auc_full:.4f} | sin top-{n_ablar}={auc_abl:.4f} | {tag}')

    df_res = pd.DataFrame(resultados)
    if len(df_res) > 0 and df_res['alerta_leakage'].any():
        print('\n  [ALERTA] Hay pares con AUC > 0.98. Revisar variables sospechosas')
        print('  (las de mayor importancia) antes de confiar en las palancas.')
    return df_res
