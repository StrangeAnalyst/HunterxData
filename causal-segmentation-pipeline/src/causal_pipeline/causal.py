"""Validacion causal con Double Machine Learning (DML).

Convierte las CANDIDATAS (correlacion) en PALANCAS (causalidad) estimando el
ATE de cada una controlando por todas las demas + los controles de contexto.

Criterios para validar una palanca de evolucion (todos deben cumplirse):
  1. ATE > 0           -> empuja hacia arriba
  2. p-value < alpha   -> el efecto no es ruido
  3. placebo < umbral  -> al romper la relacion (Y permutado) el efecto desaparece

Para involucion, el signo se interpreta distinto:
  ATE > 0 => FACTOR DE RIESGO (acelera la caida)
  ATE < 0 => FACTOR PROTECTOR (previene la caida)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    from econml.dml import LinearDML, CausalForestDML
    ECONML_OK = True
except Exception:
    ECONML_OK = False


def _dml_ate(Y, T, W, random_state, n_estimators=100, max_depth=5):
    """LinearDML con LGBMRegressor en primera etapa (requisito de econml)."""
    dml = LinearDML(
        model_y=lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  verbose=-1, n_jobs=-1),
        model_t=lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  verbose=-1, n_jobs=-1),
        random_state=random_state, cv=3,
    )
    dml.fit(Y, T, X=None, W=W)
    ate = dml.const_marginal_ate(X=None)
    ate = float(np.asarray(ate).flatten()[0])
    inf = dml.const_marginal_effect_inference(X=None)
    ci_lo = float(np.asarray(inf.conf_int()[0]).flatten()[0])
    ci_hi = float(np.asarray(inf.conf_int()[1]).flatten()[0])
    pval = float(np.asarray(inf.pvalue()).flatten()[0])
    return ate, ci_lo, ci_hi, pval


def estimar_ate_palanca(df_par, y, palanca, controles, otras_candidatas, cfg):
    """ATE de una palanca + placebo test. Devuelve dict o None."""
    rs = cfg['modelado']['random_state']
    max_ctrl = cfg['causal'].get('max_controles', 15)

    T = df_par[palanca].fillna(0).values.astype(float).reshape(-1, 1)
    if np.nanstd(T) < 1e-10:
        return None
    Y = y.values.astype(float)

    w_cols = [c for c in dict.fromkeys(list(otras_candidatas) + list(controles))
              if c in df_par.columns and c != palanca][:max_ctrl + len(otras_candidatas)]
    W = df_par[w_cols].fillna(0).values.astype(float)
    if W.shape[1] == 0:
        return None

    ate, ci_lo, ci_hi, pval = _dml_ate(Y, T, W, rs)

    # Placebo: con Y permutado el ATE deberia colapsar hacia 0
    Y_perm = np.random.RandomState(rs).permutation(Y)
    ate_placebo, _, _, _ = _dml_ate(Y_perm, T, W, rs + 1, n_estimators=50, max_depth=4)
    placebo_pct = abs(ate_placebo / ate) * 100 if abs(ate) > 1e-12 else 999.0

    return {'ate': ate, 'ci_low': ci_lo, 'ci_high': ci_hi, 'pval': pval,
            'placebo_pct': placebo_pct}


def validar_causalmente(descubrimiento, cfg, vars_info, direccion='evolucion'):
    """Corre DML sobre las candidatas de cada par. Devuelve DataFrame de resultados."""
    if not ECONML_OK:
        raise ImportError('[CAUSAL] econml no disponible. Instale econml>=0.15.')

    alpha = cfg['causal']['significancia']
    placebo_max = cfg['causal']['placebo_max_pct']
    controles = vars_info['controles']
    dim_map = vars_info['dimension_map']
    filas = []

    print(f'[CAUSAL:{direccion}] DML + placebo (alpha={alpha}, placebo<{placebo_max}%)')

    for nombre, res in descubrimiento.items():
        print(f'\n  === {nombre} ===')
        df_par, y = res['df_par'], res['y']
        candidatas = res['candidatas']

        for palanca in candidatas:
            try:
                otras = [c for c in candidatas if c != palanca]
                r = estimar_ate_palanca(df_par, y, palanca, controles, otras, cfg)
                if r is None:
                    continue
                sig = r['pval'] < alpha
                placebo_ok = r['placebo_pct'] < placebo_max

                if direccion == 'evolucion':
                    validada = r['ate'] > 0 and sig and placebo_ok
                    clase = ('VALIDADA' if validada else
                             'ESPURIA' if (sig and r['ate'] < 0) else 'NO_SIGNIFICATIVA')
                else:  # involucion: Y=1 es CAER de segmento
                    riesgo = r['ate'] > 0 and sig and placebo_ok
                    protectora = r['ate'] < 0 and sig and placebo_ok
                    validada = riesgo or protectora
                    clase = 'RIESGO' if riesgo else ('PROTECTORA' if protectora else 'NO_SIGNIFICATIVA')

                filas.append({
                    'par': nombre, 'palanca': palanca,
                    'dimension': dim_map.get(palanca, 'OTROS'),
                    'ate': r['ate'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                    'pval': r['pval'], 'placebo_pct': r['placebo_pct'],
                    'significativa': sig, 'placebo_ok': placebo_ok,
                    'validada': validada, 'clase': clase,
                })
                tag = {'VALIDADA': '[VALIDADA]', 'ESPURIA': '[ESPURIA]',
                       'RIESGO': '[RIESGO]', 'PROTECTORA': '[PROTECTORA]'}.get(clase, '')
                if tag:
                    print(f'    {palanca:38s} | ATE={r["ate"]:+.6f} | p={r["pval"]:.4f} '
                          f'| placebo={r["placebo_pct"]:5.1f}% | {tag}')
            except Exception as e:
                print(f'    {palanca:38s} | ERROR: {str(e)[:60]}')

    df_causal = pd.DataFrame(filas)
    if len(df_causal) > 0:
        n_ok = int(df_causal['validada'].sum())
        print(f'\n  Total: {n_ok} palancas validadas de {len(df_causal)} testeadas')
        _detectar_simpson(df_causal)
    return df_causal


def _detectar_simpson(df_causal):
    """Alerta si una misma palanca cambia de signo entre transiciones (Simpson)."""
    avisos = []
    for palanca, grupo in df_causal[df_causal['significativa']].groupby('palanca'):
        signos = set(grupo['ate'] > 0)
        if len(signos) > 1:
            avisos.append(palanca)
    if avisos:
        print(f'\n  [ALERTA SIMPSON] Estas palancas INVIERTEN su efecto segun la transicion: {avisos}')
        print('  => NO usarlas como palanca universal; revisar el detalle por par.')


def impacto_practico(df_causal, descubrimiento):
    """Traduce el ATE a puntos porcentuales accionables: ATE x IQR de la palanca.

    "Si muevo a un cliente del p25 al p75 de esta variable, su probabilidad
    de evolucionar sube X pp" — el numero que entiende negocio.
    """
    if len(df_causal) == 0:
        return df_causal
    impactos = []
    for _, row in df_causal.iterrows():
        iqr = np.nan
        if row['par'] in descubrimiento:
            serie = descubrimiento[row['par']]['df_par'].get(row['palanca'])
            if serie is not None:
                iqr = float(serie.quantile(0.75) - serie.quantile(0.25))
        impactos.append(row['ate'] * iqr * 100 if pd.notna(iqr) else np.nan)
    df_causal = df_causal.copy()
    df_causal['impacto_pp'] = impactos  # puntos porcentuales al mover p25->p75
    return df_causal


def dosis_respuesta_causal(df_par, y, palanca, controles, otras, cfg, n_puntos=8):
    """Curva dosis-respuesta causal (CausalForestDML) + punto optimo T*.

    Devuelve dict {dosis, ates, t_star, ate_at_star} o None.
    """
    if not ECONML_OK:
        return None
    rs = cfg['modelado']['random_state']
    T = df_par[palanca].fillna(0).values.astype(float).reshape(-1, 1)
    Y = y.values.astype(float)
    w_cols = [c for c in dict.fromkeys(list(otras) + list(controles))
              if c in df_par.columns and c != palanca]
    W = df_par[w_cols].fillna(0).values.astype(float)
    if W.shape[1] == 0:
        return None
    try:
        cf = CausalForestDML(
            model_y=lgb.LGBMRegressor(n_estimators=80, max_depth=4, verbose=-1, n_jobs=-1),
            model_t=lgb.LGBMRegressor(n_estimators=80, max_depth=4, verbose=-1, n_jobs=-1),
            n_estimators=100, random_state=rs, cv=3)
        X_het = W[:, :min(10, W.shape[1])]
        cf.fit(Y, T, X=X_het, W=W)
        t_flat = T.flatten()
        qs = np.unique(np.nanpercentile(t_flat[t_flat != 0], np.linspace(5, 95, n_puntos)))
        puntos = []
        for q in qs:
            mask = t_flat <= q
            if mask.sum() > 30:
                puntos.append((float(q), float(cf.effect(X_het[mask]).mean())))
        if len(puntos) < 3:
            return None
        dosis, ates = map(np.array, zip(*puntos))
        i = int(np.argmax(np.abs(np.diff(ates)))) + 1
        return {'dosis': dosis.tolist(), 'ates': ates.tolist(),
                't_star': float(dosis[i]), 'ate_at_star': float(ates[i]),
                'cf_model': cf, 'X_het': X_het}
    except Exception:
        return None
