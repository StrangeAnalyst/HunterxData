"""Generacion de estrategias: el "para que" de todo el pipeline.

Convierte palancas APROBADAS + cuadrantes en acciones concretas:

  1. Brecha vs segmento objetivo (gap analysis): para cada cliente Persuadable,
     cuanto le falta en cada palanca aprobada para parecerse a la mediana del
     segmento al que queremos moverlo. Reemplaza los contrafactuales DiCE del
     pipeline original por una prescripcion determinista, explicable y robusta.
  2. Score de prioridad: tau x deltaLTV / costo  -> a quien atender primero
     por peso invertido.
  3. Playbook por cuadrante y transicion (tabla accionable para NBA/campanas).
  4. Diseno del A/B test para confirmar las palancas en campo.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .quadrants import CUADRANTES


def brecha_vs_objetivo(df, cfg, df_veredicto, descubrimiento):
    """Prescripcion por cliente: gap de cada palanca APROBADA vs segmento objetivo."""
    col_id = cfg['data']['col_id']
    col_seg = cfg['data']['col_segmento']
    pct_obj = cfg['estrategia'].get('gap_percentil_objetivo', 50) / 100
    costos = cfg['estrategia'].get('costo_por_unidad', {})
    filas = []

    aprobadas = df_veredicto[df_veredicto['veredicto'] == 'APROBADA'] \
        if len(df_veredicto) > 0 else pd.DataFrame()
    if len(aprobadas) == 0:
        print('[ESTRATEGIA] Sin palancas APROBADAS: no se generan prescripciones.')
        return pd.DataFrame()

    for par, grupo in aprobadas.groupby('par'):
        if par not in descubrimiento:
            continue
        res = descubrimiento[par]
        seg_destino = res['segmento_destino']
        df_par = res['df_par']
        df_objetivo = df[df[col_seg] == seg_destino]
        clientes_origen = df_par[df_par[col_seg] == res['segmento_origen']]

        for _, p in grupo.iterrows():
            palanca = p['palanca']
            if palanca not in df_objetivo.columns:
                continue
            meta = float(df_objetivo[palanca].quantile(pct_obj))
            costo_unit = costos.get(palanca, costos.get('default', 0))
            actual = clientes_origen[palanca].fillna(0)
            gap = (meta - actual).clip(lower=0)
            con_gap = gap > 0
            filas.append(pd.DataFrame({
                col_id: clientes_origen.loc[con_gap, col_id].values,
                'par': par,
                'palanca': palanca,
                'dimension': p['dimension'],
                'valor_actual': actual[con_gap].values,
                'valor_objetivo': meta,
                'brecha': gap[con_gap].values,
                'costo_estimado': gap[con_gap].values * costo_unit,
                'impacto_pp': p.get('impacto_pp', np.nan),
            }))

    if not filas:
        return pd.DataFrame()
    df_gap = pd.concat(filas, ignore_index=True)
    print(f'[ESTRATEGIA] Prescripciones de brecha: {len(df_gap):,} '
          f'(cliente x palanca aprobada con gap > 0)')
    return df_gap


def priorizar(df_scores, df_gap, cfg, descubrimiento):
    """Score de prioridad por cliente: tau x deltaLTV / costo de cerrar la brecha."""
    if len(df_scores) == 0:
        return pd.DataFrame()
    col_id = cfg['data']['col_id']
    ltv = cfg['segmentos']['ltv_anual']

    df_p = df_scores.copy()
    delta_ltv = {}
    for nombre, res in descubrimiento.items():
        delta_ltv[nombre] = ltv[res['segmento_destino']] - ltv[res['segmento_origen']]
    df_p['delta_ltv'] = df_p['par'].map(delta_ltv).fillna(0)

    if len(df_gap) > 0:
        costo = df_gap.groupby([col_id, 'par'])['costo_estimado'].sum().rename('costo_intervencion')
        df_p = df_p.merge(costo, on=[col_id, 'par'], how='left')
    if 'costo_intervencion' not in df_p.columns:
        df_p['costo_intervencion'] = np.nan
    mediana_costo = df_p['costo_intervencion'].median()
    df_p['costo_intervencion'] = df_p['costo_intervencion'].fillna(
        mediana_costo if pd.notna(mediana_costo) else 10_000).clip(lower=1)

    df_p['score_prioridad'] = (df_p['tau'].clip(lower=0) * df_p['delta_ltv']
                               / df_p['costo_intervencion'])
    # Ranking solo tiene sentido donde hay algo que acelerar
    df_p['rank_prioridad'] = df_p.groupby('par')['score_prioridad'] \
        .rank(ascending=False, method='first')

    top = df_p[df_p['cuadrante'] == 'Persuadable'].nlargest(10, 'score_prioridad')
    print('[ESTRATEGIA] Top-10 Persuadables por ROI (tau x deltaLTV / costo):')
    for _, r in top.iterrows():
        print(f"  {str(r[col_id]):>15s} | {r['par']:25s} | score={r['score_prioridad']:8.2f} "
              f"| P={r['p_evolucion']:.2f} | tau={r['tau']:+.4f}")
    return df_p


def playbook(df_scores, df_veredicto, info_pares, cfg):
    """Tabla resumen accionable: que hacer con cada grupo, con que palanca."""
    if len(df_scores) == 0:
        return pd.DataFrame()
    filas = []
    ltv = cfg['segmentos']['ltv_anual']

    for par, info in info_pares.items():
        sub = df_scores[df_scores['par'] == par]
        aprobadas = df_veredicto[(df_veredicto['par'] == par) &
                                 (df_veredicto['veredicto'] == 'APROBADA')] \
            if len(df_veredicto) > 0 else pd.DataFrame()
        palancas_txt = ', '.join(aprobadas['palanca'].tolist()) if len(aprobadas) > 0 \
            else '(ninguna aprobada: solo monitoreo, no campana causal)'

        for cuadrante, definicion in CUADRANTES.items():
            grupo = sub[sub['cuadrante'] == cuadrante]
            if len(grupo) == 0:
                continue
            filas.append({
                'par': par,
                'cuadrante': cuadrante,
                'n_clientes': len(grupo),
                'pct_clientes': len(grupo) / len(sub),
                'regla': definicion['regla'],
                'lectura_negocio': definicion['lectura'],
                'accion_recomendada': definicion['accion'],
                'palancas_aprobadas': palancas_txt if cuadrante in ('Persuadable', 'Sleeping-dog') else '-',
                'tau_promedio': float(grupo['tau'].mean()),
                'p_promedio': float(grupo['p_evolucion'].mean()),
            })

    df_play = pd.DataFrame(filas)
    print('\n[ESTRATEGIA] Playbook por transicion y cuadrante:')
    for _, r in df_play.iterrows():
        print(f"  {r['par']:25s} | {r['cuadrante']:13s} | {r['n_clientes']:>7,} clientes "
              f"| {r['accion_recomendada'][:55]}")
    return df_play


def disenar_ab_test(df_scores, cfg):
    """Power analysis + asignacion estratificada Treatment/Control/Holdout.

    El A/B test es la confirmacion DEFINITIVA del tau: sin experimento, el
    split vertical de la matriz (impacto de intervenir) es una estimacion.
    """
    ab = cfg.get('ab_test', {})
    if not ab.get('activo', False) or len(df_scores) == 0:
        return pd.DataFrame()
    col_id = cfg['data']['col_id']
    rs = cfg['modelado']['random_state']

    # Tamano de muestra requerido por brazo
    p0, mde = ab['tasa_base'], ab['mde']
    z_a = stats.norm.ppf(1 - ab['alpha'] / 2)
    z_b = stats.norm.ppf(ab['poder'])
    p1, pb = p0 + mde, p0 + mde / 2
    n_req = int(np.ceil(((z_a * np.sqrt(2 * pb * (1 - pb)) +
                          z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / mde) ** 2))
    print(f'[A/B] Poder estadistico: se necesitan {n_req:,} clientes por brazo '
          f'(base={p0:.0%}, MDE={mde:.0%}, poder={ab["poder"]:.0%})')

    asignaciones = []
    rng = np.random.RandomState(rs)
    for par, sub in df_scores.groupby('par'):
        # Solo tiene sentido experimentar donde la intervencion puede mover algo
        elegibles = sub[sub['cuadrante'].isin(['Persuadable', 'Sleeping-dog'])].copy()
        if len(elegibles) < 30:
            continue
        n_t, n_c, n_h = ab['n_treatment'], ab['n_control'], ab['n_holdout']
        total = n_t + n_c + n_h
        if len(elegibles) < total:
            ratio = len(elegibles) / total
            n_t, n_c = int(n_t * ratio), int(n_c * ratio)
            n_h = len(elegibles) - n_t - n_c
        elegibles = elegibles.sample(frac=1, random_state=rs)
        brazos = (['Treatment'] * n_t + ['Control'] * n_c + ['Holdout'] * n_h)
        brazos += ['Holdout'] * (len(elegibles) - len(brazos))
        elegibles['brazo'] = brazos[:len(elegibles)]
        elegibles['fecha_asignacion'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        asignaciones.append(elegibles[[col_id, 'par', 'cuadrante', 'brazo', 'fecha_asignacion']])
        print(f'  {par:25s}: {len(elegibles):,} asignados '
              f'(T={n_t}, C={n_c}, H={len(elegibles) - n_t - n_c})')

    return pd.concat(asignaciones, ignore_index=True) if asignaciones else pd.DataFrame()
