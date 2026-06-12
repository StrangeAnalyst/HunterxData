"""Backtesting integrado: confirmar que lo que dice el analisis causal es VERDAD.

Antes esto era una "auditoria independiente" separada; ahora es una etapa
obligatoria del pipeline. Ninguna palanca llega a la estrategia sin pasar
por aqui. Tres pilares:

  PILAR 1 — Estabilidad temporal:
    Re-estima el ATE de cada palanca en todos los periodos historicos.
    Una palanca real debe ser significativa y con el MISMO signo en la
    mayoria de los periodos, no solo en el snapshot donde se descubrio.
      CONFIRMADA : significativa+mismo signo en >= 60% de periodos (config)
      INESTABLE  : 30-60%
      REFUTADA   : < 30%  (probable correlacion espuria del snapshot)

  PILAR 2 — Cuadrantes vs realidad (T -> T+1):
    Clasifica con datos del periodo T y observa quien SUBIO realmente en T+1.
    Con los ejes de negocio (X=P(evolucion), Y=tau) la prediccion testeable
    sin campana es: los cuadrantes de P ALTA (Persuadable, Sure-thing) deben
    transitar mas que los de P BAJA (Sleeping-dog, Lost-cause). El split por
    tau solo se puede validar del todo con el A/B test.

  PILAR 3 — Sensibilidad direccional:
    Mueve cada palanca confirmada en multiplos de su IQR y verifica que la
    probabilidad predicha se mueva en la MISMA direccion que el ATE.
    Si DML dice "positivo" pero el modelo responde "negativo", la palanca
    queda en cuarentena.

Veredicto final por palanca = combina los 3 pilares:
  APROBADA       : CONFIRMADA + placebo OK + sensibilidad coherente
  EN_OBSERVACION : CONFIRMADA con alguna alerta, o INESTABLE
  RECHAZADA      : REFUTADA o sensibilidad contradictoria
Solo las APROBADAS alimentan la estrategia.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from .data import preparar_par, codificar_categoricas, clasificar_variables
from .causal import estimar_ate_palanca, ECONML_OK


# ---------------------------------------------------------------------------
# PILAR 1 — Estabilidad temporal del ATE
# ---------------------------------------------------------------------------

def backtest_estabilidad(historia, cfg, vars_info, df_causal, pares):
    """Re-estima el ATE de cada palanca validada en cada periodo historico."""
    bt = cfg['backtesting']
    alpha = cfg['causal']['significancia']
    cortos = cfg['segmentos']['nombres_cortos']
    controles = vars_info['controles']
    filas = []

    print('[BACKTESTING P1] Estabilidad temporal del ATE')
    if len(df_causal) == 0:
        print('  Sin palancas que backtestear.')
        return pd.DataFrame()

    # Solo backtesteamos lo que el snapshot actual marco como relevante
    a_testear = df_causal[df_causal['validada'] | df_causal['significativa']]

    for (par, palanca), grupo in a_testear.groupby(['par', 'palanca']):
        seg_o = seg_d = None
        for a, b in pares:
            if f'{cortos[a]}_to_{cortos[b]}' == par:
                seg_o, seg_d = a, b
        if seg_o is None:
            continue
        signo_esperado = 1 if grupo.iloc[0]['ate'] > 0 else -1

        historial = []
        for periodo, df_p in sorted(historia.items()):
            df_par, y = preparar_par(df_p, cfg, seg_o, seg_d, [palanca],
                                     bt.get('max_sample_per_par', 10000),
                                     bt.get('random_state', 42) if 'random_state' in bt
                                     else cfg['modelado']['random_state'])
            if df_par is None or palanca not in df_par.columns:
                continue
            ctrl = [c for c in controles if c in df_par.columns][:cfg['causal'].get('max_controles', 15)]
            if not ctrl:
                # fallback: otras numericas como controles
                num = df_par.select_dtypes(include=[np.number]).columns
                ctrl = [c for c in num if c != palanca][:cfg['causal'].get('max_controles', 15)]
            try:
                r = estimar_ate_palanca(df_par, y, palanca, ctrl, [], cfg)
            except Exception:
                r = None
            if r is None:
                continue
            historial.append({'periodo': periodo, 'ate': r['ate'], 'pval': r['pval'],
                              'sig_mismo_signo': r['pval'] < alpha and
                                                 np.sign(r['ate']) == signo_esperado})

        if len(historial) < 2:
            continue
        score = sum(h['sig_mismo_signo'] for h in historial) / len(historial)
        if score >= bt['estabilidad_confirmada']:
            clase = 'CONFIRMADA'
        elif score >= bt['estabilidad_inestable']:
            clase = 'INESTABLE'
        else:
            clase = 'REFUTADA'

        ates = [h['ate'] for h in historial]
        filas.append({
            'par': par, 'palanca': palanca, 'estabilidad': score,
            'clasificacion_temporal': clase,
            'n_periodos': len(historial),
            'n_consistentes': int(sum(h['sig_mismo_signo'] for h in historial)),
            'ate_promedio': float(np.mean(ates)), 'ate_std': float(np.std(ates)),
            'historial': historial,
        })
        print(f'  {par:28s} | {palanca:35s} | {clase} ({score:.0%} de {len(historial)} periodos)')

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# PILAR 2 — Cuadrantes vs transiciones reales T -> T+1
# ---------------------------------------------------------------------------

def validar_cuadrantes_t1(historia, cfg, df_scores):
    """Contrasta la clasificacion de cuadrantes contra quien subio de verdad.

    Usa el penultimo periodo como T y el ultimo como T+1 (ground truth).
    """
    col_id = cfg['data']['col_id']
    col_seg = cfg['data']['col_segmento']
    orden = cfg['segmentos']['orden']

    print('[BACKTESTING P2] Cuadrantes vs transiciones reales (T -> T+1)')
    periodos = sorted(historia.keys())
    if len(periodos) < 2:
        print('  [SKIP] Se necesitan >= 2 periodos.')
        return None
    t0, t1 = periodos[-2], periodos[-1]
    print(f'  T={t0} (clasificacion) | T+1={t1} (realidad)')

    rank = {s: i for i, s in enumerate(orden)}
    df_t0 = historia[t0][[col_id, col_seg]].rename(columns={col_seg: 'seg_t0'})
    df_t1 = historia[t1][[col_id, col_seg]].rename(columns={col_seg: 'seg_t1'})
    df_real = df_t0.merge(df_t1, on=col_id)
    df_real['subio'] = (df_real['seg_t1'].map(rank) > df_real['seg_t0'].map(rank)).astype(int)
    tasa_base = df_real['subio'].mean()
    print(f'  Clientes observados en ambos periodos: {len(df_real):,} | tasa base de subida: {tasa_base:.2%}')

    df_m = df_scores.merge(df_real[[col_id, 'subio', 'seg_t0', 'seg_t1']], on=col_id, how='inner')
    if len(df_m) < 50:
        print('  [SKIP] Match insuficiente entre scores y transiciones reales.')
        return None

    tasas = df_m.groupby('cuadrante').agg(
        n=('subio', 'size'), subieron=('subio', 'sum'),
        tasa_real=('subio', 'mean'), tau_prom=('tau', 'mean'),
        p_prom=('p_evolucion', 'mean')).reset_index()

    print(f'\n  {"Cuadrante":<14} {"N":>9} {"Tasa real T+1":>14} {"P prom":>8} {"tau prom":>10}')
    for _, r in tasas.sort_values('tasa_real', ascending=False).iterrows():
        print(f'  {r["cuadrante"]:<14} {r["n"]:>9,} {r["tasa_real"]:>13.2%} '
              f'{r["p_prom"]:>8.3f} {r["tau_prom"]:>10.5f}')

    # Criterio testeable sin campana: lado P-alta > lado P-baja
    t = tasas.set_index('cuadrante')['tasa_real']
    lado_alto = np.nanmean([t.get('Persuadable', np.nan), t.get('Sure-thing', np.nan)])
    lado_bajo = np.nanmean([t.get('Sleeping-dog', np.nan), t.get('Lost-cause', np.nan)])
    orden_ok = bool(lado_alto > lado_bajo)
    print(f'\n  Lado P-ALTA (Persuadable+Sure-thing): {lado_alto:.2%}')
    print(f'  Lado P-BAJA (Sleeping-dog+Lost-cause): {lado_bajo:.2%}')
    print(f'  Criterio P-alta > P-baja: {"CUMPLE" if orden_ok else "NO CUMPLE — revisar umbrales"}')
    print('  Nota: el split vertical (tau) solo se confirma con el A/B test en campo.')

    # Curva Qini sobre tau: ¿ordenar por tau gana al azar?
    df_s = df_m.sort_values('tau', ascending=False).reset_index(drop=True)
    n = len(df_s)
    qini = (df_s['subio'].cumsum() - df_s['subio'].mean() * np.arange(1, n + 1)) / n
    auuc = float(qini.sum() / n)
    print(f'  AUUC (curva Qini sobre tau): {auuc:+.4f} '
          f'{"-> tau discrimina mejor que el azar" if auuc > 0 else "-> tau NO discrimina, tratar como ruido"}')

    return {'t0': t0, 't1': t1, 'tasas': tasas, 'orden_ok': orden_ok,
            'tasa_base': float(tasa_base), 'auuc': auuc, 'df_match': df_m}


# ---------------------------------------------------------------------------
# PILAR 3 — Sensibilidad direccional
# ---------------------------------------------------------------------------

def sensibilidad_direccional(descubrimiento, df_estabilidad, cfg):
    """Mueve cada palanca confirmada en multiplos de IQR y chequea la direccion."""
    pasos = cfg['backtesting'].get('sensibilidad_pasos', [-2, -1, -0.5, 0.5, 1, 2])
    filas = []

    print('[BACKTESTING P3] Sensibilidad direccional (pasos IQR: '
          f'{pasos})')
    if len(df_estabilidad) == 0:
        return pd.DataFrame()

    confirmadas = df_estabilidad[df_estabilidad['clasificacion_temporal'] == 'CONFIRMADA']
    for _, row in confirmadas.iterrows():
        par, palanca = row['par'], row['palanca']
        if par not in descubrimiento:
            continue
        res = descubrimiento[par]
        X = res['X_test']
        if palanca not in X.columns:
            continue
        modelo = res['modelo']
        iqr = float(X[palanca].quantile(0.75) - X[palanca].quantile(0.25))
        if iqr <= 0:
            continue
        p_base = modelo.predict_proba(X)[:, 1].mean()
        curva = []
        for paso in pasos:
            X_mod = X.copy()
            X_mod[palanca] = X[palanca] + paso * iqr
            curva.append((paso, float(modelo.predict_proba(X_mod)[:, 1].mean() - p_base) * 100))

        # Pendiente observada vs signo del ATE temporal promedio
        pendiente = np.polyfit([c[0] for c in curva], [c[1] for c in curva], 1)[0]
        signo_ate = np.sign(row['ate_promedio'])
        coherente = bool(np.sign(pendiente) == signo_ate)
        filas.append({'par': par, 'palanca': palanca, 'pendiente_pp_por_iqr': float(pendiente),
                      'signo_ate': int(signo_ate), 'coherente': coherente,
                      'curva': curva})
        tag = 'COHERENTE' if coherente else 'CONTRADICTORIA (cuarentena)'
        print(f'  {par:28s} | {palanca:35s} | {pendiente:+.2f}pp/IQR | {tag}')

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# VEREDICTO FINAL
# ---------------------------------------------------------------------------

def veredicto_final(df_causal, df_estabilidad, df_sensibilidad):
    """Combina los 3 pilares en un veredicto por palanca.

    APROBADA / EN_OBSERVACION / RECHAZADA. Solo APROBADA va a estrategia.
    """
    if len(df_causal) == 0:
        return pd.DataFrame()

    est = df_estabilidad.set_index(['par', 'palanca']) if len(df_estabilidad) else None
    sen = df_sensibilidad.set_index(['par', 'palanca']) if len(df_sensibilidad) else None

    filas = []
    for _, row in df_causal[df_causal['validada']].iterrows():
        clave = (row['par'], row['palanca'])
        clase_t = est.loc[clave, 'clasificacion_temporal'] if est is not None and clave in est.index else 'SIN_HISTORIA'
        estab = float(est.loc[clave, 'estabilidad']) if est is not None and clave in est.index else np.nan
        coher = bool(sen.loc[clave, 'coherente']) if sen is not None and clave in sen.index else None

        if clase_t == 'CONFIRMADA' and coher is not False:
            veredicto = 'APROBADA'
        elif clase_t in ('REFUTADA',) or coher is False:
            veredicto = 'RECHAZADA'
        else:  # INESTABLE o sin historia suficiente
            veredicto = 'EN_OBSERVACION'

        filas.append({**row.to_dict(), 'clasificacion_temporal': clase_t,
                      'estabilidad': estab, 'sensibilidad_coherente': coher,
                      'veredicto': veredicto})

    df_v = pd.DataFrame(filas)
    if len(df_v) > 0:
        print('\n[VEREDICTO] Palancas tras backtesting completo:')
        for v in ['APROBADA', 'EN_OBSERVACION', 'RECHAZADA']:
            sub = df_v[df_v['veredicto'] == v]
            print(f'  {v:15s}: {len(sub)}')
            for _, r in sub.iterrows():
                print(f"    {r['par']:28s} | {r['palanca']:35s} | "
                      f"estabilidad={r['estabilidad'] if pd.notna(r['estabilidad']) else 'n/a'}")
    return df_v


# ---------------------------------------------------------------------------
# Preparacion de historia: aplicar el mismo encoding a cada periodo
# ---------------------------------------------------------------------------

def preparar_historia(historia, cfg):
    """Aplica el encoding de categoricas a cada periodo historico (in-place)."""
    seg = cfg['segmentos']
    col_seg = cfg['data']['col_segmento']
    mitad_sup = seg['orden'][len(seg['orden']) // 2:]
    for periodo, df_p in historia.items():
        y_g = df_p[col_seg].isin(mitad_sup).astype(int)
        codificar_categoricas(df_p, cfg, y_g)
    return historia
