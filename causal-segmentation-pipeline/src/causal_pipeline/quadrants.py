"""Matriz de inversion: 4 cuadrantes con logica de negocio.

Definicion de ejes (ESTANDAR para todo el pipeline y el backtesting):

    Eje X  = P(evolucion)  -> probabilidad de que el cliente suba de segmento
    Eje Y  = tau           -> impacto causal de intervenir (cuanto ACELERA la
                              evolucion una accion del banco)

                       tau ALTO (la intervencion mueve la aguja)
                            |
        SLEEPING-DOG        |        PERSUADABLE
        P baja + tau alto   |        P alta + tau alto
        "Mucho impacto pero |        "Puede evolucionar y la
         baja probabilidad  |         intervencion lo acelera"
         de moverse hoy"    |         => INVERTIR PRIMERO
        --------------------+--------------------------->  P(evolucion)
        LOST-CAUSE          |        SURE-THING
        P baja + tau bajo   |        P alta + tau bajo
        "Ni probabilidad ni |        "Va a evolucionar solo,
         respuesta"         |         no gastar en empujarlo"
         => NO INVERTIR     |         => MONITOREAR
                            |

Lectura de negocio:
  - Lado DERECHO (Persuadable + Sure-thing): los que PUEDEN evolucionar.
  - El PERSUADABLE es el de mayor retorno: alta probabilidad Y alto impacto
    de la intervencion => evoluciona MAS RAPIDO si lo empujamos.
  - El SLEEPING-DOG es el caso contrario al Sure-thing: la palanca tiene mucho
    impacto sobre el, pero hoy su probabilidad base de moverse es baja =>
    apuesta selectiva de mediano plazo, no campana masiva.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    from econml.metalearners import XLearner
    XLEARNER_OK = True
except Exception:
    XLEARNER_OK = False


CUADRANTES = {
    'Persuadable': {
        'regla': 'P ALTA + tau ALTO',
        'lectura': 'Puede evolucionar y la intervencion lo acelera',
        'accion': 'INVERTIR PRIMERO: campana dirigida con la palanca validada',
    },
    'Sure-thing': {
        'regla': 'P ALTA + tau BAJO',
        'lectura': 'Va a evolucionar solo, el estimulo no agrega casi nada',
        'accion': 'NO gastar en empujarlo; monitorear y proteger (retencion)',
    },
    'Sleeping-dog': {
        'regla': 'P BAJA + tau ALTO',
        'lectura': 'La palanca tiene mucho impacto pero hoy no tiene probabilidad de moverse',
        'accion': 'Apuesta selectiva: desarrollar condiciones base antes de campanas masivas',
    },
    'Lost-cause': {
        'regla': 'P BAJA + tau BAJO',
        'lectura': 'Ni condiciones base ni respuesta a la intervencion',
        'accion': 'NO invertir; solo oferta estructural de bajo costo',
    },
}


def asignar_cuadrantes(p_evolucion, tau, cfg):
    """Asigna el cuadrante de cada cliente segun los umbrales del config."""
    q = cfg['cuadrantes']
    umbral_p = np.percentile(p_evolucion, q.get('umbral_p_percentil', 50))
    umbral_tau = np.percentile(tau, q.get('umbral_tau_percentil', 60))

    cuadrantes = np.where(
        p_evolucion >= umbral_p,
        np.where(tau >= umbral_tau, 'Persuadable', 'Sure-thing'),
        np.where(tau >= umbral_tau, 'Sleeping-dog', 'Lost-cause'),
    )
    return cuadrantes, float(umbral_p), float(umbral_tau)


def estimar_tau_individual(df_par, y, mejor_palanca, accionables, modelo_p, cfg):
    """tau por cliente via X-learner (tratamiento = palanca sobre su mediana).

    Devuelve (tau, p_evolucion). Si X-learner falla, cae a un proxy basado en
    la sensibilidad del modelo predictivo.
    """
    X = df_par[accionables].fillna(0)
    p_evolucion = modelo_p.predict_proba(X)[:, 1]

    T = (df_par[mejor_palanca].fillna(0) > df_par[mejor_palanca].median()).astype(int).values
    if XLEARNER_OK and 0 < T.sum() < len(T):
        try:
            xl = XLearner(
                models=lgb.LGBMClassifier(n_estimators=150, max_depth=6, verbose=-1, n_jobs=-1),
                cate_models=lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1, n_jobs=-1))
            xl.fit(y.values, T, X=X.values)
            tau = xl.effect(X.values).flatten()
            return tau, p_evolucion
        except Exception as e:
            print(f'    [WARN] XLearner fallo ({str(e)[:50]}), uso proxy de sensibilidad')

    # Proxy: cuanto sube P si llevamos la palanca a su p75
    X_up = X.copy()
    X_up[mejor_palanca] = X[mejor_palanca].clip(lower=X[mejor_palanca].quantile(0.75))
    tau = modelo_p.predict_proba(X_up)[:, 1] - p_evolucion
    return tau, p_evolucion


def segmentar_clientes(descubrimiento, df_causal, cfg, vars_info):
    """Construye la matriz de inversion por par de transicion.

    Devuelve (df_scores, info_pares):
      df_scores: 1 fila por cliente-par con p_evolucion, tau, cuadrante
      info_pares: {par: {mejor_palanca, umbral_p, umbral_tau}}
    """
    col_id = cfg['data']['col_id']
    col_seg = cfg['data']['col_segmento']
    accionables = vars_info['accionables']
    todas, info_pares = [], {}

    print('[CUADRANTES] Matriz de inversion (X=P(evolucion), Y=tau impacto causal)')

    for nombre, res in descubrimiento.items():
        df_par, y = res['df_par'], res['y']

        # Mejor palanca: la validada con menor p-value; si no hay, top Borda
        validadas = df_causal[(df_causal['par'] == nombre) & (df_causal['validada'])] \
            if len(df_causal) > 0 else pd.DataFrame()
        if len(validadas) > 0:
            mejor = validadas.sort_values('pval').iloc[0]['palanca']
            origen_palanca = 'validada_causal'
        else:
            mejor = res['candidatas'][0]
            origen_palanca = 'proxy_borda (sin palanca causal validada)'

        tau, p_evol = estimar_tau_individual(df_par, y, mejor, accionables, res['modelo'], cfg)
        cuadrante, u_p, u_tau = asignar_cuadrantes(p_evol, tau, cfg)

        df_q = pd.DataFrame({
            col_id: df_par[col_id].values,
            'par': nombre,
            'segmento_actual': df_par[col_seg].values,
            'p_evolucion': p_evol,
            'tau': tau,
            'cuadrante': cuadrante,
            'palanca_principal': mejor,
        })
        todas.append(df_q)
        info_pares[nombre] = {'mejor_palanca': mejor, 'origen_palanca': origen_palanca,
                              'umbral_p': u_p, 'umbral_tau': u_tau}

        print(f'\n  {nombre} | palanca: {mejor} ({origen_palanca})')
        conteo = pd.Series(cuadrante).value_counts()
        for c in ['Persuadable', 'Sure-thing', 'Sleeping-dog', 'Lost-cause']:
            n = int(conteo.get(c, 0))
            print(f'    {c:14s}: {n:>7,} ({n / len(cuadrante) * 100:5.1f}%) | {CUADRANTES[c]["accion"]}')

    df_scores = pd.concat(todas, ignore_index=True) if todas else pd.DataFrame()
    return df_scores, info_pares


def graficar_cuadrantes(df_scores, info_pares, titulo='Matriz de Inversion'):
    """Scatter P(evolucion) vs tau con los 4 cuadrantes, un panel por par."""
    import matplotlib.pyplot as plt

    pares = list(info_pares.keys())
    if not pares:
        return None
    colores = {'Persuadable': '#27ae60', 'Sure-thing': '#3498db',
               'Sleeping-dog': '#f39c12', 'Lost-cause': '#bdc3c7'}
    fig, axes = plt.subplots(1, len(pares), figsize=(5.5 * len(pares), 5.5), squeeze=False)

    for ax, par in zip(axes[0], pares):
        sub = df_scores[df_scores['par'] == par]
        for c, color in colores.items():
            m = sub['cuadrante'] == c
            ax.scatter(sub.loc[m, 'p_evolucion'], sub.loc[m, 'tau'],
                       c=color, label=f'{c} ({m.sum():,})', alpha=0.25, s=6)
        ax.axvline(info_pares[par]['umbral_p'], color='gray', ls='--', lw=1)
        ax.axhline(info_pares[par]['umbral_tau'], color='gray', ls='--', lw=1)
        ax.set_xlabel('P(evolucion) — probabilidad de subir')
        ax.set_ylabel('tau — impacto causal de intervenir')
        ax.set_title(par, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'{titulo}\nDerecha=pueden evolucionar | Arriba=la intervencion impacta',
                 fontweight='bold')
    fig.tight_layout()
    return fig
