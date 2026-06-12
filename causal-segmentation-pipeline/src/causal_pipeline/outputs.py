"""Persistencia de resultados (Delta) y resumen ejecutivo.

Tablas de salida (todas terminan en _<sufijo> del config):
  causal_palancas_         veredicto completo por palanca (con backtesting)
  causal_scores_clientes_  P(evolucion), tau, cuadrante y prioridad por cliente
  causal_prescripciones_   brechas por cliente x palanca aprobada
  causal_playbook_         acciones por transicion y cuadrante
  causal_ab_assignment_    asignacion del experimento
  causal_backtesting_      detalle de estabilidad temporal por palanca
"""

import pandas as pd

from .config import tabla_salida


def _guardar(spark, df, nombre_tabla):
    if df is None or len(df) == 0:
        print(f'  [SKIP] {nombre_tabla}: sin filas')
        return
    cols = [c for c in df.columns if not df[c].apply(lambda v: isinstance(v, (list, dict))).any()]
    sdf = spark.createDataFrame(df[cols])
    sdf.write.format('delta').mode('overwrite') \
        .option('overwriteSchema', 'true').saveAsTable(nombre_tabla)
    print(f'  [OK] {nombre_tabla}: {len(df):,} filas')


def guardar_resultados(spark, cfg, resultados):
    """Guarda todas las tablas Delta. `resultados` es un dict con los DataFrames."""
    if not cfg['salidas'].get('guardar_delta', True):
        print('[SALIDAS] guardar_delta=false: no se escriben tablas.')
        return
    print('[SALIDAS] Escribiendo tablas Delta...')
    mapeo = {
        'df_veredicto': 'causal_palancas',
        'df_prioridad': 'causal_scores_clientes',
        'df_gap': 'causal_prescripciones',
        'df_playbook': 'causal_playbook',
        'df_ab': 'causal_ab_assignment',
        'df_estabilidad': 'causal_backtesting',
        'df_invol_causal': 'causal_involucion_palancas',
        'df_invol_scores': 'causal_involucion_scores',
    }
    for clave, base in mapeo.items():
        df = resultados.get(clave)
        if df is not None and len(df) > 0:
            _guardar(spark, df, tabla_salida(cfg, base))


def resumen_ejecutivo_html(cfg, resultados):
    """Genera el HTML del resumen ejecutivo para displayHTML() en Databricks."""
    df_v = resultados.get('df_veredicto', pd.DataFrame())
    df_play = resultados.get('df_playbook', pd.DataFrame())
    df_scores = resultados.get('df_prioridad', resultados.get('df_scores', pd.DataFrame()))
    bt = resultados.get('validacion_t1')

    aprobadas = df_v[df_v['veredicto'] == 'APROBADA'] if len(df_v) > 0 else pd.DataFrame()
    n_persuadables = int((df_scores['cuadrante'] == 'Persuadable').sum()) if len(df_scores) > 0 else 0

    css_caja = ('text-align:center;padding:14px;background:#0f3460;'
                'border-radius:8px;min-width:150px;margin:5px;')
    h = [f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;padding:20px;
            background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;color:#eee;">
  <h1 style="text-align:center;color:#00d4ff;margin-bottom:4px;">Pipeline Causal — {cfg['caso_uso']}</h1>
  <p style="text-align:center;color:#aaa;margin-top:0;">
    Palancas validadas con backtesting + matriz de inversion + estrategias priorizadas</p>
  <div style="display:flex;justify-content:space-around;flex-wrap:wrap;margin:18px 0;">
    <div style="{css_caja}"><div style="font-size:26px;font-weight:bold;color:#27ae60;">{len(aprobadas)}</div>
      <div style="font-size:12px;color:#aaa;">Palancas APROBADAS<br>(causal + backtesting)</div></div>
    <div style="{css_caja}"><div style="font-size:26px;font-weight:bold;color:#00d4ff;">{n_persuadables:,}</div>
      <div style="font-size:12px;color:#aaa;">Clientes Persuadables<br>(invertir primero)</div></div>
    <div style="{css_caja}"><div style="font-size:26px;font-weight:bold;color:#f39c12;">
      {('SI' if bt and bt.get('orden_ok') else 'REVISAR') if bt else 'N/A'}</div>
      <div style="font-size:12px;color:#aaa;">Cuadrantes confirmados<br>con realidad T+1</div></div>
  </div>
"""]

    # Palancas aprobadas
    h.append('<h2 style="color:#27ae60;border-bottom:2px solid #27ae60;padding-bottom:6px;">'
             'Palancas en las que SI se puede confiar</h2>')
    if len(aprobadas) > 0:
        h.append('<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                 '<tr style="background:#0d1b2a;color:#00d4ff;">'
                 '<th style="padding:6px;text-align:left;">Transicion</th>'
                 '<th style="padding:6px;text-align:left;">Palanca</th>'
                 '<th style="padding:6px;">Dimension</th>'
                 '<th style="padding:6px;">Impacto (pp)</th>'
                 '<th style="padding:6px;">Estabilidad temporal</th></tr>')
        for _, r in aprobadas.iterrows():
            imp = f"{r['impacto_pp']:+.1f}" if pd.notna(r.get('impacto_pp')) else 'n/a'
            est = f"{r['estabilidad']:.0%}" if pd.notna(r.get('estabilidad')) else 'n/a'
            h.append(f'<tr style="border-bottom:1px solid #333;">'
                     f'<td style="padding:5px;">{r["par"]}</td>'
                     f'<td style="padding:5px;"><b>{r["palanca"]}</b></td>'
                     f'<td style="padding:5px;text-align:center;">{r["dimension"]}</td>'
                     f'<td style="padding:5px;text-align:center;color:#2ecc71;">{imp}</td>'
                     f'<td style="padding:5px;text-align:center;">{est}</td></tr>')
        h.append('</table>')
    else:
        h.append('<p style="color:#f39c12;">Ninguna palanca paso el filtro completo '
                 '(causal + estabilidad temporal + sensibilidad). Las estrategias de '
                 'campana deben esperar al A/B test o a mas historia.</p>')

    # Playbook
    if len(df_play) > 0:
        h.append('<h2 style="color:#00d4ff;border-bottom:2px solid #00d4ff;padding-bottom:6px;">'
                 'Playbook: que hacer con cada grupo</h2>'
                 '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                 '<tr style="background:#0d1b2a;color:#00d4ff;">'
                 '<th style="padding:6px;text-align:left;">Transicion</th>'
                 '<th style="padding:6px;">Cuadrante</th><th style="padding:6px;">Clientes</th>'
                 '<th style="padding:6px;text-align:left;">Accion</th></tr>')
        for _, r in df_play.iterrows():
            h.append(f'<tr style="border-bottom:1px solid #333;">'
                     f'<td style="padding:5px;">{r["par"]}</td>'
                     f'<td style="padding:5px;text-align:center;">{r["cuadrante"]}</td>'
                     f'<td style="padding:5px;text-align:center;">{r["n_clientes"]:,}</td>'
                     f'<td style="padding:5px;">{r["accion_recomendada"]}</td></tr>')
        h.append('</table>')

    h.append(f"""
  <hr style="border-color:#333;margin-top:22px;">
  <p style="text-align:center;color:#666;font-size:11px;">
    Config: {cfg['caso_uso']} | periodo {cfg['data']['periodo_actual']} |
    backtesting sobre {len(cfg['data']['periodos_historicos'])} periodos |
    generado {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>""")
    return ''.join(h)
