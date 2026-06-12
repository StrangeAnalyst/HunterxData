"""Carga de datos y preparacion de variables.

Responsabilidades:
  1. Cargar el snapshot actual y la historia (para backtesting) desde Delta.
  2. Codificar categoricas accionables (one-hot / target encoding) y de control.
  3. Clasificar variables en ACCIONABLES (candidatas a palanca) vs CONTROLES.
  4. Asignar cada variable a una dimension de negocio.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def cargar_periodo(spark, cfg, periodo):
    """Carga un snapshot de la tabla de features aplicando los filtros del config."""
    from pyspark.sql import functions as F

    d = cfg['data']
    sdf = spark.table(d['tabla_features']).filter(F.col(d['col_periodo']) == periodo)
    for col, val in (d.get('filtros') or {}).items():
        sdf = sdf.filter(F.col(col) == val)
    df = sdf.toPandas()
    return df


def cargar_historia(spark, cfg):
    """Carga todos los periodos historicos. Devuelve {periodo: DataFrame}."""
    historia = {}
    for p in cfg['data']['periodos_historicos']:
        try:
            df = cargar_periodo(spark, cfg, p)
            if len(df) > 0:
                historia[p] = df
                print(f'  [OK] periodo {p}: {len(df):,} clientes')
            else:
                print(f'  [WARN] periodo {p}: sin datos, se omite')
        except Exception as e:
            print(f'  [WARN] periodo {p}: error de carga ({str(e)[:60]}), se omite')
    return historia


def codificar_categoricas(df, cfg, y_global):
    """Codifica categoricas accionables y de control. Modifica df in-place.

    Devuelve (nuevas_accionables, nuevos_controles): listas de columnas creadas.
    """
    v = cfg['variables']
    nuevas_acc, nuevos_ctrl = [], []

    for col, metodo in (v.get('categoricas_accionables') or {}).items():
        if col not in df.columns:
            continue
        if metodo == 'one_hot':
            df[col] = df[col].fillna('desconocido')
            top = df[col].value_counts().head(6).index.tolist()
            limpio = df[col].where(df[col].isin(top), 'otro')
            dummies = pd.get_dummies(limpio, prefix=f'ohe_{col[:15]}', drop_first=True).astype(float)
            for dc in dummies.columns:
                df[dc] = dummies[dc].values
                nuevas_acc.append(dc)
        elif metodo == 'target_encode':
            media_global = float(y_global.mean())
            suavizado = 10
            tmp = pd.DataFrame({'cat': df[col], 'y': y_global.values})
            g = tmp.groupby('cat')['y'].agg(['mean', 'count'])
            te = (g['mean'] * g['count'] + media_global * suavizado) / (g['count'] + suavizado)
            nueva = f'te_{col[:20]}'
            df[nueva] = df[col].map(te).fillna(media_global)
            nuevas_acc.append(nueva)
            for cat in df[col].value_counts().head(5).index:
                if pd.notna(cat):
                    flag = f'is_{col[:10]}_{str(cat)[:15]}'.replace(' ', '_').replace('/', '_')
                    df[flag] = (df[col] == cat).astype(float)
                    nuevas_acc.append(flag)

    for col, metodo in (v.get('categoricas_controles') or {}).items():
        if col not in df.columns:
            continue
        if df[col].dtype.kind in 'if':
            nuevos_ctrl.append(col)
            continue
        nueva = f'ord_{col}'
        le = LabelEncoder()
        df[nueva] = le.fit_transform(df[col].fillna('DESCONOCIDO').astype(str)).astype(float)
        nuevos_ctrl.append(nueva)

    return nuevas_acc, nuevos_ctrl


def clasificar_variables(df, cfg, extras_accionables=None, extras_controles=None):
    """Separa columnas numericas en accionables vs controles segun el config.

    Devuelve dict con:
      accionables: candidatas a palanca (el banco puede moverlas)
      controles:   confounders de contexto para DML
      dimension_map: {variable: dimension de negocio}
    """
    v = cfg['variables']
    no_acc = set(v.get('no_accionables') or [])
    no_acc.update((v.get('categoricas_accionables') or {}).keys())
    no_acc.update((v.get('categoricas_controles') or {}).keys())
    no_acc.update([cfg['data']['col_id'], cfg['data']['col_segmento'], cfg['data']['col_periodo']])

    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    prefijos_encoded = ('ohe_', 'te_', 'is_mcc', 'ord_')

    accionables = []
    for c in numericas:
        if c.startswith(prefijos_encoded):
            continue  # las encoded se agregan via extras
        if c in no_acc or 'cluster' in c.lower():
            continue
        accionables.append(c)

    # Filtrar por % de nulos
    max_nulos = v.get('max_pct_nulos', 0.5)
    pct_nulos = df[accionables].isnull().mean()
    accionables = [c for c in accionables if pct_nulos[c] <= max_nulos]

    for c in (extras_accionables or []):
        if c in df.columns and c not in accionables:
            accionables.append(c)

    controles = [c for c in (v.get('controles') or []) if c in df.columns and c in numericas]
    controles += [c for c in (extras_controles or []) if c in df.columns and c not in controles]

    dimension_map = {var: _dimension(var, v.get('dimensiones') or {}) for var in accionables}

    return {'accionables': accionables, 'controles': controles, 'dimension_map': dimension_map}


def _dimension(var, dimensiones):
    for dim, patrones in dimensiones.items():
        for pat in patrones:
            if pat in var:
                return dim
    return 'OTROS'


def preparar_dataset(spark, cfg):
    """Pipeline completo de preparacion sobre el periodo actual.

    Devuelve (df, vars_info) donde vars_info es el dict de clasificar_variables.
    """
    seg = cfg['segmentos']
    col_seg = cfg['data']['col_segmento']

    print(f"[DATOS] Cargando periodo actual {cfg['data']['periodo_actual']}...")
    df = cargar_periodo(spark, cfg, cfg['data']['periodo_actual'])
    assert len(df) > 0, '[DATOS] El periodo actual no tiene datos. Revise data.* en el config.'

    print(f'  Clientes: {len(df):,} | Columnas: {len(df.columns)}')
    for s in seg['orden']:
        n = int((df[col_seg] == s).sum())
        print(f"    {seg['nombres_cortos'][s]:12s}: {n:>8,} ({n / len(df) * 100:.1f}%)")

    # y_global para target encoding: pertenecer a la mitad superior de segmentos
    mitad_sup = seg['orden'][len(seg['orden']) // 2:]
    y_global = df[col_seg].isin(mitad_sup).astype(int)

    nuevas_acc, nuevos_ctrl = codificar_categoricas(df, cfg, y_global)
    print(f'  Encoded: +{len(nuevas_acc)} accionables, +{len(nuevos_ctrl)} controles')

    vars_info = clasificar_variables(df, cfg, nuevas_acc, nuevos_ctrl)
    dims = pd.Series(vars_info['dimension_map']).value_counts()
    print(f"  Accionables: {len(vars_info['accionables'])} | Controles: {len(vars_info['controles'])}")
    print(f'  Dimensiones de negocio: {len(dims)} -> {dict(dims)}')

    return df, vars_info


def preparar_par(df, cfg, seg_origen, seg_destino, accionables, max_sample=None, random_state=17):
    """Prepara el dataset binario de un par de transicion.

    Y=1 si el cliente esta en seg_destino (el pipeline aprende que separa a
    los que YA estan arriba de los que estan abajo).
    Devuelve (df_par, y) o (None, None) si no hay muestra suficiente.
    """
    col_seg = cfg['data']['col_segmento']
    df_par = df[df[col_seg].isin([seg_origen, seg_destino])].copy()
    if len(df_par) < 100 or df_par[col_seg].nunique() < 2:
        return None, None
    if max_sample and len(df_par) > max_sample:
        df_par = df_par.sample(max_sample, random_state=random_state)
    y = (df_par[col_seg] == seg_destino).astype(int)
    return df_par, y
