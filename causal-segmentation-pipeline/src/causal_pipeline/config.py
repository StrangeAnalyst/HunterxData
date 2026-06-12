"""Carga y validacion del archivo de configuracion YAML.

El objetivo es que el pipeline falle RAPIDO y con un mensaje claro si la
configuracion esta incompleta, en vez de fallar a mitad de una corrida larga.
"""

import yaml


SECCIONES_REQUERIDAS = [
    'caso_uso', 'data', 'segmentos', 'variables', 'modelado',
    'causal', 'backtesting', 'cuadrantes', 'estrategia', 'salidas',
]

CAMPOS_DATA = ['tabla_features', 'col_id', 'col_segmento', 'col_periodo',
               'periodo_actual', 'periodos_historicos']


def cargar_config(ruta):
    """Lee el YAML, valida y devuelve el dict de configuracion enriquecido."""
    with open(ruta, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    faltantes = [s for s in SECCIONES_REQUERIDAS if s not in cfg]
    if faltantes:
        raise ValueError(f'[CONFIG] Secciones faltantes en {ruta}: {faltantes}')

    for campo in CAMPOS_DATA:
        if campo not in cfg['data']:
            raise ValueError(f"[CONFIG] data.{campo} es requerido")

    seg = cfg['segmentos']
    if len(seg.get('orden', [])) < 2:
        raise ValueError('[CONFIG] segmentos.orden necesita al menos 2 segmentos')
    for s in seg['orden']:
        if s not in seg.get('nombres_cortos', {}):
            seg.setdefault('nombres_cortos', {})[s] = s.replace(' ', '')[:12]
        if s not in seg.get('ltv_anual', {}):
            raise ValueError(f"[CONFIG] segmentos.ltv_anual no define '{s}'")

    if cfg['data']['periodo_actual'] not in cfg['data']['periodos_historicos']:
        cfg['data']['periodos_historicos'].append(cfg['data']['periodo_actual'])
    cfg['data']['periodos_historicos'] = sorted(cfg['data']['periodos_historicos'])

    bt = cfg['backtesting']
    if bt.get('activo', True) and len(cfg['data']['periodos_historicos']) < bt.get('min_periodos', 3):
        raise ValueError(
            '[CONFIG] backtesting.activo=true requiere al menos '
            f"{bt.get('min_periodos', 3)} periodos en data.periodos_historicos. "
            'Agregue mas periodos o desactive el backtesting.'
        )

    # Derivados de uso frecuente
    orden = seg['orden']
    cortos = seg['nombres_cortos']
    cfg['_derivados'] = {
        'pares_evolucion': [(orden[i], orden[i + 1]) for i in range(len(orden) - 1)],
        'pares_involucion': [(orden[i + 1], orden[i]) for i in range(len(orden) - 1)],
        'nombre_par': lambda a, b: f'{cortos[a]}_to_{cortos[b]}',
    }
    return cfg


def nombre_par(cfg, seg_origen, seg_destino):
    """Nombre canonico de un par de transicion, ej. 'Growing_to_HPU'."""
    cortos = cfg['segmentos']['nombres_cortos']
    return f'{cortos[seg_origen]}_to_{cortos[seg_destino]}'


def tabla_salida(cfg, nombre_base):
    """Nombre completo de una tabla de salida, ej. catalogo.esquema.base_sufijo."""
    s = cfg['salidas']
    return f"{s['prefijo_tablas']}.{nombre_base}_{s['sufijo']}"


def resumen_config(cfg):
    """Imprime un resumen legible de la configuracion cargada."""
    d, seg = cfg['data'], cfg['segmentos']
    cortos = seg['nombres_cortos']
    print('=' * 72)
    print(f"PIPELINE CAUSAL — {cfg['caso_uso']}")
    print('=' * 72)
    print(f"  Tabla: {d['tabla_features']}")
    print(f"  Periodo actual: {d['periodo_actual']} | Historia: {d['periodos_historicos']}")
    print(f"  Segmentos: {[cortos[s] for s in seg['orden']]}")
    print(f"  Filtros: {d.get('filtros', {})}")
    print(f"  Backtesting: {'ACTIVO' if cfg['backtesting'].get('activo') else 'desactivado'}")
    print(f"  Involucion: {'ACTIVA' if cfg.get('involucion', {}).get('activo') else 'desactivada'}")
    print(f"  Salidas: {cfg['salidas']['prefijo_tablas']}.*_{cfg['salidas']['sufijo']}")
    print('=' * 72)
