"""Genera un bundle .txt con todos los archivos del proyecto.

Util para compartir el pipeline completo por correo (un solo adjunto de texto).
Cada archivo va delimitado por encabezados claros para poder reconstruirlo.

Uso:
    python tools/build_bundle.py [salida.txt]
"""

import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

ARCHIVOS = [
    'README.md',
    'config/_template.yaml',
    'config/pyme_personas.yaml',
    'src/causal_pipeline/__init__.py',
    'src/causal_pipeline/config.py',
    'src/causal_pipeline/data.py',
    'src/causal_pipeline/integrity.py',
    'src/causal_pipeline/discovery.py',
    'src/causal_pipeline/causal.py',
    'src/causal_pipeline/backtesting.py',
    'src/causal_pipeline/quadrants.py',
    'src/causal_pipeline/strategy.py',
    'src/causal_pipeline/outputs.py',
    'notebooks/01_pipeline_causal_estandar.py',
    'tools/build_bundle.py',
]

BANNER = '=' * 78


def construir(destino):
    partes = [
        BANNER,
        'BUNDLE — PIPELINE CAUSAL ESTANDARIZADO DE SEGMENTACION',
        f'Generado: {date.today().isoformat()} | Archivos: {len(ARCHIVOS)}',
        '',
        'Para reconstruir el proyecto: crear cada archivo con la ruta indicada',
        'en su encabezado ">>> ARCHIVO:". El README.md explica todo el uso.',
        BANNER,
        '',
    ]
    for rel in ARCHIVOS:
        ruta = RAIZ / rel
        contenido = ruta.read_text(encoding='utf-8')
        partes += [BANNER, f'>>> ARCHIVO: {rel}', BANNER, contenido, '']
    Path(destino).write_text('\n'.join(partes), encoding='utf-8')
    print(f'[OK] Bundle generado: {destino} ({Path(destino).stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    salida = sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / 'bundle_pipeline_causal.txt')
    construir(salida)
