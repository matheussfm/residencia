"""Distribuição das durações dos áudios de uma lista, por classe e, opcionalmente, por grupo.

Considera só os áudios que têm rótulo (no 2021 DF, a fase eval). Lê só o cabeçalho de cada arquivo (soundfile.info), sem carregar o áudio. Mostra quantos
áudios são mais curtos que o corte do modelo (64.600 amostras a 16 kHz, ~4,04 s), e portanto
recebem preenchimento, e quantos são mais longos, e portanto são cortados.

Exemplo (dentro do container):
    python duracoes.py --lista ../listas/in_the_wild/lista.tsv --rotulos ../listas/in_the_wild/rotulos.tsv \
        --saida ../resultados/in_the_wild/duracoes.json
"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

import numpy as np
import soundfile

from _comum import erro, garantir_fora_third_party, ler_tsv

CORTE_S = 64600 / 16000


def estatisticas(d):
    return {
        'n': int(d.size),
        'min': float(d.min()),
        'p5': float(np.percentile(d, 5)),
        'p25': float(np.percentile(d, 25)),
        'mediana': float(np.median(d)),
        'media': float(d.mean()),
        'p75': float(np.percentile(d, 75)),
        'p95': float(np.percentile(d, 95)),
        'max': float(d.max()),
        'frac_curtos': float((d < CORTE_S).mean()),
        'frac_longos': float((d > CORTE_S).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lista', required=True)
    ap.add_argument('--rotulos', required=True)
    ap.add_argument('--por-grupo', action='store_true', help='também por valor da coluna grupo')
    ap.add_argument('--saida', help='JSON com as estatísticas')
    args = ap.parse_args()
    if args.saida:
        garantir_fora_third_party(args.saida)

    rot = {r['id']: r for r in ler_tsv(args.rotulos)}
    dur = {}
    for n, r in enumerate(ler_tsv(args.lista), 1):
        if r['id'] not in rot:
            continue
        info = soundfile.info(r['caminho'])
        dur[r['id']] = info.frames / info.samplerate
        if n % 50000 == 0:
            print(f'  {n} arquivos lidos', flush=True)
    if not dur:
        erro('nenhum áudio da lista tem rótulo')

    conjuntos = {'todos': list(dur)}
    for id_, r in rot.items():
        if id_ in dur:
            conjuntos.setdefault(r['rotulo'], []).append(id_)
            if args.por_grupo and r.get('grupo'):
                conjuntos.setdefault(f"grupo:{r['grupo']}", []).append(id_)
    res = {k: estatisticas(np.array([dur[i] for i in v])) for k, v in conjuntos.items()}

    if args.saida:
        with open(args.saida, 'w', encoding='utf-8') as f:
            json.dump({'lista': args.lista, 'corte_s': CORTE_S, 'estatisticas': res}, f, indent=2, ensure_ascii=False)

    print(f'\n==== DURAÇÕES (s) — corte do modelo: {CORTE_S:.2f} s ====')
    print(f"{'conjunto':22s} {'n':>7s} {'min':>6s} {'p5':>6s} {'mediana':>7s} {'média':>6s} {'p95':>6s} {'max':>7s} {'<corte':>7s} {'>corte':>7s}")
    for k, e in res.items():
        print(f"{k:22s} {e['n']:7d} {e['min']:6.2f} {e['p5']:6.2f} {e['mediana']:7.2f} {e['media']:6.2f} "
              f"{e['p95']:6.2f} {e['max']:7.2f} {100 * e['frac_curtos']:6.1f}% {100 * e['frac_longos']:6.1f}%")


if __name__ == '__main__':
    main()
