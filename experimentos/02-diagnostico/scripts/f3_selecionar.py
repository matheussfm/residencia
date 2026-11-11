"""Fase 3: sorteia os áudios cujas representações internas serão extraídas.

Roda no servidor (dentro do container), porque precisa das listas com os caminhos dos áudios,
geradas no Experimento 01 e não versionadas (experimentos/01-linha-de-base/listas/*/lista.tsv).

Amostra, com semente fixa, balanceada por classe:
  - df21_eval:   por origem (asvspoof = VCTK, vcc2018, vcc2020), --n-df21 genuínos e falsos de cada;
  - df21_hidden: --n-hidden genuínos e falsos (sem o silêncio das pontas);
  - itw:         --n-itw genuínos e falsos;
  - brs:         todos os 1.320 genuínos do teste e 1.320 falsos (264 por TTS).

Grava em listas/fase3/: lista.tsv (id, caminho), no formato do inferir.py, e amostra.tsv (id,
cenario, id_original, rotulo, locutor, grupo, pontuacao_exp01). O id é "<cenario>::<id original>".

Exemplo (no servidor):
    bash experimentos/01-linha-de-base/scripts/em_docker.sh nenhuma \
        python ../../02-diagnostico/scripts/f3_selecionar.py
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
from _comum import escrever_tsv


def amostrar(c, m, n, rng):
    idx = np.flatnonzero(m)
    return idx if idx.size <= n else np.sort(rng.choice(idx, n, replace=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--listas', default=D.LIS01, help='pasta com as listas do Experimento 01')
    ap.add_argument('--saida', default=os.path.join(D.EXP02, 'listas', 'fase3'))
    ap.add_argument('--n-df21', type=int, default=500)
    ap.add_argument('--n-hidden', type=int, default=1000)
    ap.add_argument('--n-itw', type=int, default=1500)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()
    D.garantir_fora_third_party(args.saida)
    rng = np.random.default_rng(args.semente)

    caminhos = {}
    for pasta, cen in (('asvspoof21df', 'df21'), ('in_the_wild', 'itw'), ('brspeech_test_hf', 'brs')):
        caminhos[cen] = {r['id']: r['caminho'] for r in D.ler_tsv(os.path.join(args.listas, pasta, 'lista.tsv'))}

    escolhas = []  # (cenario, Cenario, indices, fonte_caminhos)
    ev = D.carregar_df21()
    for o in ('asvspoof', 'vcc2018', 'vcc2020'):
        for r in (1, 0):
            escolhas.append(('df21_eval', ev, amostrar(ev, (ev.meta['origem'] == o) & (ev.rotulo == r), args.n_df21, rng), 'df21'))
    hid = D.carregar_df21(fase='hidden')
    for r in (1, 0):
        escolhas.append(('df21_hidden', hid, amostrar(hid, hid.rotulo == r, args.n_hidden, rng), 'df21'))
    itw = D.carregar_itw()
    for r in (1, 0):
        escolhas.append(('itw', itw, amostrar(itw, itw.rotulo == r, args.n_itw, rng), 'itw'))
    brs = D.carregar_brs()
    escolhas.append(('brs', brs, np.flatnonzero(brs.rotulo == 1), 'brs'))
    for tts in sorted(set(brs.meta['grupo']) - {'bonafide'}):
        escolhas.append(('brs', brs, amostrar(brs, brs.meta['grupo'] == tts, 264, rng), 'brs'))

    lista, amostra = [], []
    for cen, c, idx, fonte in escolhas:
        grupos = c.meta['origem'] if 'origem' in c.meta else c.meta.get('grupo')
        for i in idx:
            id_orig = str(c.ids[i])
            grupo = grupos[i] if grupos is not None else '-'
            uid = f'{cen}::{id_orig}'
            lista.append((uid, caminhos[fonte][id_orig]))
            amostra.append((uid, cen, id_orig, 'bonafide' if c.rotulo[i] else 'spoof', c.meta['locutor'][i], grupo,
                            f'{c.pont[i]:.6f}'))
    if len({u for u, _ in lista}) != len(lista):
        D.erro('ids repetidos na amostra')
    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'amostra.tsv'),
                 ['id', 'cenario', 'id_original', 'rotulo', 'locutor', 'grupo', 'pontuacao_exp01'], amostra)
    from collections import Counter
    cont = Counter((a[1], a[3]) for a in amostra)
    print(f'{len(lista)} áudios:', dict(sorted(cont.items())))


if __name__ == '__main__':
    main()
