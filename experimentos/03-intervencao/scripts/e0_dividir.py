"""Etapa 0: divisão fixa, por locutor, da amostra df21_eval da Fase 3 em treino / validação / teste.

A amostra (Exp. 02, f3_selecionar.py) tem 3.000 trials do 2021 DF eval: 500 por classe de cada
origem (VCTK/ASVspoof, VCC 2018, VCC 2020), de 88 locutores. Muitos locutores têm só uma classe
(por exemplo, 14 dos 62 locutores VCTK só têm genuínos). Por isso a divisão:
  - é feita dentro de cada origem, para que as três partições tenham as três origens;
  - percorre os locutores numa ordem sorteada (semente 0) e põe cada um na partição com o maior
    déficit relativo em relação à meta (60/20/20 dos áudios), medido nas duas classes, de modo que
    as partições fiquem equilibradas em genuínos e falsos;
  - nenhum locutor aparece em mais de uma partição.

Também registra a sobreposição com a hidden track: os 46 locutores da hidden track da amostra estão
todos no df21_eval. A avaliação na hidden track é feita de duas formas (README, seção 4): completa e
só com os locutores da partição de teste.

Saídas: listas/divisao_df21_eval.tsv (id, locutor, origem, rotulo, particao) e
        listas/divisao_df21_eval.json (+ .meta.json).
"""
import argparse
import collections
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _comum import escrever_tsv
from _interv import D

METAS = {'treino': 0.6, 'validacao': 0.2, 'teste': 0.2}


def dividir(locutores, contagem, semente):
    """locutores: lista; contagem[l] = (n_genuinos, n_falsos). Devolve dict locutor -> partição."""
    total = np.array([sum(contagem[l][j] for l in locutores) for j in (0, 1)], float)
    alvo = {p: METAS[p] * total for p in METAS}
    atual = {p: np.zeros(2) for p in METAS}
    rng = np.random.default_rng(semente)
    ordem = [locutores[i] for i in rng.permutation(len(locutores))]
    out = {}
    for l in ordem:
        n = np.array(contagem[l], float)
        # déficit relativo ponderado pelas classes que o locutor tem
        def deficit(p):
            return float(np.sum(n * (alvo[p] - atual[p]) / np.maximum(alvo[p], 1)))
        p = max(METAS, key=deficit)
        out[l] = p
        atual[p] += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()

    am = I.ler_amostra()
    ev = [r for r in am.values() if r['cenario'] == 'df21_eval']
    origem = {}
    contagem = collections.defaultdict(lambda: [0, 0])
    for r in ev:
        if origem.setdefault(r['locutor'], r['grupo']) != r['grupo']:
            D.erro(f"locutor {r['locutor']} em mais de uma origem")
        contagem[r['locutor']][0 if r['rotulo'] == 'bonafide' else 1] += 1

    part = {}
    for k, o in enumerate(sorted(set(origem.values()))):
        locs = sorted(l for l in origem if origem[l] == o)
        part.update(dividir(locs, contagem, [args.semente, k]))

    linhas = sorted((r['id'], r['locutor'], r['grupo'], r['rotulo'], part[r['locutor']]) for r in ev)
    tsv = I.DIVISAO
    escrever_tsv(tsv, ['id', 'locutor', 'origem', 'rotulo', 'particao'], linhas)

    resumo = {}
    for p in I.PARTICOES:
        sel = [x for x in linhas if x[4] == p]
        resumo[p] = {'locutores': len({x[1] for x in sel}),
                     'por_origem': {o: {'genuinos': sum(x[2] == o and x[3] == 'bonafide' for x in sel),
                                        'falsos': sum(x[2] == o and x[3] == 'spoof' for x in sel),
                                        'locutores': len({x[1] for x in sel if x[2] == o})}
                                    for o in sorted(set(origem.values()))},
                     'genuinos': sum(x[3] == 'bonafide' for x in sel), 'falsos': sum(x[3] == 'spoof' for x in sel)}
    hid = [r for r in am.values() if r['cenario'] == 'df21_hidden']
    hid_locs = {r['locutor'] for r in hid}
    sobre = {'locutores_hidden': len(hid_locs),
             'locutores_hidden_tambem_no_df21_eval': len(hid_locs & set(origem)),
             'por_particao': {p: {'locutores': len({l for l in hid_locs if part.get(l) == p}),
                                  'genuinos': sum(r['rotulo'] == 'bonafide' and part.get(r['locutor']) == p for r in hid),
                                  'falsos': sum(r['rotulo'] == 'spoof' and part.get(r['locutor']) == p for r in hid)}
                              for p in I.PARTICOES}}
    res = {'descricao': 'divisão por locutor da amostra df21_eval da Fase 3, dentro de cada origem, metas 60/20/20 '
                        'dos áudios nas duas classes', 'metas': METAS, 'particoes': resumo,
           'hidden_track': sobre, 'tsv': os.path.relpath(tsv, D.RAIZ_REPO).replace(os.sep, '/'),
           'tsv_sha256': D.sha256(tsv)}
    I.salvar_json(os.path.join(I.EXP03, 'listas', 'divisao_df21_eval.json'), res, __file__, {'semente': args.semente})

    for p in I.PARTICOES:
        r = resumo[p]
        print(f"{p:9s} {r['locutores']:2d} locutores, {r['genuinos']:4d} gen. / {r['falsos']:4d} falsos  "
              + '  '.join(f"{o}: {v['locutores']} loc. {v['genuinos']}/{v['falsos']}" for o, v in r['por_origem'].items()))
    print('hidden track por partição:', sobre['por_particao'])


if __name__ == '__main__':
    main()
